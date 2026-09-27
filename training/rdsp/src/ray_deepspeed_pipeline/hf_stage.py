"""Pipeline stages of a Hugging Face model that run the model's own forward.

A stage is a copy of the whole module tree in which everything it does not
own is swapped out: blocks outside its range pass their input through, and
other modules holding parameters it does not own become stand-ins. Calling the
model then runs its own setup code (masks, positions, rotary tables, vision
merge) and only this stage's blocks, so no architecture needs a hand-written
stage forward.

Besides the hidden state, a boundary carries the tensor arguments the model
passes to every downstream block (rotary tables, masks, ...): they may depend
on inputs only stage 0 sees (Qwen3-VL's image positions), and may differ per
block type (Gemma3's local and global rotary tables). Stage 0's loop still
calls the pass-through blocks after its range, so it records each one's
arguments; blocks sharing the same tensors are sent as one group.
"""

import copy
import json
import os

import torch
import torch.nn as nn

from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.partition import (
    attach_tp_plan,
    find_block_list,
    vision_injection_depth,
)


class _PassThrough(nn.Module):
    """A block another stage owns. Returns its input shaped like the real
    blocks' output: a tensor, or a tuple whose first item is the hidden state
    (Bloom). `form` is shared with the stage, which learns it from its own
    blocks; until then a 1-tuple, which the model's loop either indexes or
    hands on unchanged until the stage's first block replaces it."""

    def __init__(self, form: list):
        super().__init__()
        self.form = form  # [None | 0 for a tensor | tuple length]

    def forward(self, hidden_states, *args, **kwargs):
        n = self.form[0]
        if n == 0:
            return hidden_states
        return (hidden_states, *[None] * ((n or 1) - 1))


class _Unowned(nn.Module):
    """A module whose parameters another stage owns. Returns its input, or
    zeros for an embedding (a later stage's setup may still add position
    embeddings). Keeps meta-device stand-ins of its parameters, outside the
    parameter list, for model code that reads their dtype or shape."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self._embedding_dim = getattr(module, "embedding_dim", None)
        for name, p in module.named_parameters(recurse=False):
            object.__setattr__(self, name, torch.empty(p.shape, dtype=p.dtype, device="meta"))

    def forward(self, x, *args, **kwargs):
        if self._embedding_dim is None:
            return x
        dtype = getattr(self, "weight", x).dtype
        return torch.zeros(*x.shape, self._embedding_dim, dtype=dtype, device=x.device)


def _block_tensors(args: tuple, kwargs: dict) -> dict:
    """A block call's tensor arguments besides the hidden state: positional
    ones as #i, keyword ones by name, tuples of tensors as name.i."""
    out = {f"#{i}": a for i, a in enumerate(args) if i and torch.is_tensor(a)}
    for name, value in kwargs.items():
        if torch.is_tensor(value):
            out[name] = value
        elif isinstance(value, tuple) and value and all(torch.is_tensor(v) for v in value):
            out.update({f"{name}.{i}": v for i, v in enumerate(value)})
    return out


def _with_block_tensors(args: tuple, kwargs: dict, tensors: dict) -> tuple[tuple, dict]:
    """(args, kwargs) with the entries of a _block_tensors() dict put back in."""
    args, kwargs, tuples = list(args), dict(kwargs), {}
    for key, value in tensors.items():
        name, _, index = key.partition(".")
        if key.startswith("#"):
            args[int(key[1:])] = value
        elif index:
            tuples.setdefault(name, {})[int(index)] = value
        else:
            kwargs[name] = value
    for name, parts in tuples.items():
        kwargs[name] = tuple(parts[i] for i in range(len(parts)))
    return tuple(args), kwargs


_FORM_KEY = "@block_form"


def _pack(per_block: dict, rows: int, block_form: int | None) -> dict:
    """{block: {arg: tensor}} as boundary extras keyed "6+7+9/arg": blocks
    receiving the same tensors share one entry. Tensors are laid out rows
    first like the hidden state, so the boundary code can split them per
    rank; a broadcast row is expanded. Tuple-returning blocks also send their
    tuple length, which the next stage's pass-throughs need before any of its
    own blocks has run."""
    groups = {}
    for block, tensors in sorted(per_block.items()):
        signature = tuple(sorted((k, id(v)) for k, v in tensors.items()))
        groups.setdefault(signature, ([], tensors))[0].append(str(block))
    out = {}
    for blocks, tensors in groups.values():
        for name, t in tensors.items():
            if t.dim() >= 1 and t.shape[0] == 1 and rows > 1:
                t = t.expand(rows, *t.shape[1:])
            # rows first, possibly folded with another dim (Bloom: rows x heads)
            if t.dim() < 2 or t.shape[0] % rows:
                raise ValidationError(
                    f"block argument {name!r} of shape {tuple(t.shape)} has no leading "
                    f"row dimension of size {rows}, so it cannot cross a stage boundary")
            out[f"{'+'.join(blocks)}/{name}"] = t.detach()
    if block_form:
        out[_FORM_KEY] = torch.full((rows, 1), block_form, dtype=torch.int64)
    return out


def _unpack(extras: dict) -> dict:
    per_block = {}
    for key, t in extras.items():
        if key == _FORM_KEY:
            continue
        blocks, _, name = key.partition("/")
        for block in blocks.split("+"):
            per_block.setdefault(int(block), {})[name] = t
    return per_block


class HFModelStage(nn.Module):
    """forward(x=None, position_ids=None, **kwargs). Stage 0: the model's
    inputs, as x (input ids) or as keyword tensors. Later stages: x is the
    hidden state and the keyword tensors are the block arguments from
    upstream (their keys contain "/"). Returns the logits on the last stage,
    otherwise (hidden, extras) for the next stage."""

    def __init__(self, model, blocks_name: str, block_start: int, block_stop: int,
                 is_first: bool, is_last: bool, block_form: list):
        super().__init__()
        self.model = model
        self._block_form = block_form  # shared with the _PassThrough blocks
        self._blocks_name = blocks_name
        self._local = (block_start, block_stop)
        self.is_first, self.is_last = is_first, is_last
        self._hooked = False
        self._input = self._received = self._hidden = None
        self._downstream = {}

    def local_blocks(self):
        start, stop = self._local
        return [self.model.get_submodule(self._blocks_name)[i] for i in range(start, stop)]

    def _install_hooks(self):
        # installed on first use, so the pickled stage carries no bound hooks
        if self._hooked:
            return
        blocks = self.model.get_submodule(self._blocks_name)
        start, stop = self._local
        for i in range(start, len(blocks)):
            blocks[i].register_forward_pre_hook(self._before_block(i), with_kwargs=True)
        blocks[start].register_forward_hook(self._learn_block_form)
        self._hooked = True

    def _learn_block_form(self, block, args, out):
        self._block_form[0] = len(out) if isinstance(out, tuple) else 0

    def _before_block(self, index: int):
        start, stop = self._local

        def hook(block, args, kwargs):
            args, kwargs = _with_block_tensors(args, kwargs, self._received.get(index, {}))
            if index == start and not self.is_first:
                # the model's setup may transform inputs_embeds (e.g. scale it)
                if args:
                    args = (self._input, *args[1:])
                else:
                    kwargs["hidden_states"] = self._input
            if index == stop:
                # what the loop hands the next block, so it includes the
                # model's own work between blocks (Qwen3-VL adds vision
                # features after block 0 and 1) but not code after the last
                # block (norm, head, logit soft-capping), which is not ours
                self._hidden = args[0] if args else kwargs["hidden_states"]
            if index >= stop:
                self._downstream[index] = _block_tensors(args, kwargs)
            return args, kwargs
        return hook

    def _setup_input(self, x):
        """What the model's setup code gets as inputs_embeds on a later stage:
        the hidden state itself when it is embedding-shaped, else zeros of
        that shape (GLM-5.3's hidden state has 4 streams). Blocks never see
        it: the first local block gets the hidden state from the hook."""
        embed = self.model.get_input_embeddings()
        dim = getattr(embed, "_embedding_dim", None) or getattr(embed, "embedding_dim", None)
        if dim is None or tuple(x.shape[2:]) == (dim,):
            return x
        return x.new_zeros(*x.shape[:2], dim)

    def forward(self, x=None, position_ids=None, **kwargs):
        """position_ids: a sequence shard's global positions (Ulysses SP)."""
        self._install_hooks()
        extras = {k: v for k, v in kwargs.items() if "/" in k or k == _FORM_KEY}
        if x is None:
            x = {k: v for k, v in kwargs.items() if k not in extras}
        if self._block_form[0] is None and _FORM_KEY in extras:
            self._block_form[0] = int(extras[_FORM_KEY][0, 0])  # once: it syncs
        self._input, self._received, self._downstream = x, _unpack(extras), {}
        try:
            kw = {"use_cache": False}
            if position_ids is not None:
                kw["position_ids"] = position_ids
            if not self.is_first:
                out = self.model(inputs_embeds=self._setup_input(x), **kw)
            elif isinstance(x, dict):
                out = self.model(**x, **kw)
            else:
                out = self.model(input_ids=x, **kw)
            if self.is_last:
                return out.logits if hasattr(out, "logits") else out[0]
            hidden = self._hidden
            return hidden, _pack(self._downstream, hidden.shape[0], self._block_form[0])
        finally:
            self._input = self._received = self._hidden = None
            self._downstream = {}


def build_hf_stage(model: nn.Module, block_start: int, block_stop: int,
                   parameter_names: tuple[str, ...]) -> HFModelStage:
    """One stage of an HF model as an HFModelStage. Modules it does not own
    are never copied, so building from a full model costs only the stage's
    own share of memory."""
    blocks_name, blocks = find_block_list(model)
    depth = vision_injection_depth(model)
    if block_start == 0 and block_stop < depth:
        raise ValidationError(
            f"{type(model).__name__} adds vision features inside blocks 0-{depth - 1}, "
            f"which only the first stage can do: put the first cut at {depth} or later")
    owned = set(parameter_names)
    block_form = [None]
    memo = {}
    for i, block in enumerate(blocks):
        if not block_start <= i < block_stop:
            memo[id(block)] = _PassThrough(block_form)
    for name, module in model.named_modules():
        direct = [f"{name}.{p}" if name else p for p, _ in module.named_parameters(recurse=False)]
        if direct and not owned.intersection(direct):
            memo.setdefault(id(module), _Unowned(module))
    stage_model = copy.deepcopy(model, memo)
    kept = {n for n, _ in stage_model.named_parameters(remove_duplicate=False)}
    if kept != owned:
        raise ValidationError(
            f"stage over blocks [{block_start}, {block_stop}) cannot be cut out cleanly: "
            f"a module mixes parameters of different stages "
            f"({sorted(kept ^ owned)[:3]}...)")
    stage = HFModelStage(stage_model, blocks_name, block_start, block_stop,
                         is_first=block_start == 0, is_last=block_stop == len(blocks),
                         block_form=block_form)
    # what intra-stage parallelism reads: the AutoTP plan, head counts
    # (Ulysses) and MoE settings (AutoEP), all on the text model's config
    config = getattr(model, "config", None)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    plan = getattr(text_config, "base_model_tp_plan", None) or _standard_tp_plan(model, blocks)
    attach_tp_plan(stage, plan)
    stage._rdsp_hf_config = text_config
    return stage


# projection names shared by most decoder families, and how Megatron-style
# tensor parallelism splits them
_STANDARD_TP = {"self_attn.q_proj": "colwise", "self_attn.k_proj": "colwise",
                "self_attn.v_proj": "colwise", "self_attn.o_proj": "rowwise",
                "mlp.gate_proj": "colwise", "mlp.up_proj": "colwise",
                "mlp.down_proj": "rowwise",
                # per-head norms: replicated, but each rank sees only its heads
                "self_attn.q_norm": "replicated_with_grad_allreduce",
                "self_attn.k_norm": "replicated_with_grad_allreduce"}


def _standard_tp_plan(model: nn.Module, blocks) -> dict:
    """A TP plan for models whose config has none (most do not, e.g.
    Qwen3-VL): the standard projections that every block has."""
    names = [{n for n, _ in block.named_modules()} for block in blocks]
    return {f"layers.*.{name}": style for name, style in _STANDARD_TP.items()
            if all(name in block_names for block_names in names)}


def _checkpoint_files(weights_dir: str) -> dict:
    """Parameter name -> safetensors file, for an HF checkpoint directory."""
    index = os.path.join(weights_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return {k: os.path.join(weights_dir, v) for k, v in json.load(f)["weight_map"].items()}
    single = os.path.join(weights_dir, "model.safetensors")
    if not os.path.exists(single):
        raise ValidationError(f"no safetensors checkpoint in {weights_dir!r}")
    from safetensors import safe_open
    with safe_open(single, "pt") as f:
        return {k: single for k in f.keys()}


def load_meta_parameters(stage: nn.Module, weights_dir: str | None) -> None:
    """Replace every parameter still on the meta device by its value from the
    HF checkpoint in weights_dir; only this stage's tensors are read."""
    meta = [(n, p) for n, p in stage.named_parameters() if p.is_meta]
    if not meta:
        return
    if weights_dir is None:
        raise ValidationError(
            f"{len(meta)} parameters are on the meta device (e.g. {meta[0][0]}); pass "
            f"weights=<HF checkpoint dir> to rdsp.initialize() to load them per stage")
    if not isinstance(stage, HFModelStage):
        raise ValidationError("weights= needs a Hugging Face model (the HF stage builder)")
    if any(b.is_meta for b in stage.buffers()):
        raise ValidationError(
            "buffers are on the meta device too; build the model skeleton with "
            "accelerate.init_empty_weights(), which keeps buffers real")
    files = _checkpoint_files(weights_dir)
    # a tied checkpoint stores only one of each tied pair (Qwen3-VL-2B has no
    # lm_head); the untied copy is read from the tensor it was tied to
    tied = getattr(stage.model, "_tied_weights_keys", None)
    tied = tied if isinstance(tied, dict) else {}
    by_file = {}
    for name, param in meta:
        key = name.removeprefix("model.")
        if key not in files:
            key = tied.get(key, key)
        if key not in files:
            raise ValidationError(f"{key!r} is not in the checkpoint at {weights_dir!r}")
        by_file.setdefault(files[key], []).append((name, key, param))
    from safetensors import safe_open
    for path, entries in by_file.items():
        with safe_open(path, "pt") as f:
            for name, key, param in entries:
                module_name, _, leaf = name.rpartition(".")
                value = f.get_tensor(key).to(param.dtype)
                stage.get_submodule(module_name)._parameters[leaf] = nn.Parameter(
                    value, requires_grad=param.requires_grad)
