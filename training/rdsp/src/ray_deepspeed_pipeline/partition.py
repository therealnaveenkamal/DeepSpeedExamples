"""Deterministic contiguous model partitioning and stage-module builders.

Policies: explicit cuts, uniform transformer blocks, uniform sequential; no
automatic search. Stages are described by fully qualified parameter names,
never driver parameter objects.
"""

import functools
from dataclasses import dataclass

import torch.nn as nn

from ray_deepspeed_pipeline.config import (
    ExplicitCuts,
    UniformSequential,
    UniformTransformerBlocks,
)
from ray_deepspeed_pipeline.errors import ValidationError


@dataclass(frozen=True)
class StagePartition:
    index: int
    parameter_names: tuple[str, ...]
    block_start: int
    block_stop: int


def find_block_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """(name, module) of the longest ModuleList whose children share one
    class. Raises ValidationError if there is none, or a tie."""
    candidates = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, nn.ModuleList) and len(m) >= 2
        and len({type(c) for c in m}) == 1
    ]
    if not candidates:
        raise ValidationError(
            "no transformer block list found: expected a ModuleList of >=2 "
            "same-class blocks; use ExplicitCuts on a model this heuristic "
            "cannot read")
    longest = max(len(m) for _, m in candidates)
    winners = [(n, m) for n, m in candidates if len(m) == longest]
    if len(winners) > 1:
        raise ValidationError(
            f"ambiguous block lists {[n for n, _ in winners]}: "
            f"partitioning refuses to guess")
    return winners[0]


def _cuts_for(policy, n_blocks: int, stages: int) -> list[int]:
    if isinstance(policy, ExplicitCuts):
        cuts = list(policy.cuts)
        if len(cuts) != stages - 1:
            raise ValidationError(
                f"ExplicitCuts needs {stages - 1} cuts for {stages} stages, "
                f"got {len(cuts)}")
    elif isinstance(policy, (UniformTransformerBlocks, UniformSequential)):
        base, extra = divmod(n_blocks, stages)
        sizes = [base + (1 if i < extra else 0) for i in range(stages)]
        cuts, acc = [], 0
        for size in sizes[:-1]:
            acc += size
            cuts.append(acc)
    else:
        raise ValidationError(f"unknown partition policy {type(policy)}")

    if any(c <= 0 or c >= n_blocks for c in cuts):
        raise ValidationError(f"cuts must lie in (0, {n_blocks}), got {cuts}")
    if any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValidationError(f"cuts must be strictly increasing, got {cuts}")
    return cuts


def partition_parameters(model: nn.Module, policy, stages: int) -> tuple[StagePartition, ...]:
    """Assign every parameter name to exactly one stage. Blocks map by cut
    range; parameters before the block list go to the first stage, those
    after it to the last. Rejects parameters tied across stages: each stage
    would hold its own copy and see only part of the gradient."""
    if stages < 1:
        raise ValidationError(f"stages must be >= 1, got {stages}")
    blocks_name, blocks = find_block_list(model)
    cuts = _cuts_for(policy, len(blocks), stages) if stages > 1 else []
    bounds = [0] + cuts + [len(blocks)]

    def stage_of_block(b: int) -> int:
        for s in range(stages):
            if bounds[s] <= b < bounds[s + 1]:
                return s
        raise AssertionError(b)

    # declaration order stands in for topology on the sequential models
    # supported; remove_duplicate=False keeps tied names visible
    named = list(model.named_parameters(remove_duplicate=False))
    prefix = blocks_name + "."
    block_positions = [i for i, (n, _) in enumerate(named) if n.startswith(prefix)]
    if not block_positions:
        raise ValidationError(f"block list {blocks_name} has no parameters")
    first_block, last_block = block_positions[0], block_positions[-1]

    assignment: dict[str, int] = {}
    by_id: dict[int, list[str]] = {}
    for i, (name, param) in enumerate(named):
        if name.startswith(prefix):
            block_index = int(name[len(prefix):].split(".", 1)[0])
            stage = stage_of_block(block_index)
        elif i < first_block:
            stage = 0
        elif i > last_block:
            stage = stages - 1
        else:
            raise ValidationError(
                f"parameter {name} is interleaved with the block list; "
                f"v1 supports strictly sequential layouts only")
        assignment[name] = stage
        by_id.setdefault(id(param), []).append(name)

    for names in by_id.values():
        stage_set = {assignment[n] for n in names}
        if len(stage_set) > 1:
            raise ValidationError(
                f"tied parameters {names} land on stages {sorted(stage_set)}: "
                f"cross-stage tied parameters are unsupported in v1 — untie "
                f"them (e.g. tie_word_embeddings=False) or change the partition")

    return tuple(
        StagePartition(
            index=s,
            parameter_names=tuple(n for n, _ in named if assignment[n] == s),
            block_start=bounds[s],
            block_stop=bounds[s + 1],
        )
        for s in range(stages))


class GenericSequentialStage(nn.Module):
    """Pre-modules, a block slice and post-modules chained sequentially. For
    models whose blocks take only the previous output."""

    def __init__(self, pre, blocks, post):
        super().__init__()
        self.pre = nn.ModuleList(pre)
        self.blocks = nn.ModuleList(blocks)
        self.post = nn.ModuleList(post)

    def forward(self, x):
        for module in (*self.pre, *self.blocks, *self.post):
            x = module(x)
        return x

    def local_blocks(self):
        return list(self.blocks)


def _checkpointed(forward, *args, **kwargs):
    import torch.utils.checkpoint
    return torch.utils.checkpoint.checkpoint(forward, *args, use_reentrant=False, **kwargs)


def recompute_blocks(stage: nn.Module) -> None:
    """Keep only each of the stage's own blocks' inputs for backward and
    recompute the rest there. Wraps each block's forward, not the block, so
    parameter names (checkpoints, weight files) stay the same, and module
    hooks still run once, outside the recomputed part."""
    for block in stage.local_blocks():
        block.forward = functools.partial(_checkpointed, block.forward)


def build_stage_module(model: nn.Module, block_start: int, block_stop: int,
                       parameter_names: tuple[str, ...]) -> nn.Module:
    """One stage as a GenericSequentialStage of deep copies, safe to ship to
    an actor."""
    import copy

    blocks_name, blocks = find_block_list(model)
    prefix = blocks_name + "."
    named = [n for n, _ in model.named_parameters(remove_duplicate=False)]
    first_block = min(i for i, n in enumerate(named) if n.startswith(prefix))

    pre_paths, post_paths = [], []
    for name in parameter_names:
        if name.startswith(prefix):
            continue
        path = name.rsplit(".", 1)[0]
        bucket = pre_paths if named.index(name) < first_block else post_paths
        if path not in bucket:
            bucket.append(path)

    return GenericSequentialStage(
        pre=[copy.deepcopy(model.get_submodule(p)) for p in pre_paths],
        blocks=[copy.deepcopy(blocks[i]) for i in range(block_start, block_stop)],
        post=[copy.deepcopy(model.get_submodule(p)) for p in post_paths],
    )


class CausalLMStage(nn.Module):
    """Stage of an HF llama-style causal LM (Qwen, Llama). Each stage carries
    its own copy of the parameter-free rotary embedding and recomputes (cos,
    sin) locally rather than shipping them between stages."""

    def __init__(self, emb, layers, norm, head, rotary):
        super().__init__()
        self.emb = emb
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.head = head
        self.rotary = rotary

    def forward(self, x, position_ids=None):
        import torch
        if self.emb is not None:
            x = self.emb(x)
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        cos_sin = self.rotary(x, position_ids)
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, position_embeddings=cos_sin)
        if self.norm is not None:
            x = self.norm(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def local_blocks(self):
        return list(self.layers)


def build_causal_lm_stage(model: nn.Module, block_start: int, block_stop: int,
                          parameter_names: tuple[str, ...]) -> nn.Module:
    """One stage of an HF llama-style causal LM as a CausalLMStage."""
    import copy

    blocks_name, blocks = find_block_list(model)  # e.g. "model.layers"
    parent_path = blocks_name.rsplit(".", 1)[0]
    parent = model.get_submodule(parent_path)
    names = set(parameter_names)

    def owns(sub):
        return any(n.startswith(f"{parent_path}.{sub}.") for n in names)

    stage = CausalLMStage(
        emb=copy.deepcopy(parent.embed_tokens) if owns("embed_tokens") else None,
        layers=[copy.deepcopy(blocks[i]) for i in range(block_start, block_stop)],
        norm=copy.deepcopy(parent.norm) if owns("norm") else None,
        head=copy.deepcopy(model.lm_head)
            if any(n.startswith("lm_head.") for n in names) else None,
        rotary=copy.deepcopy(parent.rotary_emb),
    )
    attach_tp_plan(stage, getattr(model, "config", None))
    # attention implementation, head counts and MoE routing for the engine.
    # Not named `config`: DeepSpeed would prefer HF's unfiltered TP plan.
    stage._rdsp_hf_config = getattr(model, "config", None)
    return stage


_CAUSAL_LM_PARTS = ("embed_tokens", "norm", "rotary_emb")
# model types whose forward CausalLMStage reproduces, checked on GPU with
# every intra-stage layout; other llama-shaped models differ in details it
# would silently drop (Gemma's embedding scale, Gemma2's logit soft-capping)
_CAUSAL_LM_TYPES = ("llama", "qwen3", "qwen3_moe")


def select_stage_builder(model: nn.Module):
    """The stage builder a model needs, decided from its structure.

    HF causal LMs of a _CAUSAL_LM_TYPES type with the llama-style layout
    (`embed_tokens`, `norm` and `rotary_emb` next to the block list, `lm_head`
    at the top) get build_causal_lm_stage, unless they use eager attention:
    that stage passes no mask, which SDPA and flash attention read as causal
    but eager attention reads as none. Other HF models (a `config`
    attribute) get build_hf_stage, which runs the model's own forward.
    Everything else gets build_stage_module. A rotary embedding outside an HF
    model raises ValidationError, since the generic chain would silently drop
    the positions."""
    blocks_name, _ = find_block_list(model)
    parent_path = blocks_name.rsplit(".", 1)[0] if "." in blocks_name else ""
    parent = model.get_submodule(parent_path) if parent_path else model
    present = {p for p in _CAUSAL_LM_PARTS if isinstance(getattr(parent, p, None), nn.Module)}
    has_head = isinstance(getattr(model, "lm_head", None), nn.Module)
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    eager = getattr(config, "_attn_implementation", None) == "eager"
    if (model_type in _CAUSAL_LM_TYPES and not eager
            and present == set(_CAUSAL_LM_PARTS) and has_head):
        return build_causal_lm_stage
    if hasattr(model, "config"):
        from ray_deepspeed_pipeline.hf_stage import build_hf_stage  # imports this module
        return build_hf_stage
    if "rotary_emb" in present:
        missing = [p for p in _CAUSAL_LM_PARTS if p not in present]
        if not has_head:
            missing.append("lm_head")
        raise ValidationError(
            f"{type(model).__name__} has a rotary embedding next to its block "
            f"list {blocks_name!r} but not the rest of the llama-style causal-LM "
            f"layout (missing {missing}); no built-in stage builder can pass it "
            f"positions correctly")
    return build_stage_module


def attach_tp_plan(stage: nn.Module, config) -> None:
    """Carry the HF model's tensor-parallel plan onto a stage module.

    Only colwise/rowwise entries are kept: anything else makes AutoTP fall
    back to model-type presets, which do not recognize a stage module.
    `replicated_with_grad_allreduce` entries (Qwen3's q_norm/k_norm) stay
    replicated but see per-rank heads, so they are listed in
    `_rdsp_tp_grad_allreduce` for the adapter's TP gradient all-reduce."""
    plan = dict(getattr(config, "base_model_tp_plan", None) or {})
    stage._tp_plan = {k: v for k, v in plan.items()
                      if v.lower() in ("colwise", "rowwise")}
    stage._rdsp_tp_grad_allreduce = tuple(
        k for k, v in plan.items() if v.lower() == "replicated_with_grad_allreduce")
