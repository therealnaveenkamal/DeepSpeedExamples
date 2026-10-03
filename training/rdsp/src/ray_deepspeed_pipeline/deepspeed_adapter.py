"""One pipeline stage's DeepSpeed engine, driven one microbatch at a time.

The coordinator owns the global step and its all-stage ready/apply barrier,
so the engine runs with gradient_accumulation_steps=1 and the adapter scales
the loss by 1/n_microbatches itself: letting DeepSpeed count micro-steps would
fire the optimizer mid-step, before the barrier. engine.step() runs exactly
once, on apply().

Backward crosses the stage boundary as a scalar vector-Jacobian product
through public engine.backward(); DeepSpeed forbids direct tensor.backward()
under ZeRO-0.
"""

import functools
import json

import torch

# TP plan style -> AutoTP layer spec fields
_TP_SPECS = {"colwise": {"partition_type": "column"},
             "rowwise": {"partition_type": "row"},
             "colwise_gather_output": {"partition_type": "column", "gather_output": True},
             "qkv_colwise": {"partition_type": "column", "shape": [3, -1]}}


def _autotp_can_gather() -> bool:
    """Whether this DeepSpeed's AutoTP layer specs take gather_output (older
    versions ignore the key and leave the output split)."""
    import dataclasses
    try:
        from deepspeed.module_inject.autotp_config import TPLayerSpec
    except ImportError:
        return False
    return "gather_output" in {f.name for f in dataclasses.fields(TPLayerSpec)}


def _tp_partition_config(stage_module) -> dict | None:
    """DeepSpeed AutoTP layer rules from the stage's HF TP plan, or None.
    Expert weights are AutoEP's and never TP-split."""
    plan = getattr(stage_module, "_tp_plan", None) or {}
    specs = []
    for pattern, style in plan.items():
        style = style.lower()
        if ".experts" in pattern or style not in _TP_SPECS:
            continue
        if _TP_SPECS[style].get("gather_output") and not _autotp_can_gather():
            continue  # left whole: correct, just not split
        regex = ".*" + pattern.replace(".", r"\.").replace("*", r"[^.]+") + r"\.weight$"
        specs.append({"patterns": [regex], **_TP_SPECS[style]})
    return {"use_default_specs": False, "layer_specs": specs} if specs else None


def _keep_unsplit_module_sizes(stage_module, rules: dict) -> None:
    """Mark every module that directly holds no projection the AutoTP rules
    split as already handled, so AutoTP leaves its size attributes
    (embed_dim, hidden_size, num_heads, ...) whole. Some DeepSpeed versions
    divide them on every module they walk, which breaks modules that read
    them in forward but keep their weights whole (a vision patch embedding,
    a patch merger)."""
    import re

    patterns = [re.compile(p) for spec in rules["layer_specs"] for p in spec["patterns"]]
    for name, module in stage_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            continue
        split = any(isinstance(child, torch.nn.Linear)
                    and any(p.match(f"{name}.{child_name}.weight") for p in patterns)
                    for child_name, child in module.named_children())
        if not split:
            module.replaced = True


def _ulysses_mpu(stage_module, micro_batch_size: int, sp: int, backend: str):
    """The mpu DeepSpeed needs for Ulysses sequence parallelism over the
    stage's HF attention layers. Every SP rank reports model-parallel rank 0:
    each holds a full weight copy, and otherwise SP ranks >= 1 cannot reload a
    checkpoint."""
    import types

    import deepspeed.comm as dscomm
    from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF

    hf_config = getattr(stage_module, "_rdsp_hf_config", None)
    if hf_config is None:
        raise ValueError("sequence parallelism needs an HF attention stage "
                         "(build_causal_lm_stage attaches the model config)")
    dscomm.init_distributed(dist_backend=backend, dist_init_required=False)
    sp_mpu = UlyssesSPAttentionHF.register_with_transformers(
        types.SimpleNamespace(config=hf_config),
        core_attn_implementation=hf_config._attn_implementation,
        sequence_parallel_size=sp, micro_batch_size=micro_batch_size,
        seq_length_is_variable=True)
    mpu = types.SimpleNamespace(**{k: getattr(sp_mpu, k) for k in dir(sp_mpu)
                                   if not k.startswith("__")})
    mpu.get_model_parallel_rank = lambda: 0
    mpu.get_model_parallel_world_size = lambda: 1
    return mpu


def _deepspeed_engine_factory(stage_module, ds_config: dict):
    """One stage-local DeepSpeed engine. The stage's TP/SP/EP degrees come
    from ds_config; model-specific details from the HF config the stage
    module carries."""
    import deepspeed

    conf = json.loads(json.dumps(ds_config))  # deep copy
    conf["gradient_accumulation_steps"] = 1  # the coordinator accumulates (module doc)
    # DeepSpeed derives train_batch_size from the per-rank micro batch and its
    # own data-parallel world, which excludes TP
    conf.pop("train_batch_size", None)
    hf_config = getattr(stage_module, "_rdsp_hf_config", None)
    kwargs = {"model": stage_module, "config": conf, "dist_init_required": False}

    tp_conf = conf.get("tensor_parallel", {})
    tp = int(tp_conf.get("autotp_size", 1) or 1)
    if tp > 1 and "partition_config" not in tp_conf and "preset_model" not in tp_conf:
        rules = _tp_partition_config(stage_module)
        if rules is not None:
            tp_conf["partition_config"] = rules
            _keep_unsplit_module_sizes(stage_module, rules)

    sp = int(conf.get("sequence_parallel_size", 1) or 1)
    if sp > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        kwargs["mpu"] = _ulysses_mpu(stage_module,
                                     int(conf["train_micro_batch_size_per_gpu"]), sp, backend)

    ep_conf = conf.get("expert_parallel")
    if ep_conf and int(ep_conf.get("autoep_size", 1)) > 1:
        if hf_config is not None:
            ep_conf.setdefault("preset_model", getattr(hf_config, "model_type", None))
            ep_conf.setdefault("top_k", hf_config.num_experts_per_tok)
            ep_conf.setdefault("route_norm", bool(getattr(hf_config, "norm_topk_prob", True)))
        # DeepSpeed full-matches module names; stages name layers `layers.<i>`
        # (CausalLMStage) or `model.model.layers.<i>` (HFModelStage)
        ep_conf.setdefault("moe_layer_pattern", r"(.*\.)?layers\.\d+\.mlp")
        ep_conf.setdefault("use_grouped_mm", torch.cuda.is_available())
        # no model_parameters: AutoEP creates the expert parameters inside
        # initialize(), and DeepSpeed collects them (and MoE groups) itself
    else:
        kwargs["model_parameters"] = stage_module.parameters()

    engine, _, _, _ = deepspeed.initialize(**kwargs)
    return engine


def observed_mesh(engine) -> dict | None:
    """This rank's dp/tp/sp coordinates as DeepSpeed's process groups define
    them; None for non-DeepSpeed engines (CPU test stubs)."""
    if "deepspeed" not in type(engine).__module__ and not hasattr(engine, "_engine"):
        return None
    from deepspeed.utils import groups

    def probe(fn_name):
        fn = getattr(groups, fn_name, None)
        try:
            return int(fn()) if fn is not None else None
        except Exception:
            return None

    mesh = {
        "dp_rank": probe("_get_data_parallel_rank"),
        "dp_world": probe("_get_data_parallel_world_size"),
        "tp_rank": probe("get_tensor_model_parallel_rank"),
        "tp_world": probe("get_tensor_model_parallel_world_size"),
        "sp_rank": probe("_get_sequence_parallel_rank"),
        "sp_world": probe("_get_sequence_parallel_world_size"),
    }
    if (mesh.get("sp_world") or 1) > 1:
        # under Ulysses DeepSpeed reports no plain data-parallel world (its
        # gradient group is the whole stage); only the SP coordinates apply
        mesh.pop("dp_rank", None)
        mesh.pop("dp_world", None)
    return {k: v for k, v in mesh.items() if v is not None}


def _to_device(x, device):
    """A tensor, or a dict of tensors (first-stage inputs, boundary extras)."""
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device) if x is not None else None


def _all_reduce_sum(grad, group):
    grad = grad.clone()
    torch.distributed.all_reduce(grad, group=group)
    return grad


class _AverageGradOverGroup(torch.autograd.Function):
    """Identity forward; backward replaces the gradient by its average over
    `group`."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad / torch.distributed.get_world_size(ctx.group), None


def _average_input_grad(group, module, args):
    return (_AverageGradOverGroup.apply(args[0], group), *args[1:])


class DeepSpeedStageAdapter:
    """Forward, external-gradient backward, ready/apply, eval and checkpoint
    shards for one stage. loss_fn is required on the terminal stage and
    forbidden elsewhere. input_grads: first-stage inputs (keys of a dict
    input) whose gradient backward() returns (colocated vision features)."""

    def __init__(self, stage_module, ds_config, n_microbatches: int,
                 is_first: bool, is_last: bool, loss_fn=None,
                 engine_factory=None, input_grads: tuple[str, ...] = ()):
        if isinstance(ds_config, str):
            ds_config = json.loads(ds_config)
        assert (loss_fn is not None) == is_last, \
            "loss_fn belongs on the terminal stage and only there"
        self.n_mb = n_microbatches
        self.is_first = is_first
        self.is_last = is_last
        self.loss_fn = loss_fn
        self.input_grads = input_grads
        self.engine = (engine_factory or _deepspeed_engine_factory)(
            stage_module, ds_config)
        self.device = next(self.engine.module.parameters()).device
        self._install_tp_grad_allreduce()
        self._install_folded_moe_grad_average()
        self._rng_at_generation_start = None
        self._acts = {}    # mb -> (input_leaf_or_None, output)
        self._losses = {}  # mb -> graph loss (terminal only)
        self._backwards = 0

    def _install_tp_grad_allreduce(self) -> None:
        """Sum gradients of replicated parameters whose gradient differs per
        TP rank (Qwen3 q_norm/k_norm see only their rank's heads) over the TP
        group. Tensor hooks fire before accumulation, so every ZeRO stage sees
        the sum; every TP rank runs the same backward, so collectives match."""
        import fnmatch

        patterns = getattr(self.engine.module, "_rdsp_tp_grad_allreduce", ())
        if not patterns:
            return
        try:
            from deepspeed.utils import groups
            if groups.get_tensor_model_parallel_world_size() <= 1:
                return
            group = groups.get_tensor_model_parallel_group()
        except Exception:
            return  # not a DeepSpeed TP engine
        # plan keys are relative to the base model; stage names may carry a prefix
        globs = ["*" + (p if p.endswith("*") else p + ".*") for p in patterns]
        for name, param in self.engine.module.named_parameters():
            if any(fnmatch.fnmatch(name, g) for g in globs):
                param.register_hook(functools.partial(_all_reduce_sum, group=group))

    def _install_folded_moe_grad_average(self) -> None:
        """Average the input gradient of each folded MoE layer over TP.
        AutoEP folding splits experts over TP peers, and each rank's gradient
        below the layer is only correct once averaged over TP. DeepSpeed does
        that for parameter gradients, but everything upstream (earlier layers
        and the gradient sent to the previous stage) would see per-rank values.
        Averaging here gives every rank the true gradient."""
        for module in self.engine.module.modules():
            handles = getattr(module, "folding_group_handles", None)
            if handles is None or handles.spec.tp_size <= 1:
                continue
            group = module.tp_group
            module.register_forward_pre_hook(functools.partial(_average_input_grad, group))

    # -- forward ------------------------------------------------------------

    def _call(self, x, position_offset, extras):
        # everything as flat keyword tensors: under AutoTP, DeepSpeed's
        # first-forward check that TP ranks got the same inputs compares only
        # top-level tensors and raises on a nested dict, on some ranks only
        kwargs = dict(extras or {})
        if position_offset is not None:
            # a sequence shard: rotary and Ulysses need its GLOBAL positions
            kwargs["position_ids"] = torch.arange(
                position_offset, position_offset + x.shape[1], device=self.device).unsqueeze(0)
        if isinstance(x, dict):
            return self.engine(**x, **kwargs)
        return self.engine(x, **kwargs)

    def forward(self, mb: int, x, labels=None, position_offset=None, extras=None,
                loss_weight: float = 1.0):
        """Returns the loss on the last stage, otherwise the boundary output:
        the hidden state, or (hidden, extras) if the stage module adds block
        arguments for the next stage. x may be a dict on the first stage."""
        x = _to_device(x, self.device)
        if not self.is_first:
            x = x.detach().requires_grad_(True)  # the cut
            inp = x
        else:
            x, inp = self._input_leaves(x)
        out = self._call(x, position_offset, _to_device(extras, self.device))
        hidden = out[0] if isinstance(out, tuple) else out
        self._acts[mb] = (inp, hidden)
        if self.is_last:
            loss = self.loss_fn(hidden, labels.to(self.device)) * loss_weight
            self._losses[mb] = loss
            return float(loss.detach())
        return self._boundary(out)

    def _input_leaves(self, x):
        """(x, {name: leaf}) with the input_grads entries of a dict input
        made leaves that collect their gradient; None if there are none."""
        names = [k for k in self.input_grads if isinstance(x, dict) and k in x]
        if not names:
            return x, None
        x = dict(x)
        for k in names:
            x[k] = x[k].detach().requires_grad_(True)
        return x, {k: x[k] for k in names}

    def _boundary(self, out):
        # stays on the device: it leaves over NCCL
        if isinstance(out, tuple):
            return out[0].detach(), out[1]
        return out.detach()

    def eval_forward(self, mb: int, x, labels=None, position_offset=None, extras=None,
                     loss_weight: float = 1.0):
        with torch.no_grad():
            out = self._call(_to_device(x, self.device), position_offset,
                             _to_device(extras, self.device))
            if self.is_last:
                return float(self.loss_fn(out, labels.to(self.device)) * loss_weight)
            return self._boundary(out)

    # -- backward -----------------------------------------------------------

    def backward(self, mb: int, grad=None):
        inp, out = self._acts.pop(mb)
        # only the step's last backward is an accumulation boundary: with gas=1
        # ZeRO-1/2 otherwise treat every backward as one and overwrite (not add)
        # the reduced gradient, keeping only the last microbatch's
        set_boundary = getattr(self.engine, "set_gradient_accumulation_boundary", None)
        if set_boundary is not None:
            set_boundary(self._backwards == self.n_mb - 1)
        if self.is_last:
            loss = self._losses.pop(mb)
            self.engine.backward(loss / self.n_mb)  # summed grads = full-step mean
        else:
            grad = grad.to(self.device)
            # scalar VJP: d/d(out) of (out * grad).sum() is exactly grad
            self.engine.backward((out * grad).sum())
        self._backwards += 1
        if self.is_first:
            return None if inp is None else {k: self._boundary(t.grad) for k, t in inp.items()}
        return self._boundary(inp.grad)

    # -- step barrier -------------------------------------------------------

    def ready(self) -> bool:
        return self._backwards == self.n_mb and not self._acts

    def apply(self) -> bool:
        self.engine.step()
        self._backwards = 0
        self._losses.clear()
        return True

    # -- checkpoint shards and generations ---------------------------------

    def drained(self) -> bool:
        """No microbatch in flight: nothing cached, no partial accumulation."""
        return not self._acts and not self._losses and self._backwards == 0

    def save_shard(self, save_dir: str, tag: str) -> bool:
        """DeepSpeed checkpoint of this stage; a collective over the stage
        world. Writes no `latest` pointer: the manifest owns that."""
        if not self.drained():
            raise RuntimeError("save requested with microbatches in flight")
        self.engine.save_checkpoint(save_dir, tag=tag, save_latest=False)
        return True

    def load_shard(self, load_dir: str, tag: str, *, load_optimizer_states=True,
                   load_lr_scheduler_states=True) -> bool:
        path, _ = self.engine.load_checkpoint(
            load_dir, tag=tag, load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states)
        if path is None:
            raise RuntimeError(f"DeepSpeed found no checkpoint at {load_dir}/{tag}")
        self.reset()
        return True

    def reset(self) -> None:
        """Discard cached activations AND accumulated gradients, so a retried
        step does not add onto an abandoned one."""
        self._acts.clear()
        self._losses.clear()
        self._backwards = 0
        self.engine.zero_grad()
        optimizer = getattr(self.engine, "optimizer", None)
        if optimizer is not None:  # ZeRO>=1 keeps partitioned grads here
            optimizer.zero_grad()
            # ZeRO-2's running sum of reduced microbatch gradients survives
            # zero_grad() (private DeepSpeed state; GPU abandoned-step test)
            running = getattr(optimizer, "all_grad_tensors", None)
            if isinstance(running, dict):
                for key in list(running):
                    running[key] = None
            # with optimizer offload the running sum lives in host buffers, and
            # the micro-step counter decides whether a backward adds to them
            if getattr(optimizer, "cpu_offload", False):
                optimizer.accumulated_grads_in_cpu = {}
                optimizer.micro_step_id = -1  # DeepSpeed's INITIAL_MICRO_STEP_ID

    def begin_generation(self) -> None:
        """Call on a new generation's first command. Leftovers of an abandoned
        generation are discarded and the RNG rewinds to where it started, so a
        retry draws the same dropout masks. A drained stage is untouched."""
        if not self.drained():
            self.reset()
            if self._rng_at_generation_start is not None:
                self._set_rng(self._rng_at_generation_start)
        self._rng_at_generation_start = self._get_rng()

    @staticmethod
    def _get_rng() -> dict:
        import random

        import numpy as np
        state = {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state()
        return state

    @staticmethod
    def _set_rng(state: dict) -> None:
        import random

        import numpy as np
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda"])

    def save_rng(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._get_rng(), path)

    def load_rng(self, path: str) -> None:
        self._set_rng(torch.load(path, weights_only=False))
        self._rng_at_generation_start = None

    # -- introspection (numpy: safe across process boundaries) --------------

    def named_parameters_numpy(self):
        return {n: p.detach().float().cpu().numpy().copy()
                for n, p in self.engine.module.named_parameters()}
