"""Lowers a PipelineConfig to an ExecutionPlan.

Simple and advanced configs share this one path; the simple path just leaves
overrides at their defaults. All validation happens here, before any actor exists.
"""

import json

from ray_deepspeed_pipeline.boundary import Grid, conversion_name
from ray_deepspeed_pipeline.config import ExplicitCuts, PipelineConfig
from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.partition import (
    partition_parameters,
    vision_config,
    vision_injection_depth,
)
from ray_deepspeed_pipeline.plan import (
    SCHEMA_VERSION,
    ExecutionPlan,
    FailureSpec,
    ScheduleSpec,
    StageConnection,
    StageSpec,
    VisionSpec,
)


def resolve_microbatches(pipeline_config: PipelineConfig, ds_config: dict | None) -> int:
    """Global microbatch count: PipelineConfig.microbatches, else DeepSpeed's
    gradient_accumulation_steps. Both given must agree; neither is an error."""
    gas = None
    if isinstance(ds_config, dict) and "gradient_accumulation_steps" in ds_config:
        gas = int(ds_config["gradient_accumulation_steps"])
        if gas < 1:
            raise ValidationError(f"gradient_accumulation_steps must be positive, got {gas}")
    microbatches = pipeline_config.microbatches
    if microbatches is None and gas is None:
        raise ValidationError(
            "global microbatch count unresolved: set PipelineConfig.microbatches "
            "or gradient_accumulation_steps in the DeepSpeed config")
    if microbatches is None:
        return gas
    if gas is not None and gas != microbatches:
        raise ValidationError(
            f"PipelineConfig.microbatches ({microbatches}) != DeepSpeed "
            f"gradient_accumulation_steps ({gas}); make them equal or omit one")
    return microbatches


def global_microbatch_rows(ds_config: dict | None, microbatches: int) -> int:
    """Rows in one global microbatch (one data-loader entry): train_batch_size
    / microbatches if set, else train_micro_batch_size_per_gpu, else 1."""
    conf = ds_config if isinstance(ds_config, dict) else {}
    if "train_batch_size" in conf:
        tbs = int(conf["train_batch_size"])
        if tbs % microbatches:
            raise ValidationError(
                f"train_batch_size {tbs} is not divisible by the global "
                f"microbatch count {microbatches}")
        return tbs // microbatches
    return int(conf.get("train_micro_batch_size_per_gpu", 1))


def _stage_ds_config(ds_config: dict | None, override, rows_per_rank: int) -> str:
    """Canonical JSON of the user's DeepSpeed config specialised for one stage."""
    conf = dict(ds_config) if isinstance(ds_config, dict) else {}
    conf["train_micro_batch_size_per_gpu"] = rows_per_rank
    # DeepSpeed clips each stage by its own norm (default 1.0), not the global
    # norm; lower() rejects clipping, so it is forced off here.
    conf["gradient_clipping"] = 0.0
    if override is not None:
        _apply_override(conf, override)
    return json.dumps(conf, sort_keys=True, separators=(",", ":"))


def _apply_override(conf: dict, override) -> None:
    if override.zero_stage is not None:
        zero = dict(conf.get("zero_optimization", {}))
        zero["stage"] = override.zero_stage
        conf["zero_optimization"] = zero
    if override.tp > 1:
        tp_conf = dict(conf.get("tensor_parallel", {}))
        tp_conf["autotp_size"] = override.tp
        conf["tensor_parallel"] = tp_conf
    if override.sp > 1:
        conf["sequence_parallel_size"] = override.sp
    if override.ep > 1:
        ep_conf = dict(conf.get("expert_parallel", {}))
        ep_conf.update(enabled=True, autoep_size=override.ep)
        conf["expert_parallel"] = ep_conf
    if override.offload_optimizer:
        zero = dict(conf.get("zero_optimization", {}))
        if int(zero.get("stage", 0)) not in (1, 2):
            raise ValidationError(
                f"stage {override.stage}: optimizer offload needs ZeRO stage 1 or 2, "
                f"got {zero.get('stage', 0)}; set StageOverride(zero_stage=...)")
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        conf["zero_optimization"] = zero


def _stage_grid(override, index: int, n_stages: int, rows: int) -> tuple[Grid, int]:
    """Validate one stage's rank grid; returns (grid, ep)."""
    num_gpus = override.num_gpus if override else 1
    tp, sp, ep = (override.tp, override.sp, override.ep) if override else (1, 1, 1)
    if num_gpus % (tp * sp):
        raise ValidationError(
            f"stage {index}: num_gpus {num_gpus} is not divisible by tp*sp = "
            f"{tp}*{sp}; the stage grid needs num_gpus = dp*sp*tp")
    dp = num_gpus // (tp * sp)
    if rows % dp:
        raise ValidationError(
            f"stage {index}: a global microbatch of {rows} rows cannot be split "
            f"evenly over {dp} data-parallel ranks")
    if num_gpus % ep:
        raise ValidationError(
            f"stage {index}: ep {ep} does not divide num_gpus {num_gpus}")
    if sp > 1 and index == n_stages - 1:
        raise ValidationError(
            f"stage {index}: sequence parallelism on the terminal (loss) stage "
            f"is unsupported — shifted-label losses straddle sequence shards")
    if override is not None and override.fold and ep == 1:
        raise ValidationError(f"stage {index}: fold=True requires ep > 1")
    if override is not None and override.fold != (ep > 1 and tp > 1):
        raise ValidationError(
            f"stage {index}: Parallel Folding is exactly AutoEP (ep > 1) sharing "
            f"the stage with AutoTP (tp > 1); declare fold=True iff both are set")
    if tp > 1 and sp > 1:
        raise ValidationError(
            f"stage {index}: AutoTP and Ulysses sequence parallelism cannot share "
            f"a stage (DeepSpeed's AutoTP replaces the sequence-parallel mpu)")
    return Grid(dp=dp, sp=sp, tp=tp), ep


def _colocated_vision(model, pipeline_config: PipelineConfig, ds_config: dict | None) -> VisionSpec:
    """Validate colocated vision for this model and config."""
    from ray_deepspeed_pipeline.vision import VISION_OPTIMIZERS, find_vision_encoder

    module = find_vision_encoder(model)
    depth = vision_injection_depth(model)
    if depth:
        raise ValidationError(
            f"{type(model).__name__} adds vision features inside the decoder (blocks "
            f"0-{depth - 1}), so its vision encoder must share the first stage; "
            f"colocated vision supports encoders whose output enters only the input "
            f"embeddings")
    partition = pipeline_config.partition
    if isinstance(partition, ExplicitCuts) and partition.cuts and partition.cuts[0] == 0:
        raise ValidationError(
            "a first cut at 0 makes a vision-only first stage, which colocated vision "
            "leaves empty; cut after at least one block")
    ignored = [o.stage for o in pipeline_config.stage_overrides if o.compile_vision]
    if ignored:
        raise ValidationError(
            f"stage {ignored[0]}: compile_vision compiles the encoder a stage holds, and "
            f"colocated vision puts it on no stage; use ColocatedVision(compile=True)")
    conf = ds_config if isinstance(ds_config, dict) else {}
    if conf.get("fp16", {}).get("enabled", False):
        raise ValidationError("colocated vision supports bf16 or fp32, not fp16: the "
                              "encoder's gradients would miss the loss scale")
    kind = conf.get("optimizer", {}).get("type", "AdamW")
    if kind.lower() not in VISION_OPTIMIZERS:
        raise ValidationError(f"colocated vision supports Adam, AdamW or SGD, got {kind!r}")
    names = tuple(n for n, _ in model.named_parameters() if n.startswith(module + "."))
    return VisionSpec(module=module, parameter_names=names,
                      recompute=pipeline_config.colocated_vision.recompute,
                      compile=pipeline_config.colocated_vision.compile,
                      encode_per_microbatch=(
                          pipeline_config.colocated_vision.encode_per_microbatch))


def _check_tp_divides_heads(model, tp: int, stage: int) -> None:
    """TP splits attention heads across a stage's GPUs, so it must divide the
    key/value head count (read from the model's text config, if it has one)."""
    config = getattr(model, "config", None)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    heads = getattr(text, "num_key_value_heads", None)
    if tp > 1 and isinstance(heads, int) and heads % tp:
        raise ValidationError(
            f"stage {stage}: tp={tp} does not divide the model's {heads} key/value "
            f"heads; use a tp that divides {heads}")


def lower(model, pipeline_config: PipelineConfig, ds_config: dict | None) -> ExecutionPlan:
    """Validate and lower; raises ValidationError before any Ray actor exists."""
    n = pipeline_config.stages
    microbatches = resolve_microbatches(pipeline_config, ds_config)
    clip = 0.0
    if isinstance(ds_config, dict):
        clip = float(ds_config.get("gradient_clipping", 0.0))
    if clip > 0:
        raise ValidationError(
            f"gradient_clipping={clip} would clip each stage by its own gradient "
            f"norm, which is not global-norm clipping; unsupported in v1 — set "
            f"gradient_clipping to 0 (rdsp also turns off DeepSpeed's default "
            f"of 1.0 for every stage)")
    overrides = {}
    for override in pipeline_config.stage_overrides:
        if override.stage in overrides:
            raise ValidationError(f"duplicate StageOverride for stage {override.stage}")
        overrides[override.stage] = override
    first = overrides.get(0)
    if first is not None and first.sp > 1 and \
            vision_config(model) is not None:
        raise ValidationError(
            "stage 0: sequence parallelism on the first stage of a vision model is "
            "unsupported; image positions depend on the whole sequence")
    vision = None
    if pipeline_config.colocated_vision is not None:
        vision = _colocated_vision(model, pipeline_config, ds_config)
    stage_gpus = tuple(overrides[i].num_gpus if i in overrides else 1 for i in range(n))
    partitions = partition_parameters(model, pipeline_config.partition, n, stage_gpus,
                                      exclude=vision.module if vision else None)

    rows = global_microbatch_rows(ds_config, microbatches)
    stages, grids = [], []
    for part in partitions:
        override = overrides.get(part.index)
        grid, ep = _stage_grid(override, part.index, n, rows)
        _check_tp_divides_heads(model, grid.tp, part.index)
        grids.append(grid)
        stages.append(StageSpec(
            index=part.index,
            parameter_names=part.parameter_names,
            block_start=part.block_start,
            block_stop=part.block_stop,
            num_gpus=grid.world,
            ds_config_json=_stage_ds_config(ds_config, override, rows // grid.dp),
            process_group_tag=f"stage{part.index}",
            dp=grid.dp, tp=grid.tp, sp=grid.sp, ep=ep,
            fold=bool(override.fold) if override else False,
            rows_per_rank=rows // grid.dp,
            recompute=bool(override.recompute) if override else False,
            compile=bool(override.compile) if override else False,
            compile_vision=bool(override.compile_vision) if override else False,
        ))

    # a ConnectionOverride only asserts the conversion the layouts imply
    for conn in pipeline_config.connection_overrides:
        if conn.dest != conn.source + 1 or not 0 <= conn.source < n - 1:
            raise ValidationError(
                f"connection ({conn.source}->{conn.dest}) is not an adjacent forward "
                f"edge of a {n}-stage pipeline")
        derived = conversion_name(grids[conn.source], grids[conn.dest])
        if conn.conversion is not None and conn.conversion != derived:
            raise ValidationError(
                f"connection {conn.source}->{conn.dest} declares conversion "
                f"{conn.conversion!r} but the stage layouts require {derived!r}")

    connections = tuple(
        StageConnection(
            source=i, dest=i + 1,
            conversion=conversion_name(grids[i], grids[i + 1]),
        )
        for i in range(n - 1)
    )

    return ExecutionPlan(
        schema_version=SCHEMA_VERSION,
        global_microbatches=microbatches,
        stages=tuple(stages),
        connections=connections,
        schedule=ScheduleSpec(kind=pipeline_config.schedule,
                              global_microbatches=microbatches),
        failure=FailureSpec(poison_on_partial_apply=True),
        microbatch_rows=rows,
        colocated_vision=vision,
        prefetch=pipeline_config.prefetch,
    )
