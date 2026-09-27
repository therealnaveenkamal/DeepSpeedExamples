"""Public PipelineConfig and its declarative per-stage overrides.

Only the compiler reads these; the runtime executes the lowered ExecutionPlan.
"""

from dataclasses import dataclass

from ray_deepspeed_pipeline.errors import ValidationError


@dataclass(frozen=True)
class UniformTransformerBlocks:
    """Split the model's transformer blocks as evenly as possible across stages."""


@dataclass(frozen=True)
class UniformSequential:
    """Split a plain sequential block list as evenly as possible across stages."""


@dataclass(frozen=True)
class BalancedTransformerBlocks:
    """Cut so the most expensive stage is as cheap as possible. A block's cost
    is estimated by its parameter count (compute per token scales with it);
    embeddings are lookups and count as nothing. The first stage also carries
    what precedes the blocks (a vision encoder), the last what follows them
    (norm, output head). An estimate: measure, then tune with ExplicitCuts."""


@dataclass(frozen=True)
class ExplicitCuts:
    """Contiguous partition at explicit block boundaries."""

    cuts: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "cuts", tuple(self.cuts))


PartitionConfig = (UniformTransformerBlocks | UniformSequential | BalancedTransformerBlocks
                   | ExplicitCuts)


@dataclass(frozen=True)
class CheckpointPolicy:
    save_optimizer_state: bool = True


@dataclass(frozen=True)
class StageOverride:
    """Per-stage GPU count, parallelism and memory settings.

    The stage's ranks form a grid num_gpus = dp x sp x tp, with dp derived.
    ep (DeepSpeed AutoEP) folds onto those ranks and must divide num_gpus;
    `fold` enables AutoEP Parallel Folding. recompute: keep only each
    block's input for backward and rerun the block there (activation
    checkpointing), trading about a third more compute for memory."""

    stage: int
    num_gpus: int = 1
    zero_stage: int | None = None  # None: inherit the DeepSpeed config's value
    tp: int = 1
    sp: int = 1
    ep: int = 1
    fold: bool = False
    recompute: bool = False


@dataclass(frozen=True)
class ConnectionOverride:
    source: int
    dest: int
    # None: derived from the two stages' layouts; if given, must match it.
    conversion: str | None = None
    buffer_limit: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    stages: int
    partition: PartitionConfig
    schedule: str = "1f1b"
    microbatches: int | None = None
    checkpoint: CheckpointPolicy | None = None
    stage_overrides: tuple[StageOverride, ...] = ()
    connection_overrides: tuple[ConnectionOverride, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "stage_overrides", tuple(self.stage_overrides))
        object.__setattr__(self, "connection_overrides", tuple(self.connection_overrides))
        if self.stages < 1:
            raise ValidationError(f"stages must be >= 1, got {self.stages}")
        if self.schedule != "1f1b":
            raise ValidationError(f"unsupported schedule {self.schedule!r}; v1 supports '1f1b'")
        if self.microbatches is not None and self.microbatches < 1:
            raise ValidationError(f"microbatches must be positive, got {self.microbatches}")
        if not isinstance(self.partition, (UniformTransformerBlocks, UniformSequential,
                                           BalancedTransformerBlocks, ExplicitCuts)):
            raise ValidationError(
                "partition must be a partition policy (UniformTransformerBlocks, "
                "UniformSequential, BalancedTransformerBlocks, or ExplicitCuts)")
        for override in self.stage_overrides:
            if not isinstance(override, StageOverride):
                raise ValidationError(
                    f"stage_overrides entries must be StageOverride, got {type(override)}")
            if not 0 <= override.stage < self.stages:
                raise ValidationError(f"StageOverride.stage {override.stage} out of range")
            if override.num_gpus < 1:
                raise ValidationError(
                    f"StageOverride.num_gpus must be >= 1, got {override.num_gpus}")
            for knob in ("tp", "sp", "ep"):
                if getattr(override, knob) < 1:
                    raise ValidationError(
                        f"StageOverride.{knob} must be >= 1, got {getattr(override, knob)}")
        for connection in self.connection_overrides:
            if not isinstance(connection, ConnectionOverride):
                raise ValidationError(
                    "connection_overrides entries must be ConnectionOverride, "
                    f"got {type(connection)}")
