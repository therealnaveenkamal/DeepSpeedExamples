"""ExecutionPlan: the normalized plan every PipelineConfig lowers to.

The runtime reads only this, never PipelineConfig. Plain, JSON-canonical data
so the plan has a stable hash; checkpoints are bound to that hash.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass

SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class StageSpec:
    index: int
    # fully qualified names in model order; the actor builds its module from them
    parameter_names: tuple[str, ...]
    # contiguous block range [block_start, block_stop) within the model's
    # transformer block list; first/last stage also own the pre/post modules
    block_start: int
    block_stop: int
    num_gpus: int
    # canonical JSON of this stage's ordinary DeepSpeed config
    ds_config_json: str
    process_group_tag: str
    # stage-local rank grid (num_gpus = dp * sp * tp); ep folds onto it
    dp: int = 1
    tp: int = 1
    sp: int = 1
    ep: int = 1
    fold: bool = False
    # rows of one global microbatch each dp shard holds
    rows_per_rank: int = 1
    # rerun blocks in backward instead of keeping their activations
    recompute: bool = False


@dataclass(frozen=True)
class StageConnection:
    source: int
    dest: int
    # identity | replicate-to-shard | shard-to-replicate | shard-to-shard
    conversion: str
    buffer_limit: int


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str  # "1f1b"
    global_microbatches: int


@dataclass(frozen=True)
class CheckpointSpec:
    save_optimizer_state: bool


@dataclass(frozen=True)
class FailureSpec:
    poison_on_partial_apply: bool


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: str
    global_microbatches: int
    stages: tuple[StageSpec, ...]
    connections: tuple[StageConnection, ...]
    schedule: ScheduleSpec
    checkpoint: CheckpointSpec
    failure: FailureSpec
    # rows in one global microbatch (the data loader's entry size)
    microbatch_rows: int = 1

    def to_canonical_dict(self) -> dict:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True,
                          separators=(",", ":"))

    def plan_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
