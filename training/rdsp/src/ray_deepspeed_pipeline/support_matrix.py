"""Supported configurations. Default-deny: a row is supported only once its
validation run has passed and is recorded here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SupportRow:
    row_id: str
    dtype: str
    stages: int
    gpus_per_stage: int
    dp: int
    tp: int
    ep: int
    sp: int
    zero_stage: int
    schedule: str
    boundaries: str
    model_fixture: str
    microbatches: int
    supported: bool


FIRST_ROW = SupportRow(
    row_id="p6-first-row",
    dtype="bf16",
    stages=2,
    gpus_per_stage=1,
    dp=1, tp=1, ep=1, sp=1,
    zero_stage=0,
    schedule="1f1b",
    boundaries="identity",
    model_fixture="Qwen/Qwen3-0.6B",
    microbatches=4,
    # Evidence, 2026-08-30, Modal 2xL4: eval/train loss parity vs the unsplit
    # model (rel 2e-2, BF16), loss decreases, exact microbatch consumption, eval
    # leaves weights unchanged, engine surface. torch 2.13.0+cu130,
    # deepspeed 0.19.3+53a2ac44, transformers 5.16.1.
    supported=True,
)


@dataclass(frozen=True)
class HeterogeneousRow:
    """A per-stage-heterogeneous configuration.

    `stages` describes each stage's grid as text; `requires` lists rows that must
    be supported first; `evidence` records the passing run of a supported row.
    """

    row_id: str
    scope: str
    stages: tuple[str, ...]
    requires: tuple[str, ...] = ()
    supported: bool = False
    evidence: str = ""
    note: str = ""  # why an unsupported row is not yet validated


_PARITY_EVIDENCE = (
    "2026-09-23 Modal 8xL4, deepspeed 0.19.3+53a2ac44 (rerun after the ZeRO-1/2 "
    "accumulation-boundary fix): test_row_parity_global_microbatch_checkpoint_failure[{row}] "
    "PASSED — SGD loss parity vs single-process fp32 baseline (rel 1e-4, learnable data, "
    "sensitivity-checked), exact N-entry consumption, checkpoint round trip into a fresh "
    "differently-initialized pipeline, non-zero-rank kill -> StepFailed -> whole-pipeline "
    "recovery"
)

P8_ROWS = (
    HeterogeneousRow(
        "p8-static-boundary", "replicate<->shard StageConnection conversion",
        ("1 GPU", "dp=2", "1 GPU"),
        supported=True,
        evidence=_PARITY_EVIDENCE.format(row="p8-static-boundary")),
    HeterogeneousRow(
        "p8-resource-mesh", "differing per-stage GPU counts and meshes",
        ("dp=2", "dp=4", "dp=2"),
        supported=True,
        evidence=_PARITY_EVIDENCE.format(row="p8-resource-mesh")),
    HeterogeneousRow(
        "p8-dp-zero", "differing per-stage DP degrees and ZeRO stages",
        ("dp=4 ZeRO-2", "dp=2 ZeRO-1", "dp=2 ZeRO-0"),
        supported=True,
        evidence=_PARITY_EVIDENCE.format(row="p8-dp-zero")),
    HeterogeneousRow(
        "p8-autotp", "stage-local AutoTP",
        ("1 GPU", "tp=2", "tp=2 x dp=2"),
        supported=True,
        evidence=(
            "2026-09-23 Modal 8xL4, deepspeed 0.19.3+53a2ac44, tiny random Qwen3 (fp32): "
            "test_row_parity_global_microbatch_checkpoint_failure[p8-autotp] PASSED (SGD loss "
            "parity rel 1e-4, exact consumption, checkpoint round trip with per-TP-rank shards, "
            "rank-kill recovery); test_row_gradient_parity_per_parameter[p8-autotp] PASSED — "
            "91 parameters, worst update error 2.4e-5 (TP shards vs matching reference "
            "slices). Requires the rdsp q_norm/k_norm TP gradient-sum hook (DeepSpeed AutoTP "
            "leaves them partial)")),
    HeterogeneousRow(
        "p8-autoep-folding", "stage-local AutoEP plus Parallel Folding",
        ("1 GPU", "ep=4", "ep=2 folded with tp=2"),
        supported=True,
        evidence=(
            "2026-09-23 Modal 8xL4, deepspeed 0.19.3+53a2ac44, tiny random Qwen3-MoE "
            "(8 experts, top-2, fp32): "
            "test_row_parity_global_microbatch_checkpoint_failure[p8-autoep-folding] PASSED "
            "(ep=4 stage and ep=2 x tp=2 folded stage; SGD loss parity rel 1e-4, exact "
            "consumption, checkpoint round trip, rank-kill recovery); "
            "test_row_gradient_parity_per_parameter PASSED — 76 parameters (AutoEP-renamed "
            "expert/router weights covered by loss parity), worst 3.0e-4 (folded-stage "
            "input_layernorm), rest <= 1.8e-5. Requires the rdsp TP-average hook at folded "
            "MoE inputs")),
    HeterogeneousRow(
        "p8-sequence-parallel", "stage-local Ulysses sequence parallelism",
        ("sp=2", "sp=2 x dp=2", "1 GPU"),
        supported=True,
        evidence=(
            "2026-09-23 Modal 8xL4, deepspeed 0.19.3+53a2ac44, tiny random Qwen3 (fp32): "
            "test_row_parity_global_microbatch_checkpoint_failure[p8-sequence-parallel] "
            "PASSED (SGD loss parity rel 1e-4, exact consumption, checkpoint round trip incl. "
            "SP ranks >= 1 via the model-rank-0 mpu wrapper, rank-kill recovery); "
            "test_row_gradient_parity_per_parameter PASSED — 91 parameters, worst 2.4e-5. "
            "Gradient convention: SP stages average over dp only (DeepSpeed sums over SP)")),
    HeterogeneousRow(
        "p8-four-stage-mixed", "composed topology of Design §5.2.1 (32 GPUs, 4 nodes)",
        ("dp=4 x sp=2, ZeRO-2", "tp=4 x dp=2, ZeRO-1",
         "ep=8 AutoEP + Parallel Folding", "tp=2 x dp=4"),
        requires=("p8-static-boundary", "p8-resource-mesh", "p8-dp-zero",
                  "p8-autotp", "p8-autoep-folding", "p8-sequence-parallel"),
        note=("all six prerequisites supported, and the same four stage kinds composed "
              "at 8-GPU scale pass every check on one node "
              "(test_*[p8-four-stage-mixed-mini], 85 parameters, worst update error 2.4e-5). "
              "The exact 32-GPU, 4-node run is blocked by the Modal workspace limit of 10 "
              "concurrent GPUs (multi-node Modal also requires 8 GPUs per node), so "
              "cross-node rendezvous/placement/node-local checkpoints are validated only on "
              "one node (2026-09-23). Ready to run: "
              "modal run scripts/modal_cluster.py --tests "
              "\"tests/integration/test_heterogeneous_pipeline.py -k four-stage-mixed\"")),
)

ROWS = (FIRST_ROW, *P8_ROWS)
