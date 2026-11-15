# Megatron-Bridge vs rdsp, 8× H100

Each layout has its own directory with `scripts/`, `logs/`, `profiles/`, `results/` and a conclusion.

| Directory | Model | Layout | Why this layout |
|---|---|---|---|
| `9b_tp4pp2dp1/` | Qwen3.5-9B | TP4×PP2×DP1 | Fits without recompute. A pipeline comparison of the two systems' mechanics (a Megatron user would run 9B as TP4×DP2). |
| `9b_tp2pp2dp2/` | Qwen3.5-9B | TP2×PP2×DP2 | The 9B pipeline layout with data parallelism. Unprofiled runs only. |
| `9b_tp2pp1dp4/` | Qwen3.5-9B | TP2×DP4, no pipeline | 9B without a pipeline. Unprofiled runs only. |
| `27b_tp4pp2dp1/` | Qwen3.5-27B | TP4×PP2×DP1 | Stopped after the preflights: it fits only with full recompute (see its `results/conclusion.md`). |
| `4b_tp2pp2dp2/` | Qwen3.5-4B | TP2×PP2×DP2 | The pipeline layout people would choose for 4B. |
| `4b_dp8/` | Qwen3.5-4B | DP8 (rdsp: one stage) | What people would actually run for 4B on 8× H100. |

## Data and batch, the same in every layout

- CORD-v2 receipts. rdsp replays the rows Megatron-Bridge exported: same rows, same order, same microbatch groups.
- 64 rows per step, sequence length 2,048, 50 steps.
- 1 row per GPU per microbatch (Megatron `micro_batch_size=1`). The microbatch count is 64 ÷ DP:

| Layout | DP | Rows per microbatch (all DP copies) | Microbatch count | Idle share from the pipeline |
|---|---|---|---|---|
| 9B TP4×PP2×DP1 | 1 | 1 | 64 | 1/65 ≈ 1.5% |
| 9B TP2×PP2×DP2 | 2 | 2 | 32 | 1/33 ≈ 3% |
| 9B TP2×PP1×DP4 | 4 | 4 | 16 | none (no pipeline) |
| 4B TP2×PP2×DP2 | 2 | 2 | 32 | 1/33 ≈ 3% |
| 4B DP8 | 8 | 8 | 8 | none (no pipeline) |

rdsp flags: `--rows <DP> --microbatches <64/DP>`. Megatron flags: `train.global_batch_size=64 train.micro_batch_size=1`.

## Equal work only

Both systems pad every row to 2,048 positions, so any gap comes from the systems, not the amount of work. Each layout gets one unprofiled 50-step run per system, then profiled runs. (rdsp's own padding, each microbatch to its longest row, is left out by decision of 2026-10-11.)
