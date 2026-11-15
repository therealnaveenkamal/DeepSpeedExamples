# MegatronMIMO vs rdsp, Qwen3.5-9B, 8× H100

The vision encoder runs on its own GPUs, after the language model's GPUs (ranks 0–3). rdsp mirrors each MIMO grid with a vision-only first stage (`--cuts 0`) and the same language grid.

| Layout | Vision | Language | GPUs |
|---|---|---|---|
| a `9b_v1_tp2pp2dp1/` | 1 GPU | TP2×PP2×DP1, 64 microbatches of 1 row | 5 |
| b `9b_v1_tp2pp1dp2/` | 1 GPU | TP2×DP2, 32 microbatches of 2 rows | 5 |
| c `9b_<best>_tp2pp2dp1/` | MIMO's fastest of: TP2, TP4 | TP2×PP2×DP1 | 6–8 |
| d `9b_<best>_tp2pp1dp2/` | MIMO's fastest of: DP2, TP2, TP2×DP2 | TP2×DP2 | 6–8 |

**Outcome (2026-10-11):** the 1-GPU vision layout was MIMO's fastest for both language grids (`sweep/results.md`), so c = a and d = b.

| Layout | MIMO | rdsp |
|---|---|---|
| a vision 1 GPU + TP2×PP2×DP1 | 9.52 s | 9.11 s (4% faster) |
| b vision 1 GPU + TP2×DP2 | 8.11 s | 8.22 s (about even) |

"Fastest" comes from 15-step MIMO runs in `sweep/` (median of steps 6–15), with the 1-GPU vision run counted too. MIMO allows vision DP only when the microbatch has at least as many rows, so c (1 row per microbatch) can only use vision TP.

## The same on both systems

- CORD-v2, 64 rows per step, every row padded to 2,048 (MIMO's default; rdsp `--pad-multiple 0`).
- MIMO's loss: the mean over each microbatch, then over microbatches (rdsp `--loss microbatch-mean`).
- AdamW lr 1e-5 constant, betas 0.9/0.95, eps 1e-8, no weight decay, no clipping, no warm-up; bf16 with fp32 gradients and master weights; no recompute.
- rdsp's vision encoder runs eager (no `torch.compile`), as in the Bridge runs.
- 50 steps, median of steps 6–50.

Scripts: `scripts/run.sh` (every layout's exact grids for both systems), `queue.sh` (the run order). Medians: `../bridge_vs_rdsp/medians.py <layout>/logs`.

## Profiles (2026-10-11)

Layouts a and b were profiled with py-spy (untraced) and Nsight Systems: `scripts/run.sh pyspy|nsys <layout>`, queue `queue_profile.sh`, per-role breakdown `scripts/pyspy_roles.py`, Nsight screenshots in `<layout>/results/screens/`.

- a (`9b_v1_tp2pp2dp1/results/conclusion.md`), rdsp 4% faster: MIMO is limited by its vision GPU (Megatron/TE encoder forward about 2× rdsp's, plus blocking hand-offs); rdsp's faster vision forward moves its limit to the language GPUs.
- b (`9b_v1_tp2pp1dp2/results/conclusion.md`), even: both are limited by their language GPUs; MIMO's forward is slower, rdsp's backward is slower.
- py-spy cannot see rdsp's waits for the next stage's gradient (they happen inside the backward pass); Nsight can. Both conclusions were corrected for this.
