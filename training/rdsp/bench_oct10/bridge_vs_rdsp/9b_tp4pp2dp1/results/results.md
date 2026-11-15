# 9B TP4×PP2×DP1: unprofiled results (2026-10-11)

8× H100 SXM 80 GB (p5.48xlarge, us-west-2c), one box, one session. Equal work: every row padded to 2,048 on both systems. 64 rows per step, 1 row per microbatch, 64 microbatches, 50 steps, no recompute. Settings: `settings.md`.

| | Megatron-Bridge | rdsp |
|---|---|---|
| **Median step time, steps 6–50** | **18.24 s** | **14.63 s** |
| Range, steps 6–50 | 17.50–19.73 s | 14.18–15.18 s |
| First step (warm-up) | 299 s | 26 s |
| Cut | even 16 \| 16 | balanced, chose 16 \| 16 |
| Peak memory per GPU (3-step preflight) | 31.2 GB | ~24 GB |
| Loss, step 1 | 0.8413 | 0.8413 |
| Loss difference, steps 2–50 | — | mean −0.0024, largest 0.049 |

rdsp's step is 19.8% shorter (Megatron takes 1.25× as long) at the same work and the same cut. Logs: `logs/bridge.log`, `logs/rdsp-equal.log`. Where the 3.6 s go is the job of the profiled runs (`../profiles`, conclusion to follow).

Read with care: one round, no repeats yet. On PCIe, repeats landed within about 2%.
