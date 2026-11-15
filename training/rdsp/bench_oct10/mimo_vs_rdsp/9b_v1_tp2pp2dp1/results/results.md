# a. Vision 1 GPU + language TP2×PP2×DP1: results (2026-10-11)

8× H100 SXM, 5 GPUs used. Equal work: every row padded to 2,048. 64 microbatches of 1 row. 50 steps, no profiler. Settings: `../README.md` and `../scripts/run.sh`.

| | MegatronMIMO | rdsp |
|---|---|---|
| Median step time, steps 6–50 | 9.52 s | 9.11 s |
| Loss, step 1 | 0.8413 | 0.8408 |
| Loss difference, steps 2–50 | — | mean −0.0020 |

rdsp's step is 0.41 s (4%) shorter. Logs: `logs/mimo.log`, `logs/rdsp.log`. One run per system.
