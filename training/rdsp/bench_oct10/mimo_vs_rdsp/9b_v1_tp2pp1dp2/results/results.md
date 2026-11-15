# b. Vision 1 GPU + language TP2×DP2: results (2026-10-11)

8× H100 SXM, 5 GPUs used. Equal work: every row padded to 2,048. 32 microbatches of 2 rows (1 per language DP copy). 50 steps, no profiler. Settings: `../README.md` and `../scripts/run.sh`.

| | MegatronMIMO | rdsp |
|---|---|---|
| Median step time, steps 6–50 | 8.11 s | 8.22 s |
| Loss, step 1 | 0.8412 | 0.8408 |
| Loss difference, steps 2–50 | — | mean −0.0038 |

MIMO's step is 0.11 s (1%) shorter: about even. Logs: `logs/mimo.log`, `logs/rdsp.log`. One run per system.
