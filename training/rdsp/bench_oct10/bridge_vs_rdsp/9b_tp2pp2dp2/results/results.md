# 9B TP2×PP2×DP2: results (2026-10-11)

8× H100 SXM, same box and session as the TP4×PP2×DP1 run. Equal work: every row padded to 2,048. 50 steps, no profiler. Settings: `settings.md`.

| | Megatron-Bridge | rdsp |
|---|---|---|
| Median step time, steps 6–50 | 9.09 s | 7.62 s |
| Loss, step 1 | 0.8412 | 0.8412 |
| Loss difference, steps 2–50 | — | mean +0.0004 |

rdsp's step is 1.5 s (16%) shorter. Logs: `logs/bridge.log`, `logs/rdsp.log`. One run per system, not profiled.
