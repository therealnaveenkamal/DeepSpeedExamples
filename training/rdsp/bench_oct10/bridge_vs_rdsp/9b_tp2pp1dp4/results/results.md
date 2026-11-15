# 9B TP2×DP4, no pipeline: results (2026-10-11)

8× H100 SXM, same box and session as the TP4×PP2×DP1 run. Equal work: every row padded to 2,048 (Bridge with `dataset.pad_to_max_length=true`). 50 steps, no profiler. Settings: `settings.md`.

| | Megatron-Bridge | rdsp (one stage) |
|---|---|---|
| Median step time, steps 6–50 | 6.45 s | 5.84 s |
| Loss, step 1 | 0.8412 | 0.8412 |
| Loss difference, steps 2–50 | — | mean +0.0028 |

rdsp's step is 0.6 s (9%) shorter. With no pipeline, this compares DeepSpeed + Hugging Face code against Megatron + Transformer Engine on the same layout. Logs: `logs/bridge.log`, `logs/rdsp.log`. One run per system, not profiled.
