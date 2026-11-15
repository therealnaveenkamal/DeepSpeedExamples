# 27B TP4×PP2×DP1: stopped after the preflights (2026-10-11)

**Status: not benchmarked.** Without recompute, neither system fits 27B in this layout on 8× H100 (both ran out of memory on stage 0 in step 2). With full recompute both fit, but recompute makes every step mostly recomputation, which hides the differences this comparison is meant to find. The user stopped the layout and moved to 9B TP4×PP2×DP1, which fits without recompute.

## What the 3-step preflights showed

| Run | Step time | Peak memory per GPU | Log |
|---|---|---|---|
| Megatron-Bridge, no recompute | step 1 only; out of memory in step 2 (79.2 GB) | 63.2 GB after step 1 | `logs/preflight-bridge.log` |
| rdsp, no recompute, vision compiled | crashed: "CUDA error: misaligned address" in the compiled vision encoder | — | `logs/preflight-rdsp.log` |
| rdsp, no recompute, vision eager | step 1 only; out of memory in step 2 | 69.0 GB at step 1's optimizer step | `logs/preflight-rdsp-nocompile.log` |
| Megatron-Bridge, full recompute | 36.4 s (step 3) | 61.9–65.4 GB | `logs/preflight-bridge-rc.log` |
| rdsp, full recompute, vision eager, `--profile` on | 36.6 s, 37.5 s (steps 2–3) | 62.9 GB (stage 0), 69.0 GB (stage 1) | `logs/preflight-rdsp-rc.log` |

Step-1 losses agreed: Megatron 0.7169, rdsp 0.7161. Both systems cut 32 | 32 (rdsp's balanced cut chose the even split: at 27B the vision encoder is about 4% of the work).

These are three-step numbers, with rdsp's per-phase timers on; they are not benchmark results.
