# rdsp fixes for Qwen3.5-27B: GPU session plan

Written 2026-10-17. Nothing here has run on a GPU yet.

## Why

On 27B, rdsp was 1–8% behind Megatron-Bridge on the data-parallel-heavy layouts. The
TP2×PP2×DP2 profile (`../bridge_vs_rdsp/27b_tp2pp2dp2/logs/rdsp-profile10.log`) shows why:

| Stage | Forward (s/step) | Backward (s/step) | Busy |
|---|---|---|---|
| 0: vision encoder + 32 layers | 5.4 | 11.3 | ~94% |
| 1: 32 layers + output layer + loss | 4.0 | 7.9 | ~68% |

The cut was 32 | 32: the parameter-count estimate put the vision encoder's cost close to
the output layer's cost. Measured, stage 0 costs 4.8 s more per step. Matrix multiplies are
only ~17% of GPU time (Nsight), so most of the gap is CPU-side.

## The four fixes (code done, CPU-tested: 438 unit + 123 integration tests pass)

| # | Flag | What it does | Checked on CPU |
|---|---|---|---|
| 1 | `--vision-attn varlen` | Vision attention as one PyTorch `varlen_attn` call per layer. Under SDPA, Hugging Face reads the image lengths back from the GPU 3 times per layer: 27 layers × 3 × 32 microbatches × 2 (forward and recompute) = 5,184 GPU syncs per step on stage 0. | Same features and gradients as SDPA (CPU stand-in for the CUDA kernel); host reads no longer grow with depth |
| 2 | stage key `recompute_blocks=N` | Recompute only the first N decoder layers of a stage (Bridge's "block" method; Bridge's best D run recomputed 24 of 32). The vision encoder is always recomputed. | Fewer saved bytes, same gradients |
| 3 | `--cuts measured` | rdsp runs 3 steps at learning rate 0 with stage timers on, charges each stage's unexplained time (vision, embedding, output layer, kernel launches) to that stage, re-cuts, and restarts from the loaded weights. A stage only gains layers its GPU memory can hold. | Solver, memory guard, and rebuild on real Ray actors |
| 4 | `--master-dtype remainder` | Exact fp32 master weights stored as the bf16 weight plus a 16-bit remainder: 2 bytes per parameter instead of 4 (Megatron's `store_param_remainders`). About 6.3 GiB less per GPU on TP2×PP2×DP2. | Bit-identical to PyTorch AdamW (single, foreach, fused); bit-identical to DeepSpeed's fp32 master until a value lies exactly halfway between two bf16 numbers (Megatron rounds those the same way); checkpoints keep the remainder |

Dropped: ZeRO-2 on the idle stage. ZeRO-2 reduce-scatters gradients on every microbatch (32
per step), which is what made the D ZeRO-2 run 21.61 s instead of 17.70 s. Fix 4 frees the same memory
without that communication.

## Queue (`queue.sh`, unattended, rdsp only)

Bridge's numbers are the 2026-10-17 runs: same box type, data and settings.

| Order | Run | Layout | Compare with | Est. min |
|---|---|---|---|---|
| 1 | prep (environment, weights, Bridge's exported rows) | — | — | 25 |
| 2 | GPU checks (`scripts/gpu_checks.py`): varlen vs SDPA on the real encoder; remainder vs fp32 master memory, step time and values | 1 GPU | — | 5 |
| 3 | fix-d1: fixes 1 + 4, full recompute | TP2×PP2×DP2 | Bridge 17.47, rdsp 17.70 | 17 |
| 4 | fix-d2: + measured cut | TP2×PP2×DP2 | same | 20 |
| 5 | fix-d3: + recompute 20 of 32 (fit check; else 24) | TP2×PP2×DP2 | Bridge 17.05 (24 of 32) | 26–32 |
| 6 | fix-c1: fixes 1 + 4, full recompute | TP4×DP2 (no PP) | Bridge 26.95, rdsp 29.26 | 26 |
| 7 | fix-e1: fixes + measured cut | TP2×PP4 | Bridge 24.70, rdsp 22.65 | 22 |
| 8 | fix-e2: + recompute 12 of 16 (fit check) | TP2×PP4 | same | 28 |
| 9 | fix-a1: fixes + measured cut, no recompute | TP4×PP2 | Bridge 25.31, rdsp 22.78 | 22 |
| 10 | fix-c2: recompute 48 of 64 (rdsp only; no Bridge pair) | TP4×DP2 | — | 32 |

A fix whose GPU check fails is left out of every run (the queue prints `=== FIXES: ...`).

## Time

On one p5.48xlarge:

| Scope | Time |
|---|---|
| Through step 6 (D and C: the headline and the no-pipeline parity) | ~2.0 h |
| Whole queue | ~3.8 h (3.5–4.2) |
