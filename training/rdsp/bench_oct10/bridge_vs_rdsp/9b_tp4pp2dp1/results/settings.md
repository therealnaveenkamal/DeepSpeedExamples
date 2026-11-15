# 9B TP4×PP2×DP1: every setting, side by side

Checked before any GPU spend. Bridge: `qwen35_vl_9b_sft_4gpu_h100_bf16_config` (r0.6.0) with the overrides in `bench/mimo/runs.sh` (`STD_TRAIN`), TP4 PP2. rdsp: `scripts/rdsp.sh`. The 9B recipe differs from the 4B recipe used in every earlier run only in model path and TP.

## Held equal

| Setting | Megatron-Bridge | rdsp |
|---|---|---|
| Hardware | 8× H100 SXM 80 GB, one p5.48xlarge, NVLink | same box, same session |
| Weights | HF Qwen/Qwen3.5-9B, converted | HF Qwen/Qwen3.5-9B |
| Output layer | own weight (`tie_word_embeddings` false) | own weight |
| MTP head | off (`mtp_num_layers=null`) | none |
| Trained parts | everything (language, vision, projection) | everything |
| Data rows and order | CORD-v2, Bridge's sampler | Bridge's exported rows, replayed |
| Rows per step / per microbatch / microbatches | 64 / 1 / 64 | 64 / 1 / 64 |
| Sequence length, padding (equal-work run) | 2,048, every row padded (forced under PP) | 2,048, every row padded (`--pad-multiple 0`) |
| Pipeline schedule | 1F1B, 2 stages | 1F1B, 2 stages |
| Vision encoder placement | stage 0, TP4 | stage 0, TP4 |
| Tensor parallelism | TP4, no sequence parallel, no TP comm overlap | TP4, no sequence parallel |
| Embedding and output layer | split by vocabulary over TP | split by vocabulary over TP |
| GDN (linear attention) | split by heads over TP, FLA kernels | split by heads over TP, FLA kernels |
| Attention | causal, TE `auto` backend | causal, SDPA flash (no padding mask) |
| Loss | token mean (`calculate_per_token_loss`) | token mean |
| Optimizer math | Adam, weight decay 0 (= AdamW) | AdamW, weight decay 0 |
| lr, betas, eps, clipping, warm-up | 1e-5 constant, 0.9/0.95, 1e-8, off, 0 | same |
| Parameter / master / gradient dtype | bf16 / fp32 / fp32 | bf16 / fp32 / fp32 |
| Optimizer-state split | distributed optimizer, DP1: nothing to split | ZeRO-1, DP1: nothing to split |
| Gradient reduce overlap | off | none |
| CUDA graphs | off | off |
| Recompute | none | none |
| Steps, timing | 50 steps, median of steps 6–50 | same |

## Different

| Setting | Megatron-Bridge | rdsp | Why it differs |
|---|---|---|---|
| Layers per stage | even 16 \| 16 | `--cuts balanced`, which chose 16 \| 16 (log: `stages (blocks): [(0, 16), (16, 32)]`) | same split here |
| Model code and kernels | Megatron rebuild, Transformer Engine layers, fused cross entropy, fused fp32 gradient accumulation in the GEMM | Hugging Face modules, Liger norm/SwiGLU, vocabulary-sharded loss, fp32 accumulation as a separate add | by design: each system's kernels |
| Vision encoder kernels | TE layers | HF blocks, eager (no `torch.compile`: the same encoder crashed compiled under TP4 on 27B, see `../27b_tp4pp2dp1/results/settings.md`) | by design |
| Process model | one SPMD process per GPU (torchrun) | one Ray actor per GPU plus a CPU driver | by design |
| Pipeline sends | `torch.cuda.synchronize()` after every send/receive | asynchronous sends, waited at step end | by design (each system's code) |
| **Data path** | tokenizes and processes images in its data loader while training | loads the already-processed rows, next step shipped during this one (`--prefetch`) | side effect of replaying Bridge's rows; check the profile for data-loader time |
| **NaN check** | `check_for_nan_in_grad` on: scans gradients every step | none | Bridge recipe default; small cost |
| **Python garbage collector** | `manual_gc`: automatic GC off, collect every 100 steps (never within 50) | automatic GC on | Bridge recipe default; may spare Bridge GC pauses |

The by-design rows are what the comparison measures. The three side-effect rows stay at each system's default and are named in the results; the profiles show whether any of them costs measurable time.
