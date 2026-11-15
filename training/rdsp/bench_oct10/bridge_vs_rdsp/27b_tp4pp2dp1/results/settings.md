# 27B TP4×PP2×DP1: every setting, side by side

Checked before any GPU spend. Bridge: `qwen35_vl_27b_sft_16gpu_h100_bf16_config` (r0.6.0) with the overrides in `bench/mimo/runs.sh` (`STD_TRAIN`), TP4 PP2. rdsp: `scripts/rdsp.sh`. The 27B recipe differs from the 4B recipe used in every earlier run only in model path, TP/PP and `pipeline_dtype`.

## Held equal

| Setting | Megatron-Bridge | rdsp |
|---|---|---|
| Hardware | 8× H100 SXM 80 GB, one p5.48xlarge, NVLink | same box, same session |
| Weights | HF Qwen/Qwen3.5-27B, converted | HF Qwen/Qwen3.5-27B |
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
| Recompute | full: every layer (`recompute_granularity=full`, `uniform`, 1 layer) | every block on both stages (`recompute=1`) |
| Steps, timing | 50 steps, median of steps 6–50 | same |

## Different

| Setting | Megatron-Bridge | rdsp | Why it differs |
|---|---|---|---|
| Layers per stage | even 32 \| 32 | `--cuts balanced` | by design: how each system cuts |
| Model code and kernels | Megatron rebuild, Transformer Engine layers, fused cross entropy, fused fp32 gradient accumulation in the GEMM | Hugging Face modules, Liger norm/SwiGLU, vocabulary-sharded loss, fp32 accumulation as a separate add | by design: each system's kernels |
| Vision encoder kernels | TE layers | HF blocks, eager (no `torch.compile`: it crashed under TP4, see below) | by design |
| Process model | one SPMD process per GPU (torchrun) | one Ray actor per GPU plus a CPU driver | by design |
| Pipeline sends | `torch.cuda.synchronize()` after every send/receive | asynchronous sends, waited at step end | by design (each system's code) |
| **Data path** | tokenizes and processes images in its data loader while training | loads the already-processed rows, next step shipped during this one (`--prefetch`) | side effect of replaying Bridge's rows; check the profile for data-loader time |
| **NaN check** | `check_for_nan_in_grad` on: scans gradients every step | none | Bridge recipe default; small cost |
| **Python garbage collector** | `manual_gc`: automatic GC off, collect every 100 steps (never within 50) | automatic GC on | Bridge recipe default; may spare Bridge GC pauses |

The by-design rows are what the comparison measures. The three side-effect rows stay at each system's default and are named in the results; the profiles show whether any of them costs measurable time.

## Decisions made on the box (2026-10-11)

- **Recompute on both.** Without it, both systems finished step 1 and ran out of memory on stage 0 in step 2, once Adam's state (created in step 1) was resident: Megatron at 79.2 GB with 63.2 GB allocated after step 1; rdsp the same way. Step-1 losses matched (Megatron 0.7169, rdsp 0.7161). Logs: `logs/preflight-bridge.log`, `logs/preflight-rdsp-nocompile.log`.
- **rdsp's vision encoder runs eager.** `torch.compile` of the vision blocks under TP4 failed with "CUDA error: misaligned address" (`logs/preflight-rdsp.log`); eager, the same encoder ran (`logs/preflight-rdsp-nocompile.log`). Likely cause, not proven: the 27B vision MLP is 4,304 wide, 1,076 per TP rank, so its rows are 2,152 bytes, not a multiple of 16, while the 4B encoder (4,096 wide) compiled fine. The encoder is about 4% of a 27B step, so compile was worth well under 1%.
