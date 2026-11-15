# 27B TP4×PP2×DP1 without recompute: both systems fit with bf16 gradients (2026-10-12)

3-step fit checks on one p5.48xlarge (8× H100 80 GB, us-east-2a). Not benchmark numbers: rdsp ran with `--profile` timers on, and 3 steps is too few for losses.

## Settings that changed from the Oct 11 preflights (everything else as in `settings.md`)

| | Megatron-Bridge | rdsp |
|---|---|---|
| Recompute | off | off |
| Gradient buffer | bf16 (`ddp.grad_reduce_in_fp32=false`) | bf16 (`--grad-dtype bf16`) |
| Master weights | fp32, stored as bf16 weight + 16-bit remainder (`store_param_remainders`, exact) | fp32 copy (DeepSpeed ZeRO-1 optimizer) |
| Adam moments | fp32 | fp32 |
| Adam kernel | Transformer Engine FusedAdam (precision-aware optimizer) | PyTorch fused AdamW (`--fused-adam`) |
| Precision preset | `mixed_precision=null`, bf16 fields set by hand (`scripts/meg_extra_bf16.txt`) | — |

Bridge's printed configuration confirmed `main_grads_dtype: bfloat16` and `use_precision_aware_optimizer: True`; master and Adam stayed at Megatron's fp32 defaults (`logs/preflight-bridge-bf16grads.log`).

## Memory, stage 0, one GPU (GiB; card 79.2)

| | Megatron-Bridge | rdsp |
|---|---|---|
| Held between steps (PyTorch) | 49.3 | 45.9 |
| Peak allocated by PyTorch | 66.0 | 72.1 (optimizer step), 70.5 (forward/backward) |
| Peak used on the card, all of it (`nvidia-smi`, every 0.5 s) | 72.8 (GPU 0) | not sampled; PyTorch found the card full 4 times and retried |
| Stage 1 peak on the card | 62.4 | 58.0 (PyTorch) |

Megatron holds less because its fp32 master costs 2 bytes per parameter (remainders) instead of 4, and its step makes no fp32 copy of the gradients. rdsp's optimizer step holds the bf16 gradients, a flat bf16 copy and an fp32 copy at once (45.9 + 6.5 + 6.5 + 13.1 = 72.0).

## Steps (3-step runs; first step includes start-up)

| Run | Step 2 | Step 3 | Loss 1 | Loss 2 | Loss 3 |
|---|---|---|---|---|---|
| Megatron, bf16 grads, no recompute | 26.6 s | 27.0 s | 0.7168 | 0.3470 | 0.2896 |
| rdsp, bf16 grads + fused AdamW, no recompute (`--profile` on) | 26.5 s | 24.6 s | 0.7161 | 0.3457 | 0.2994 |
| Megatron, fp32 grads, full recompute (Oct 11) | — | 36.4 s | 0.7167 | 0.3464 | 0.2868 |
| rdsp, fp32 grads, full recompute (Oct 11, `--profile` on) | 36.6 s | 37.5 s | 0.7161 | 0.3456 | 0.2692 |

Logs: `logs/preflight-bridge-bf16grads.log`, `logs/nvsmi-bridge-bf16.csv`, `logs/preflight-rdsp-bf16grads-fused.log`.
