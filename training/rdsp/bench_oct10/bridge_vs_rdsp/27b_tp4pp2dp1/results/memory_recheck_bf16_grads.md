# 27B TP4×PP2×DP1: does bf16 gradient accumulation alone fit? (recheck, 2026-10-11)

No GPU run; built from the two no-recompute preflight logs plus the optimizer code. Stage 0, one GPU, GiB.

## Measured limits (at the out-of-memory error in step 2)

| | rdsp | Megatron-Bridge |
|---|---|---|
| Card | 79.18 | 79.18 |
| Used outside PyTorch (CUDA, NCCL, ...) | 3.73 | 9.22 |
| Left for PyTorch | 75.45 | 69.95 |
| Held between steps, after step 1 | 58.98 (`preflight-rdsp-nocompile.log:320`) | 63.19 (`preflight-bridge.log:2884`) |
| Died at | 75.29, in a forward pass (MLP matmul) | 69.86, in a forward pass (attention) |

Megatron's outside-PyTorch memory grew after step 1: in step 1 PyTorch reserved up to 75.39, so it was at most 3.8 then.

## rdsp (DeepSpeed 53a2ac4, ZeRO-1, DP1: nothing is split)

Parameters on stage 0: 3.51B. 2 bytes each = 6.54 GiB, 4 bytes = 13.08 GiB.

| | fp32 gradients (as run) | bf16 gradients | bf16 gradients + fused AdamW |
|---|---|---|---|
| Optimizer DeepSpeed picks | BF16_Optimizer | ZeRO-1 optimizer | ZeRO-1 optimizer |
| Held between steps | 58.98 (measured) | 45.8 | 45.8 |
| Forward/backward peak (+ bf16 grads 6.5, activations 17.9 measured in step 1) | 76.9 → **out of memory** (died at 75.29) | 70.2 | 70.2 |
| Optimizer-step peak | 72.07 (measured) | 45.8 + 6.5 bf16 flat grads + 13.1 fp32 copy of all grads (`stage_1_and_2.py:2274`) + 13.1 AdamW temp = **78.5 → out of memory** | 65.4 |

The AdamW temp is PyTorch's foreach AdamW computing sqrt(exp_avg_sq) for all parameters at once: measured as the step peak minus the held memory (72.07 − 58.98 = 13.09 = 4 B × 3.51B). Fused AdamW (`--fused-adam`, PyTorch `fused=True`) does not make it: CPU check `scripts/adamw_temp_memory.py` -> `results/adamw_temp_memory.txt` (foreach allocates 1.00x the master size in one step, fused 0). DeepSpeed hands the flag to torch AdamW (`results/bf16_states_check.txt`, mode bf16grads_fused: same losses as foreach).

## Megatron-Bridge (MCore core_v0.19.0)

| | fp32 gradients (as run) | bf16 gradients only | precision-aware optimizer: bf16 grads, fp32 master stored as remainders, fp32 Adam |
|---|---|---|---|
| Held between steps | 63.19 (measured) | 56.7 | about 50.1 |
| Room for activations (69.95 − held) | 6.8 → died | 13.3 (rdsp needs 17.9 for the same model) → **likely out of memory** | about 19.9 → maybe |
| Optimizer-step peak | — | 56.7 + 13.1 fp32 copy of all grads (`distrib_optimizer.py:2782`, `.float()`) = **69.8 of 69.95** | no fp32 copy (`decoupled_grad`), fused Adam |

The bf16 options need `mixed_precision=null` plus the bf16 fields set by hand: Bridge's `bf16_mixed` resets `grad_reduce_in_fp32` to True after command-line overrides (`mixed_precision.py:148-156`). Untested.

## Conclusion

bf16 gradients alone fit neither system. Most likely to fit with equal numerics (bf16 gradients, exact fp32 master, fp32 Adam on both): rdsp with `--grad-dtype bf16 --fused-adam` (flag added 2026-10-11), Megatron with the precision-aware optimizer. Estimated margins: rdsp about 5 GiB, Megatron about 2 GiB if its activations match rdsp's. Only a 3-step run settles it.
