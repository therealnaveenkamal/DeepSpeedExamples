# a. Vision 1 GPU + language TP2×PP2×DP1: why rdsp is 4% faster

> **Corrected 2026-10-11.** The first version said rdsp's vision backward is 2.9 s heavier than MIMO's and its vision GPU is busy 97%. Nsight shows that rdsp's vision GPU waits for the language GPUs' gradient **inside** its backward pass (the autograd thread blocks in `cudaStreamSynchronize` while an NCCL receive runs). py-spy cannot see that wait and counted it as backward work. Those two claims are withdrawn; what follows is what the data supports.

## Result

| | MegatronMIMO | rdsp |
|---|---|---|
| Time per training step | 9.52 s | 9.11 s |

rdsp is 0.41 s per step faster. Equal work: every row padded to 2,048 on both, 64 microbatches of 1 row, same loss curve (`results.md`).

## Why: MIMO is held back by its vision GPU, rdsp is not

Five GPUs: the vision encoder on 1 GPU, then two language stages of 2 GPUs (TP2).

**MIMO: the vision GPU is the slowest stage.**

- Nsight (`screens/1_nsight_mimo_vision_cycle.jpg`): on every microbatch the first language GPU blocks for about 44 ms (traced) in `cudaMemcpyAsync`, the hand-off's `.tolist()`, until the vision GPU finishes its forward pass and sends the result.
- py-spy, untraced, MIMO's vision GPU thread per step: forward 4.9 s, backward 3.6 s, waiting in the hand-off code 1.0 s.

**rdsp: the vision GPU waits for the language GPUs.**

- Nsight (`screens/2_nsight_rdsp_vision_cycle.jpg`): rdsp's vision forward takes 40 ms per microbatch (traced) vs MIMO's 92 ms. Its backward pass then blocks about 66 ms, until the first language GPU finishes its own backward pass and the gradient arrives. Over the traced step that is 3.4 s.
- So rdsp's limit is its language GPUs, not its vision GPU.

| Vision GPU, seconds per step (py-spy, untraced) | MIMO | rdsp |
|---|---|---|
| Forward pass (almost all in the encoder) | 4.9 | **2.3** |
| Backward pass | 3.6 | 6.5, including waits for the gradient (not separable) |
| Waiting in hand-off code | 1.0 | 0.25 |

### Why MIMO's vision forward is slower

MIMO uses Megatron-Bridge's vision encoder (Transformer Engine), as in the Bridge comparison. Per step it takes 2–4× rdsp's CPU time in rotary embedding, attention and MLP (`pyspy_roles.txt`). Only its once-per-layer wait for the GPU is shorter (0.15 s vs 0.27 s).

### MIMO's hand-off

```
megatron/core/pipeline_parallel/bridge_communicator.py
977  shape = grad_shape_tensor.tolist()      # CPU waits until the peer has sent it
```

Every hand-off between the vision and language GPUs first exchanges tensor shapes, then reads them from the GPU, which blocks the CPU. The same code builds debug strings that print `tensor.sum()` even when debug logging is off (`bridge_communicator.py:690`), which also makes the CPU wait (0.9% of the vision thread).

## Why more vision GPUs did not help MIMO

MIMO's vision GPU is its slowest stage, but its **GPU** is idle most of the step (Nsight, `nsys_*.txt`):

| MIMO vision GPU, per traced step (11.9 s) | |
|---|---|
| GPU busy computing | 3.4 s |
| Kernels launched | 158,000 |

The limit is the CPU launching about 158,000 kernels per step. Splitting the encoder over more GPUs (TP or DP) leaves each GPU's CPU with about the same number of launches and adds communication. The MIMO sweep agrees: every multi-GPU vision layout was slower (`../sweep/results.md`). The sweep runs were not profiled.

## Limits

- One run per system, no repeats.
- py-spy shares are 40 s per process while steady, times the median step of the unprofiled run. "Waiting" means time py-spy can see in sync or send/receive code. For rdsp, waits inside the backward pass are invisible to it.
- Nsight slows the CPU side, rdsp more than MIMO (traced steps 12.8 s vs 11.9 s, from 9.1 s and 9.5 s). How long rdsp's vision GPU waits without tracing is not measured.
- Nsight warns that not all CUDA events may have been collected (the app is stopped at the end of the window). Each TP pair's kernel counts match, so this trace looks complete.
- rdsp's vision encoder runs without `torch.compile` (off since it crashed under TP4). Not relevant here: rdsp's vision GPU is not its limit in this layout.
