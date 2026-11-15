# b. Vision 1 GPU + language TP2×DP2: why it is even

> **Corrected 2026-10-11.** The first version said rdsp is limited by its vision GPU because its vision backward is 2.3× heavier. Nsight shows that rdsp's vision GPU waits for the language GPUs' gradient **inside** its backward pass, which py-spy counted as backward work. That claim is withdrawn.

## Result

| | MegatronMIMO | rdsp |
|---|---|---|
| Time per training step | 8.11 s | 8.22 s |

MIMO is 0.11 s (1%) faster: about even. Equal work: every row padded to 2,048 on both, 32 microbatches of 2 rows, same loss curve (`results.md`).

## Why: both are limited by their language GPUs, which are about equally fast

Five GPUs: the vision encoder on 1 GPU, then one language stage on 4 GPUs (TP2 × DP2).

**On both systems the vision GPU waits for the language GPUs.** Nsight, per traced step:

| | MIMO vision GPU | rdsp vision GPU |
|---|---|---|
| CPU blocked waiting on the language GPUs | 3.0 s (main thread, MIMO's hand-off `.tolist()`) | 4.4 s (inside the backward pass) |
| GPU busy computing | 3.5 s (of 9.6 s) | 3.0 s (of 12.0 s) |

So the language GPUs set the pace on both. They are the last stage, so their backward pass has no later stage to wait for, and py-spy's numbers for them are sound:

| Language GPU (mean of 4), seconds per step, py-spy | MIMO | rdsp |
|---|---|---|
| Forward pass | 3.8 | 3.3 |
| Backward pass | 3.4 | **4.5** |
| Busy, total | 7.3 | 7.8 |
| Waiting (sync, send/receive) | 0.8 | 0.4 |
| Step | 8.1 | 8.2 |

- MIMO's language forward is 0.5 s slower (Megatron/Transformer Engine code).
- rdsp's language backward is 1.1 s slower (1.3×).
- rdsp waits 0.4 s less.
- Net: about even.

## What would move the result

- **rdsp:** a lighter language backward pass (1.1 s behind MIMO's). Not investigated yet.
- **MIMO:** a lighter language forward pass, and fewer waits.

## Limits

- One run per system, no repeats.
- py-spy shares are 40 s per process while steady, times the median step of the unprofiled run. Waits inside rdsp's backward pass are invisible to py-spy. That matters for the vision GPU, not for the language GPUs (the last stage).
- Nsight slows the CPU side (steps 9.6–12 s instead of about 8 s). It is used only for wait times, GPU busy time and kernel counts.
- **Nsight lost kernel records on 2 of the 4 language GPUs, on both systems.** One TP rank of each pair shows 2.4–3.2× fewer kernels than its partner, and Nsight warns "Not all CUDA events might have been collected" because the app is stopped at the end of the window. Language-GPU numbers from Nsight are not used.
