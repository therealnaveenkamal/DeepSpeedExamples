# 9B on 8× H100: why rdsp is faster than Megatron-Bridge

> **Correction, 2026-10-11.** An earlier version said Bridge's vision encoder costs 2.2 s more and its backward pass 1.2 s more (cause unknown). That split was wrong. The py-spy analysis counted every stack with a `torch/autograd` frame as "backward", but fused kernels and Transformer Engine layers run their *forward* pass through `torch/autograd/function.py` too. That moved forward time into "backward": 24% of Bridge's samples, 13% of rdsp's. Fixed in `scripts/analyze_pyspy.py`; the numbers below are recomputed. The step times and the total gap did not change.

## Result

| | Megatron-Bridge | rdsp |
|---|---|---|
| Time per training step | 18.24 s | 14.63 s |

rdsp is 3.6 s per step faster (20%).

The comparison is fair: same model, same data, same batch, every row padded to 2,048 tokens on both, same layer split (16 | 16), same loss curve. Details in `results.md` and `settings.md`.

## Why: Bridge's vision encoder

The pipeline has two stages. Stage 0 runs the vision encoder plus 16 language layers. Stage 1 runs the other 16 layers. Stage 0 is slower, so it decides the step time; stage 1 spends about half of every step waiting for it, on both systems.

Where stage 0's CPU thread spends each step:

| Stage 0, seconds per step | Bridge | rdsp | Bridge is slower by |
|---|---|---|---|
| **Vision encoder, forward pass** | 6.8 | 3.2 | **+3.6** |
| Language layers, forward pass | 2.7 | 2.4 | +0.3 |
| Loading the batch | 0.6 | 0.2 | +0.4 |
| Waiting after each send to stage 1 | 0.4 | 0.0 | +0.4 |
| Backward pass | 7.5 | 8.4 | **−0.9** |
| Other | 0.3 | 0.4 | −0.1 |
| **Total** | 18.2 | 14.6 | **+3.6** |

The vision encoder's forward pass is the whole gap. rdsp gives back 0.9 s in the backward pass.

### Inside the vision encoder

Bridge takes about twice as long as rdsp in every part of the encoder, not in one spot:

| Vision encoder forward, seconds per step | Bridge | rdsp | Gap |
|---|---|---|---|
| Attention: waiting for the GPU, once per layer | 1.43 | 0.79 | +0.64 |
| Attention: rotary position embedding | 1.01 | 0.41 | +0.60 |
| Attention: everything else | 2.44 | 1.10 | +1.34 |
| MLP | 1.28 | 0.54 | +0.75 |
| Outside the layers (patch embedding, merger) | 0.62 | 0.34 | +0.28 |
| **Total** | 6.79 | 3.18 | **+3.60** |

- **The wait (+0.6 s).** Both encoders make the CPU wait for the GPU once per layer, 1,728 times a step (27 layers × 64 microbatches).
  - Bridge: it stores the longest image length as a GPU number (`vision_model.py:386`). Transformer Engine compares it in an `if` before every attention call (`dot_product_attention/utils.py:320`).
  - rdsp: Hugging Face calls `lengths.tolist()` (`modeling_qwen3_5.py:1070`).
  - Bridge's wait costs more (1.4 s vs 0.8 s). Why is not measured.
- **The rest (+3.0 s): more CPU time per layer.** On this layout the GPUs are idle most of the time on both systems, so CPU time per layer is step time. Bridge's encoder layers run through Transformer Engine modules and `torch._dynamo` disable wrappers (visible in the flame graph); rdsp's are plain PyTorch layers from Hugging Face. Bridge spends about twice the time in each part. Which wrapper costs what is not split further.

### The backward pass (−0.9 s, rdsp is slower)

rdsp's thread spends 0.9 s more in the backward pass. rdsp also launches 1.6× more GPU kernels per step than Bridge (Nsight). A likely reason, not proven: more GPU work in backward, which the CPU thread waits for.

### Waiting after each send (+0.4 s)

Bridge's setting `batch_p2p_sync: true` makes the CPU wait for the GPU after each send to the other stage. rdsp sends without waiting.

## What this is not

- Not faster GPU kernels. rdsp launches 1.6× more GPU kernels than Bridge and does more GPU work.
- Not less work. Both pad to 2,048 tokens.
- Not the layer split. Both use 16 | 16.

On this layout, the GPUs are idle most of the time on both systems. The winner is the system whose CPU keeps the GPU busier, and the difference is almost all in the vision encoder.

## How we measured

| Tool | What it gave us |
|---|---|
| Normal 50-step runs | The step times (18.24 s vs 14.63 s) |
| py-spy (CPU sampler, low overhead) | Where each second of stage 0 goes (tables above). Flame graphs: `flame_bridge.svg`, `flame_rdsp.svg`, screenshots in `screens/` |
| Nsight Systems (GPU tracer) | Kernel counts and GPU work. It slows the CPU a lot, so we did not use it for timing. |
| Reading the source | The code behind the per-layer wait |

Files: `pyspy_*.txt` (`pyspy_vision_parts.txt` for the second table), `nsys_*.txt`, `cpu_sync_*.txt`, `idle_*.txt` in this folder.

## Limits

- One run per system, no repeats.
- py-spy sampled 2 of Bridge's 4 stage-0 GPUs (it failed on the other 2) and all 4 of rdsp's.
- py-spy sees Python frames only. Inside a part it cannot tell Python work from time blocked in a C call (a kernel launch, a wait).
- The "likely reason" for the backward pass is not tested.
- 16 | 16 is not balanced for either system. A split that gives stage 1 more layers would speed up both. Not tested.
