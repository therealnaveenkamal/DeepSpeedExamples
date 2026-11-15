# 27B TP4×PP2×DP1: memory per GPU, before any run

Exact parameter counts from the model built on the meta device (`scripts/memory_count.py`), split as rdsp splits it after the GDN fix (Megatron splits the same tensors). 18 bytes per parameter: bf16 weight, fp32 gradient, fp32 master copy and Adam's two moments, nothing shared out (DP1).

| Cut | Stage 0 | Stage 1 |
|---|---|---|
| even 32 \| 32 (Megatron) | 3.51B params → 63.3 GB | 3.36B → 60.5 GB |
| 30 \| 34 | 3.33B → 59.9 GB | 3.55B → 63.9 GB |

Activations, estimated (no recompute, 2,048 tokens, TP4, no sequence parallel): about 168 MB per layer per microbatch. Under 1F1B stage 0 holds 2 microbatches of its layers (~10.7 GB) plus the vision encoder's (~3 GB); stage 1 holds 1 (~5.4 GB) plus the logits (~1 GB). Add ~3–4 GB for CUDA, NCCL and fragmentation.

| | Stage 0 | Stage 1 |
|---|---|---|
| Megatron, even cut | ~80 GB: likely out of memory | ~70 GB |
| rdsp, 30 \| 34 | ~77 GB: tight | ~74 GB |

So the 3-step preflight decides: if either system runs out of memory, both run with full recompute.
