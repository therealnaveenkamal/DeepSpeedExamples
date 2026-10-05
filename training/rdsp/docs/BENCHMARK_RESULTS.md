# Benchmark results

rdsp against MegatronMIMO and Megatron-Bridge, training Qwen3.5-2B and 4B on CORD-v2. One AWS g7.48xlarge: 8× RTX PRO 4500 Blackwell, 32 GB, PCIe, no NVLink. Settings held equal, software versions and how to rerun: [REPRODUCE.md](REPRODUCE.md). Raw logs: `bench/published/{2b,4b}/`; `python bench/mimo/summarize.py bench/published/2b` prints the table rows, and `tests/unit/test_bench_summarize.py` checks them against the README.

## Results

Median over steps 6–50. Tokens/s counts real tokens only; both systems see the same tokens. Cost assumes $5.32/h for 8 GPUs, prorated to 5 for the MIMO layout.

| Model | Layout | Megatron | rdsp | Step time | Megatron tok/s | rdsp tok/s | Megatron $/M tok | rdsp $/M tok |
|---|---|---|---|---|---|---|---|---|
| 2B | Vision 1 GPU + language TP2×DP2, vs MIMO | 9.64 s | 8.30 s | −14% | 6,780 | 7,837 | 0.136 | 0.118 |
| 2B | TP2×PP2×DP2, vs Bridge | 7.39 s | 5.30 s | −28% | 8,711 | 12,187 | 0.170 | 0.121 |
| 4B | TP2×PP2×DP2, vs Bridge | 10.04 s | 8.69 s | −13% | 6,459 | 7,460 | 0.229 | 0.198 |
| 4B | TP4×PP2×DP1, vs Bridge | 15.93 s | 13.29 s | −17% | 4,073 | 4,901 | 0.363 | 0.302 |

Loss: every pair starts from the same step-0 loss. Over steps 1–49, rdsp's mean difference from Megatron is +0.0005 (2B MIMO layout), +0.003 (4B TP2×PP2×DP2) and −0.0024 (4B TP4×PP2×DP1). Steps 4–5 vary by up to 0.08 between repeated runs of either system, from GPU kernels that aren't bit-for-bit deterministic.

## rdsp settings per layout

All runs: `--prefetch --sharded-loss --drop-padding-mask --pad-per-microbatch --cuts balanced`, with the recipe in `recipes/`.

| Layout | Recipe | Extra settings | Cut chosen |
|---|---|---|---|
| Vision 1 GPU + TP2×DP2 | `qwen35-2b-vision1-tp2dp2.sh` | `--cuts 0` (vision-only stage), `--loss microbatch-mean`, 2 rows × 32 microbatches, stage 0 compiled | encoder alone |
| TP2×PP2×DP2 (2B) | `qwen35-2b-tp2pp2dp2.sh` | `--untie-embeddings`, 2 rows × 32 microbatches, vision encoder compiled | 5 of 24 layers on stage 0 |
| TP2×PP2×DP2 (4B) | `qwen35-4b-tp2pp2dp2.sh` | same | 13 of 32 |
| TP4×PP2×DP1 (4B) | `qwen35-4b-tp4pp2dp1.sh` | `--untie-embeddings`, 1 row × 64 microbatches, vision encoder compiled | 12 of 32 |

Megatron used its default even split (12 of 24, 16 of 32). It accepts uneven splits through flags; we didn't tune them.

## The balanced cut

In a two-stage pipeline the step runs at the pace of the slower stage. `--cuts balanced` estimates each part's work before training and minimises the most expensive stage, each stage's cost divided by its GPU count:

- A decoder layer costs its parameter count; so does the output head, on the last stage.
- The vision encoder costs its parameter count times a ratio measured from the first batch:

```
ratio = Σ_images patches × (1 + 2 × patches × width / layer_params) / text tokens computed
```

The second term is the encoder's attention over each image: about 4 × patches × width multiply-adds per patch, against 2 × layer_params for the matrix products. A CORD-v2 image has about 3,800 patches, so attention makes the encoder about 1.6× more expensive than its parameters suggest. "Text tokens computed" counts each microbatch at its padded length, not the configured 2048.

An earlier version divided by 2048 and left out attention. Both errors made the encoder look about half as expensive, so the cut landed too late. We checked the corrected estimate by timing every candidate cut for a few steps:

| Layout | Even | Old estimate | Balanced | Fastest measured |
|---|---|---|---|---|
| 2B TP2×PP2×DP2 | 12 | 11 (6.07 s) | 5 (5.30 s) | 5 |
| 4B TP2×PP2×DP2 | 16 | 16 (OOM) | 13 (8.69 s) | 13 |
| 4B TP4×PP2×DP1 | 16 | 16 (14.54 s) | 12 (13.29 s) | 11 (13.15 s) |

Under TP4 the estimate misses by one layer (about 1%). The encoder's small matrices lose efficiency when split four ways, and the estimate doesn't model that. The timing sweep was only a check and isn't part of rdsp.

## What moved the numbers

rdsp started 40% slower than MegatronMIMO in MIMO's own layout. 2B step time after each change:

| Change | Step time |
|---|---|
| Start: encoder shares the first stage, with recompute | 13.51 s |
| Vision-only first stage: frees enough memory to drop recompute | 11.3 s |
| Prefetch: the next step's data (about 576 MB of images) ships while this step runs; dispatch fell from 1.1 s to 1 ms | 10.3 s (12-step run) |
| `torch.compile` on the vision encoder: its forward and backward fell from 7.8 s to 6.7 s per step | 9.62 s |
| Split-vocabulary loss: TP ranks keep their slice of the 248k-wide logits instead of gathering 2 GB per microbatch | 9.25 s |
| No padding mask: rows pad only on the right, so causal attention ignores the padding anyway, and SDPA picks its flash kernel | 9.05 s |
| bf16 gradients freed after fp32 accumulation (2 bytes per parameter less, as in Megatron), and padding per microbatch (1.71 → 1.20 padded tokens per real token) | 8.30 s |

Same chain for 4B TP2×PP2×DP2: 13.51 s → 11.57 s → 10.58 s → 8.71 s → 8.69 s. The freed gradients removed the out-of-memory errors that had forced recompute, and the corrected balanced cut did the rest.

## Not included

- **4B in MIMO's layout.** Neither system fits it in 32 GB without recompute.
- **2B TP4.** Qwen3.5-2B has 2 key/value heads, which can't be split 4 ways.
- **9B and up.** These need 80 GB-class GPUs in these layouts.
- **Colocated vision.** rdsp only; neither Megatron system supports it for Qwen3.5. 4B ran at 9.66 s, slower than TP2×PP2×DP2; 2B didn't fit.
- **Repeats.** One 50-step run per configuration, except the rdsp 4B TP2×PP2×DP2 repeat (8.71 s and 8.69 s).
- **Interconnect.** On NVLink machines, rdsp's communication savings would matter less.

## Validation provenance

The supported-layout rows in [CONTRACTS.md](CONTRACTS.md) passed on Modal (8× L4, real DeepSpeed and NCCL, fp32 tiny models, SGD with momentum), with:
- loss parity against the unsplit model within 1e-4;
- per-parameter update parity within 2.4e-5 (3e-4 with AutoEP folding);
- checkpoint round trip;
- rank-kill recovery.

The two-stage baseline uses Qwen3-0.6B in bf16. The three vision placements match the unsplit model on 4× L40S (`tests/integration/test_vl_layouts_gpu.py`, tiny Qwen3.5-VL, fp32). Evidence per row is in `src/ray_deepspeed_pipeline/support_matrix.py`.
