# Reproducing the benchmark

rdsp against MegatronMIMO and Megatron-Bridge, on Qwen3.5-2B and 4B, in three layouts. One 8-GPU node, about 3 hours.

## Node

- 8 NVIDIA GPUs with at least 32 GB each. The published numbers come from an AWS g7.48xlarge: 8× RTX PRO 4500 Blackwell, 32 GB, PCIe, no NVLink.
- NVIDIA driver and CUDA, Docker with the NVIDIA container runtime.
- About 300 GB of free disk: the NeMo container, two models in Hugging Face and Megatron formats, and the exported samples.

## Setup

```bash
git clone <this repository> && cd DeepSpeedExamples/training/rdsp
bash bench/setup_node.sh
```

This builds `~/venv` (torch 2.13, DeepSpeed at commit `53a2ac44`, transformers 5.17.0, rdsp) and pulls `nvcr.io/nvidia/nemo:26.08`, which holds Megatron-Bridge and MegatronMIMO.

## Runs

`bench/mimo/bench.sh` runs any set of layouts, each pair back to back with Megatron first, and prints the table:

| Layout | Megatron side | rdsp recipe | GPUs |
|---|---|---|---|
| `noncoloc` | MegatronMIMO: vision on 1 GPU, language TP2×DP2 | `qwen35-<size>-vision1-tp2dp2.sh` | 5 |
| `tp2` | Megatron-Bridge TP2×PP2×DP2 | `qwen35-<size>-tp2pp2dp2.sh` | 8 |
| `tp4` | Megatron-Bridge TP4×PP2×DP1 | `qwen35-<size>-tp4pp2dp1.sh` | 8 |
| `coloc` | none (rdsp only): vision on every GPU, language TP2×PP2×DP2 | `qwen35-<size>-coloc-tp2pp2dp2.sh` | 8 |

```bash
MODEL=Qwen3.5-2B bash bench/mimo/bench.sh noncoloc tp2
MODEL=Qwen3.5-4B PRICE=5.32 bash bench/mimo/bench.sh all
MODEL=Qwen3.5-4B ROUNDS=2 bash bench/mimo/bench.sh tp2 tp4      # repeat for noise
DRY_RUN=1 MODEL=Qwen3.5-4B bash bench/mimo/bench.sh all         # print the plan only
```

The first run per model converts the checkpoint (the Megatron side needs its own format) and exports the training samples; rdsp replays the exact samples Megatron-Bridge's loader produces, in the same order. Both are reused afterwards. A layout without a recipe for the model is skipped (Qwen3.5-2B has no TP4: 2 key/value heads).

Each run is 50 steps (`STEPS`). Logs go to `~/bench/<model>/r<round>/<case>.log`, with `summary.txt` next to them: median step time over steps 6–50, real tokens/s, and rdsp's cost per million tokens at `PRICE` $/h for the node. Single cases still run through `bench/mimo/runs.sh <case>`.

## Expected results

| Model | Layout | Megatron | rdsp |
|---|---|---|---|
| 2B | Vision on 1 GPU + language TP2×DP2 (vs MIMO) | 9.64 s | 8.30 s |
| 2B | TP2×PP2×DP2 (vs Bridge) | 7.39 s | 5.30 s |
| 4B | TP2×PP2×DP2 (vs Bridge) | 10.04 s | 8.69 s |
| 4B | TP4×PP2×DP1 (vs Bridge) | 15.93 s | 13.29 s |

Noise:
- Repeated runs land within about 2% (rdsp 4B TP2×PP2×DP2: 8.71 s and 8.69 s).
- The 4B Megatron numbers come from a second machine of the same type.
- Around steps 4–5, losses vary by up to 0.08 between repeated runs on both systems, because some GPU kernels aren't bit-for-bit deterministic.

## Settings held equal

- Sequence length 2048, 64 rows per step, one row per GPU per microbatch.
- AdamW, learning rate 1e-5 constant, betas 0.9 / 0.95, eps 1e-8; no weight decay, clipping or warm-up.
- bf16 compute; fp32 gradients and optimizer state, split over data parallel.
- No activation recompute, 50 steps.
- Megatron defaults that differ are overridden in `runs.sh`: clipping, warm-up, cosine decay, the multi-token-prediction head, the shuffling sampler.

Differences that remain:
- **Loss averaging.** Megatron-Bridge trains on the per-token mean. MegatronMIMO trains on per-microbatch means and can't be changed, so the rdsp run against it does the same.
- **Padding.** Megatron-Bridge pads each row to a multiple of 128 tokens. MegatronMIMO pads every row to 2048 by default; `MIMO_EXTRA="--pad-to-seq-length false"` turns on its dynamic padding (9.60 s vs 9.64 s). rdsp pads each microbatch to its longest row.
- **Pipeline split.** Megatron splits the decoder layers evenly. rdsp computes its split from the model and the first batch (`--cuts balanced`).
- **Embeddings.** In the pipeline layouts, Megatron ties the input embedding and the output layer across stages; rdsp trains two copies.

## Cost

About 3 hours of node time for both models, roughly $16 at the $5.32/h spot price of a g7.48xlarge in us-east-1 (October 2026).
