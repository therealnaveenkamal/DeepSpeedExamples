# Getting started

Train a model with rdsp on one multi-GPU node.

## Install

Linux, CUDA GPUs, Python 3.12. `bench/setup_node.sh` does all of this on a fresh node. By hand:

```bash
uv venv -p 3.12 && . .venv/bin/activate
uv pip install "torch==2.13.*" torchvision "transformers==5.17.0" ray cloudpickle accelerate safetensors datasets pillow
DS_BUILD_OPS=0 uv pip install --no-build-isolation \
    "git+https://github.com/deepspeedai/DeepSpeed.git@53a2ac44fb664bea838df3981ba4366b91643070"
uv pip install -e .
uv pip install flash-linear-attention liger-kernel   # Qwen3.5 linear attention, fused kernels
uv pip install --no-build-isolation causal-conv1d
```

The DeepSpeed commit is the one every result was measured with.

## Run a recipe

`recipes/` has one script per validated layout:

| Recipe | Model | Layout | GPUs |
|---|---|---|---|
| `qwen35-2b-vision1-tp2dp2.sh` | Qwen3.5-2B | vision on 1 GPU + language TP2×DP2 | 5 |
| `qwen35-2b-tp2pp2dp2.sh` | Qwen3.5-2B | TP2×PP2×DP2 | 8 |
| `qwen35-4b-tp2pp2dp2.sh` | Qwen3.5-4B | TP2×PP2×DP2 | 8 |
| `qwen35-4b-tp4pp2dp1.sh` | Qwen3.5-4B | TP4×PP2×DP1 | 8 |
| `qwen35-4b-vision1-tp2dp2.sh` | Qwen3.5-4B | vision on 1 GPU + language TP2×DP2 (80 GB GPUs) | 5 |
| `qwen35-4b-coloc-tp2pp2dp2.sh` | Qwen3.5-4B | vision on every GPU + language TP2×PP2×DP2 (80 GB GPUs) | 8 |
| `qwen3-text-pp.sh` | Qwen3-0.6B | DP2 (ZeRO-2) → TP2 | 4 |

```bash
./recipes/qwen35-2b-tp2pp2dp2.sh
./recipes/qwen35-2b-tp2pp2dp2.sh --steps 200 --lr 2e-5   # extra flags override the recipe
```

Each step prints one line:

```
step 12 loss 0.0931 5312 ms 12187 real tok/s 24672 padded tok/s real 64725 supervised 12304
```

`real` counts tokens that aren't padding; `supervised` counts the tokens the loss is computed on. The first 2–5 steps include compilation and are slower.

## Change the model, data or layout

**Model.** `--model` takes a Hugging Face id. Vision-language models go through `train_vl.py`, text models through `train.py`. Each stage loads only its own weights.

**Data.** `train_vl.py --dataset`:

- `cord-v2`: downloads CORD-v2, scales images to at most `--max-pixels`, and pads rows to `--seq` (with `--pad-multiple N --pad-per-microbatch`, each microbatch to its longest row, as the recipes do).
- `exported:DIR`: steps exported from Megatron-Bridge's loader (`bench/mimo/runs.sh export`). With `--pad-multiple 128 --pad-per-microbatch`, each microbatch is padded to its longest row.
- `synthetic`: random images, for smoke tests.

For your own data, write a loader that yields `(inputs, labels)` entries (see `docs/CONTRACTS.md`, Data) and call `rdsp.initialize(...)` directly. `train_vl.py` is a complete example.

**Layout.** Flags:

- `--stages N`: pipeline stages.
- `--stage S:gpus=G,tp=T,zero=Z`: stage S's GPU count, tensor parallelism and ZeRO stage. The data-parallel degree is G / T. Other keys: `sp`, `ep`, `recompute=1`, `offload=1`, `compile=1`, `compile_vision=1`.
- `--cuts balanced` (default): rdsp estimates where to cut from the model and the first batch. `--cuts even` splits the decoder layers evenly; `--cuts 5,18` sets them by hand. A first cut of `0` puts the vision encoder alone on stage 0.
- `--colocated-vision`: run the vision encoder on every GPU instead of on stage 0.

`docs/CONTRACTS.md` lists the validated layouts and what isn't supported.

## Common failures

**Not enough GPUs.** The layout asks for more GPUs than the node has:

```
ValidationError: the pipeline needs 8 GPUs; this Ray cluster has 4. Reduce gpus= in the stage layouts, or add GPUs
```

On a multi-node cluster with enough GPUs in total, a stage that doesn't fit on one node fails after `RDSP_PLACEMENT_TIMEOUT_S` (default 600 s) with `stage S: cannot place G GPUs (STRICT_PACK) in this Ray cluster`.

**TP doesn't divide the key/value heads.** Qwen3.5-2B has 2 key/value heads:

```
ValidationError: stage 0: tp=4 does not divide the model's 2 key/value heads; use a tp that divides 2
```

**Out of memory.** `torch.OutOfMemoryError` on the first or second step, in order of cost:

1. Move layers off the full stage: stage 0 holds the vision encoder plus the most microbatches in flight, so try a lower first cut, e.g. `--cuts 4`.
2. Add data parallelism so ZeRO splits the optimizer state over more GPUs.
3. `recompute=1` on the full stage. This keeps only each block's input and recomputes the rest, about 30% more compute on that stage.

**`--drop-padding-mask needs rows padded on the right only`.** Your data has padding before real tokens. Drop the flag.
