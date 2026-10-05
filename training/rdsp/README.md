# rdsp: pipeline parallelism on Ray + DeepSpeed

rdsp (`ray_deepspeed_pipeline`) splits a model into pipeline stages and gives each stage its own GPU count and parallel layout. Every stage is a group of Ray actors, one per GPU, each running a stock DeepSpeed engine. Stages pass activations and gradients over NCCL in a 1F1B schedule. You can split a Hugging Face vision-language model so the vision encoder gets its own stage and layout, and train it from the HF checkpoint without converting it.

On 8 PCIe GPUs, rdsp takes 13–28% less time per training step (1.15–1.40× the throughput) than Megatron-Bridge and MegatronMIMO on Qwen3.5-2B and 4B, with the same layouts, data and settings.

## Terms

| Term | Meaning |
|---|---|
| Stage | A contiguous slice of the model on its own group of GPUs. |
| PP | Pipeline parallelism: the number of stages. |
| TP | Tensor parallelism: each layer's matrices split across GPUs. |
| DP | Data parallelism: copies of a stage, each on different rows. ZeRO-1 splits the optimizer state across them. |
| Cut | The decoder layer where one stage ends and the next begins. |
| Microbatch | The rows one stage processes at a time. A step is many microbatches, with gradients accumulated across them. |
| TP2×PP2×DP2 | 2 stages, each 4 GPUs as TP2 × DP2: 8 GPUs. |

## Results

Median step time over steps 6–50. Lower is better.

| Model | Layout | GPUs | Megatron | rdsp | Step time |
|---|---|---|---|---|---|
| Qwen3.5-2B | Vision on 1 GPU + language TP2×DP2, vs MegatronMIMO | 5 | 9.64 s | 8.30 s | −14% |
| Qwen3.5-2B | TP2×PP2×DP2, vs Megatron-Bridge | 8 | 7.39 s | 5.30 s | −28% |
| Qwen3.5-4B | TP2×PP2×DP2, vs Megatron-Bridge | 8 | 10.04 s | 8.69 s | −13% |
| Qwen3.5-4B | TP4×PP2×DP1, vs Megatron-Bridge | 8 | 15.93 s | 13.29 s | −17% |

Throughput at the same steps, in real (non-padding) tokens/s: 7,837 vs 6,780; 12,187 vs 8,711; 7,460 vs 6,459; 4,901 vs 4,073 (rdsp first). Qwen3.5-2B can't run TP4: it has 2 key/value heads.

**Hardware.** AWS g7.48xlarge: 8× NVIDIA RTX PRO 4500 Blackwell (32 GB) on PCIe, no NVLink, so every TP all-reduce and stage-to-stage transfer goes over PCIe. $5.32/h spot in us-east-1.

**Software.** torch 2.13, DeepSpeed 0.19.3 at `53a2ac44`, transformers 5.17.0. Megatron side: `nvcr.io/nvidia/nemo:26.08`, NVIDIA's training scripts unmodified.

**Held equal.**
- Data: CORD-v2. rdsp replays the exact samples Megatron-Bridge's loader produces, in the same order.
- Batches: sequence length 2048, 64 rows per step, one row per GPU per microbatch.
- Optimizer: AdamW, lr 1e-5 constant, betas 0.9/0.95, eps 1e-8, no weight decay, clipping or warm-up.
- Precision: bf16 compute, fp32 gradients and optimizer state split over data parallel.
- No activation recompute.

**Not equal.**
- Pipeline split: Megatron splits decoder layers evenly. rdsp picks the cut from the model and the first batch (`--cuts balanced`).
- Padding: rdsp pads each microbatch to its longest row. Megatron-Bridge pads rows to multiples of 128. MegatronMIMO pads to 2048; its dynamic padding gives 9.60 s instead of 9.64 s.
- Loss averaging: the MegatronMIMO run trains on per-microbatch means (its only mode), so the rdsp run against it does the same. The Bridge runs train on the per-token mean.
- Embeddings: in the pipeline layouts, Megatron ties the input embedding and output layer across stages. rdsp trains two copies.

Logs behind every number: `bench/published/`. How to rerun all of it: [docs/REPRODUCE.md](docs/REPRODUCE.md). What moved the numbers: [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

## Install

Linux, CUDA, Python 3.12.

```bash
uv venv -p 3.12 && . .venv/bin/activate
uv pip install "torch==2.13.*" torchvision "transformers==5.17.0" ray cloudpickle accelerate safetensors datasets pillow
DS_BUILD_OPS=0 uv pip install --no-build-isolation \
    "git+https://github.com/deepspeedai/DeepSpeed.git@53a2ac44fb664bea838df3981ba4366b91643070"
uv pip install -e ".[dev]"
uv pip install flash-linear-attention liger-kernel && uv pip install --no-build-isolation causal-conv1d
```

Importing `ray_deepspeed_pipeline` doesn't import Ray or DeepSpeed; only the stage actors do.

## Quickstart

```bash
./recipes/qwen35-2b-tp2pp2dp2.sh                  # Qwen3.5-2B on CORD-v2, 8 GPUs
./recipes/qwen35-2b-tp2pp2dp2.sh --steps 200      # extra flags override the recipe
```

`recipes/` has one script per benchmarked layout, plus a text-only Qwen3 recipe. [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) covers the flags, your own data, and common failures.

## Python API

```python
import ray, ray_deepspeed_pipeline as rdsp

def count_tokens(labels):            # tokens the loss is computed on
    return int((labels[:, 1:] != -100).sum())

ray.init()
engine, _, _, _ = rdsp.initialize(
    model=model,                    # HF model, built under accelerate.init_empty_weights()
    weights="/path/to/hf/checkpoint",  # each stage loads only its own tensors
    config=ds_config,               # DeepSpeed config dict
    loss_fn=rdsp.TokenMeanLoss(rdsp.next_token_loss_sum, count_tokens),  # mean over the step's tokens
    pipeline_config=rdsp.PipelineConfig(
        stages=2,
        partition=rdsp.BalancedTransformerBlocks(vision_token_ratio=ratio),
        stage_overrides=(rdsp.StageOverride(stage=0, num_gpus=4, tp=2, zero_stage=1),
                         rdsp.StageOverride(stage=1, num_gpus=4, tp=2, zero_stage=1)),
    ),
)
loss = engine.train_batch(data_iter)   # consumes exactly M (inputs, labels) entries
engine.save_checkpoint("ckpt/", tag="step100")
```

If the input embedding and output layer share a weight and land on different stages, untie them first (`train_vl.py --untie-embeddings`); on one stage they stay tied. Every public name, what it guarantees and what it raises are listed in [docs/CONTRACTS.md](docs/CONTRACTS.md). `train_vl.py` is a complete training script.

## How it works

- **Planning.** `rdsp.initialize` lowers the config into an immutable plan: cuts, each stage's grid (`gpus = dp × sp × tp`), each stage's DeepSpeed config, and a hash that binds checkpoints. Every check, including the cluster's GPU count, runs before any actor starts.
- **Stages.** Each stage runs the HF model's own `forward`. Blocks outside the stage pass their input through, and a hook injects what arrived from upstream. No per-architecture code is needed.
- **Step.** The driver hands each stage its 1F1B op list. Stages exchange activations and gradients directly over NCCL. Each stage applies its optimizer step only after every stage reports its backward done.
- **Failures.** A failure before any update raises `StepFailed`: weights are unchanged and a retry is exact. A failure during the update raises `PipelinePoisoned` until a checkpoint is loaded.
- **Vision encoder.** There are three placements:
  - on the first stage, sharing its layout;
  - on its own stage, via a first cut of 0;
  - colocated on every GPU.
- **Balanced cuts.** `--cuts balanced` charges the vision encoder per image patch. It measures patches per text token from the first batch, then picks the cut that minimises the slowest stage.
- **Memory.** Each bf16 gradient is folded into the fp32 accumulator as soon as it exists, then freed, saving 2 bytes per parameter (12 instead of 14 at DP=2, as in Megatron). Each microbatch is padded only to its own longest row.

Design and the reasoning behind it: [docs/ENGINEERING.md](docs/ENGINEERING.md).

## Model coverage

| Builder | Models | Validation |
|---|---|---|
| `HFModelStage` | Any HF model with a `config`: Llama, Mistral, Qwen2/3/3.5, Qwen3-MoE, Mixtral, Gemma 1–3, Phi-3, OLMo2, Granite, Cohere, StarCoder2, StableLM, DeepSeek-V3, GLM-4/4-MoE/5.3, GPT-2, GPT-NeoX, Falcon, Bloom, Mamba, Qwen3-VL, Qwen3.5-VL | Pipeline loss equals the unsplit model's on CPU (`tests/integration/test_hf_families.py`). Qwen3, Qwen3-MoE and Qwen3.5-VL validated on GPU. |
| `CausalLMStage` | Llama, Qwen3, Qwen3-MoE | GPU-validated in every layout in CONTRACTS.md |
| `GenericSequentialStage` | Plain sequential models | CPU |

TP inside a stage uses the model's AutoTP plan. Configs without one get a plan for the standard projection names. Qwen3-VL injects vision features into its first decoder blocks, so its encoder can't be colocated and the first cut must come after them.

## Tests

```bash
pytest -q          # CPU: full runtime on Ray with a stub engine (~430 tests); GPU tests skip
ruff check .
```

The GPU tests (parity against the unsplit model, checkpoint round trip, rank-kill recovery) run on Modal: `scripts/modal_tests.py`.

## Docs

| | |
|---|---|
| [GETTING_STARTED](docs/GETTING_STARTED.md) | Install, recipes, flags, your own data, common failures |
| [REPRODUCE](docs/REPRODUCE.md) | Rerun the benchmark |
| [CONTRACTS](docs/CONTRACTS.md) | Public API guarantees, errors, supported layouts |
| [ENGINEERING](docs/ENGINEERING.md) | Design |
| [BENCHMARK_RESULTS](docs/BENCHMARK_RESULTS.md) | Per-layout settings, what moved the numbers |

## Limitations

- **Validation.**
  - Benchmarked on one 8-GPU node.
  - The multi-node path is tested on Ray but not on real hardware.
- **Schedule.**
  - 1F1B only, no interleaved stages.
  - No global-norm gradient clipping: a nonzero `gradient_clipping` is rejected.
- **Weights.** No parameters tied across stages.
- **Colocated vision.** Matches the unsplit model, but isn't fast yet on 32 GB GPUs: Qwen3.5-4B TP2×PP2×DP2 takes 9.66 s per step against 8.69 s with the encoder on the first stage.
- **Balanced cuts.**
  - The cost model ignores communication.
  - It counts MoE experts in full, not by the active fraction.
  - `ExplicitCuts` overrides it.
- **Checkpoints.** Node-local: they survive actor loss, not node loss.
