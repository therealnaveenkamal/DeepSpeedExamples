# rdsp: pipeline parallelism on Ray + DeepSpeed with per-stage layouts

`ray_deepspeed_pipeline` splits a model into pipeline stages. Each stage is a
group of Ray actors (one per GPU) running an unmodified DeepSpeed engine, and
each stage has its own GPU count and its own DP / TP / SP / EP / ZeRO settings.
DeepSpeed and Ray are used through public APIs only.

## Install

```bash
pip install -e ".[dev]"      # rdsp, pytest, ruff
pip install deepspeed transformers datasets
```

Python ≥ 3.12. Importing `ray_deepspeed_pipeline` does not import `ray` or
`deepspeed`; the actors do.

## Run

```bash
./run.sh                                  # Qwen3-0.6B, 4 GPUs: [DP=2, ZeRO-2] -> [TP=2]
python train.py --stages 4                # 4 single-GPU stages, even layer split
python train.py --stages 4 --cuts 7,14,21 \
    --stage 0:gpus=2,zero=2 --stage 1:gpus=2,tp=2 \
    --stage 2:gpus=2,sp=2  --stage 3:gpus=2          # 8 GPUs, four layouts (not yet run at this size)
python demo.py                            # CPU only, ~10 s, pipeline vs unsplit model
```

`train.py` flags:

| Flag | Default | |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | HF causal LM; tied embeddings are untied automatically |
| `--stages` | 2 | pipeline stages |
| `--cuts` | even split | block index where each stage after the first starts, e.g. `7,14,21`; `balanced`: cost-balanced |
| `--stage` | 1 GPU, no parallelism | per-stage layout, repeatable: `<i>:gpus=N,zero=Z,tp=T,sp=S,ep=E,fold=1,recompute=1` |
| `--microbatches` | 8 | microbatches per optimizer step |
| `--rows` | 4 | sequences per microbatch |
| `--seq` | 512 | tokens per sequence |
| `--steps` | 100 | optimizer steps |
| `--lr` | 1e-5 | AdamW learning rate |
| `--zero` | 0 | ZeRO stage for stages without `zero=` |
| `--dtype` | `bf16` | `bf16` or `fp32` |
| `--data` | `wikitext` | `wikitext` (WikiText-103) or `synthetic` (random tokens) |
| `--deepspeed_config` | — | JSON file replacing the generated config; `train_batch_size` must equal `--rows` × `--microbatches` |
| `--save_dir` | — | write a checkpoint after the last step |

Output: one line per step with loss, step time and tokens/s.

## API

```python
import ray, ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.config import StageOverride

ray.init()
engine, _, _, _ = rdsp.initialize(
    model=model,                      # HF causal LM, embeddings untied
    config=ds_config,                 # DeepSpeed config dict or JSON path
    loss_fn=loss_fn,                  # loss_fn(logits, labels), last stage only
    pipeline_config=rdsp.PipelineConfig(
        stages=2,
        partition=rdsp.UniformTransformerBlocks(),      # or rdsp.ExplicitCuts((14,))
        stage_overrides=(StageOverride(stage=0, num_gpus=2, zero_stage=2),
                         StageOverride(stage=1, num_gpus=2, tp=2)),
    ),
)
loss = engine.train_batch(data_iter=it)   # it yields (input_ids, labels); consumes M per call
engine.save_checkpoint("ckpt/")           # engine.load_checkpoint("ckpt/")
```

| `PipelineConfig` field | |
|---|---|
| `stages` | number of stages |
| `partition` | `UniformTransformerBlocks()` (equal layer counts), `BalancedTransformerBlocks()` (equal estimated cost), or `ExplicitCuts((i, ...))` |
| `microbatches` | optional; must equal `gradient_accumulation_steps` |
| `stage_overrides` | `StageOverride(stage, num_gpus=1, zero_stage=None, tp=1, sp=1, ep=1, fold=False, recompute=False)`; `recompute`: rerun each block in backward instead of keeping its activations |
| `checkpoint` | `CheckpointPolicy(save_optimizer_state=True)` |

DeepSpeed config rules:

| Key | Rule |
|---|---|
| `gradient_accumulation_steps` | = microbatches per step (M) |
| `train_batch_size` | = M × rows per microbatch |
| `train_micro_batch_size_per_gpu` | set per stage by rdsp (rows / stage DP degree) |
| `gradient_clipping` | must be 0 (per-stage clipping ≠ global clipping) |
| pipeline / Ray keys | rejected |

Engine surface: `train_batch`, `eval_batch`, `save_checkpoint`, `load_checkpoint`,
`global_steps`. `forward`, `backward`, `step` and `module` raise
`UnsupportedEngineMethod`.

## Architecture

| Component | Process | Role |
|---|---|---|
| `compiler.lower()` | driver | block list → stage cuts, per-stage grid (`gpus = dp·sp·tp`), per-stage DeepSpeed config, plan hash |
| `schedule.generate_commands()` | driver | per-stage 1F1B op list; stage *s* of *N* does `min(N−1−s, M)` warm-up forwards |
| `coordinator` | driver | one step: data, dispatch, ready barrier, apply, failure handling, checkpoints |
| `stage_group` | driver | placement groups (`STRICT_PACK`, one node per stage), actor start-up, mesh check, link set-up |
| `stage_worker.StageWorkerActor` | 1 per GPU | `run_step()`: executes the op list, exchanges tensors with neighbours |
| `deepspeed_adapter` | 1 per GPU | wraps `deepspeed.initialize()`; forward, external-gradient backward, apply, save/load |
| `p2p` | 1 per GPU | two `ProcessGroupNCCL`s over all ranks of all stages: `fwd` (activations), `bwd` (gradients) |
| `boundary` | 1 per GPU | which rank sends which rows to whom; gradient scale |

Process groups per GPU: DeepSpeed's stage-local world (intra-stage collectives)
plus rdsp's `fwd` and `bwd` groups (inter-stage point-to-point). Rendezvous for
`fwd`/`bwd` is a `TCPStore` in stage 0 rank 0.

Step sequence:

1. Driver takes exactly M `(inputs, labels)` entries; short iterator → `StepFailed`, nothing dispatched.
2. Generation id += 1; one Ray call per rank: `run_step(generation, ops, inputs|labels)`.
3. Per op, forward: `recv(fwd)` from overlapping upstream cells → assemble → `x.detach().requires_grad_()` → `engine(x)` → `send(fwd)` (async).
4. Per op, backward: `recv(bwd)` → assemble → × `dp_s / dp_(s+1)` → `engine.backward((out * g).sum())` → `send(bwd, x.grad)`. Last stage: `engine.backward(loss / M)`.
5. Only the M-th backward is a DeepSpeed accumulation boundary (intra-stage gradient reduction happens there).
6. Each rank returns `{losses (last stage), ready}`. All ready → driver sends `apply` → `engine.step()` once per rank.

Boundary rules:

- A rank's cell = (row block of `rows / dp`, sequence block of `seq / sp`); TP ranks share a cell.
- Only TP rank 0 of a cell sends; every rank of an overlapping destination cell receives.
- First message per (direction, peer) per step carries a 10-int shape header.
- Gradient scale `dp_s / dp_(s+1)`: DeepSpeed averages over DP and sums over SP.

Engine settings rdsp overrides per stage: `gradient_accumulation_steps = 1`
(rdsp accumulates; loss scaled by 1/M), `gradient_clipping = 0`.

## Failure semantics

| When | Result | State after |
|---|---|---|
| Before any stage applies (exception, dead actor, p2p timeout) | `StepFailed` | weights unchanged; links rebuilt; retry discards leftovers and rewinds RNG, so it is exact |
| During apply | `PipelinePoisoned` | all calls refused until `load_checkpoint()` |

Checkpoint layout: `<dir>/<tag>/stage<s>/` (DeepSpeed files + `rng/rank<r>.pt`),
`<dir>/<tag>/manifest.json` (written last, atomic; plan hash, step, data
position, SHA-256 per file), `<dir>/latest`. Load verifies the manifest, rebuilds
all actors if any died (preferring previous nodes), verifies each stage's files
on its node, then loads.

## Supported layouts

Parity against the unsplit model on one device, 8×L4, real DeepSpeed + NCCL
(fp32, SGD with momentum, learnable data, per-parameter update comparison):

| Layout | GPUs per stage | Tolerance |
|---|---|---|
| Single-GPU stages, Qwen3-0.6B bf16 | 1 · 1 | bf16 rounding |
| Row-splitting boundary | 1 · 2 · 1 | loss 1e-4 |
| Uneven GPU counts | 2 · 4 · 2 | loss 1e-4 |
| Mixed ZeRO | 4 (Z2) · 2 (Z1) · 2 (Z0) | loss 1e-4 |
| AutoTP | 1 · 2 · 4 (TP×DP) | update 2.4e-5 |
| Ulysses SP | 2 · 4 (SP×DP) · 1 | update 2.4e-5 |
| AutoEP, folded onto TP | 1 · 4 · 2 | update 3e-4 |
| SP · TP · EP+TP · TP | 2 · 2 · 2 · 2 | update 2.4e-5 |

Constraints: SP not on the last stage; TP and SP not in the same stage; rows
divisible by every stage's DP degree; no parameters tied across stages; no
gradient clipping.

## Tests

```bash
pytest -q            # CPU: full runtime on Ray with a torch stub engine (336 tests, ~6 min)
ruff check .
modal run scripts/modal_tests.py --gpus L4:8 \
    --tests "tests/integration/test_p6_first_row.py tests/integration/test_p7_checkpoint_gpu.py tests/integration/test_heterogeneous_pipeline.py"
```

GPU tests skip on CPU and run on Modal (billed). `scripts/modal_cluster.py` runs
multi-node layouts.

## Results (Qwen3-0.6B, H100, vs Megatron-Core 0.19.2)

Same pretrained weights, WikiText-103, matched fused kernels and gradient
precision; both systems back to back on one pinned host.

| | 1 GPU | 4 GPUs | 8 GPUs |
|---|---|---|---|
| Loss difference per step (median) | 0.05% | 0.05% | 0.05% |
| Throughput ratio, 8,192-token microbatches | 1.10× | 1.11× | 1.02× |
| Throughput ratio, 2,048-token microbatches | 1.24× | 1.30× | 1.29× |

Balanced 4- and 5-stage layouts: 1.18×. A 2-GPU output stage vs a plain extra
stage at 0.6B: 0.91–0.97×. Details: `docs/BENCHMARK_RESULTS.md`,
`docs/presentation/rdsp-report.html`, raw data in `bench/results/`.

## Limitations

- Single node, ≤ 8 GPUs validated; multi-node path untested on real hardware.
- 1F1B only; no interleaved stages.
- No global-norm gradient clipping.
- `BalancedTransformerBlocks` estimates cost from parameter counts (embeddings free, vision encoder on the first stage, head on the last). It reproduces the hand-picked 9·9·9·1 for Qwen3-0.6B on 4 stages and gives 7·9·9·8·8·8·8·7 for Qwen3-VL-32B on 8; the vision encoder's real cost grows with image resolution, and the output head measured at about 5–6 layers of time against its 10-layer parameter estimate, so measure and tune with `ExplicitCuts`. MoE experts are counted in full, not by the active fraction.
- Node-local checkpoints survive actor loss, not node loss.

### Model coverage

Automatic split: longest `ModuleList` of same-class modules, cut by layer count
(or `ExplicitCuts`). Stage builders:

| Model | Builder | Status |
|---|---|---|
| Llama, Qwen3, Qwen3-MoE with SDPA/flash attention | `CausalLMStage` | GPU-validated, all intra-stage layouts |
| Every other HF model (`config` attribute) | `HFModelStage` (`hf_stage.py`) | CPU-tested, DP/ZeRO stages only |
| Plain sequential (each block takes only the previous output) | `GenericSequentialStage` | supported |

`HFModelStage` families trained on CPU through `rdsp.initialize()` (3 stages,
middle one data-parallel, loss equal to the unsplit model over 2 steps,
`tests/integration/test_hf_families.py`): Llama, Mistral, Qwen2, Qwen3,
Qwen3-MoE, Qwen3.5, Mixtral, Gemma, Gemma2, Gemma3, Phi-3, OLMo2, Granite,
Cohere, StarCoder2, StableLM, DeepSeek-V3, GLM-4, GLM-4-MoE, GPT-2, GPT-NeoX,
Falcon, Bloom, Mamba; plus Qwen3-VL (`test_vl_pipeline.py`).

How it works: every stage runs the model's own `forward`. Blocks outside the
stage return their input (shaped like the real blocks' output); modules whose
parameters the stage does not own return their input, or zeros for an
embedding. A pre-hook on each block injects what arrived from upstream; the
stage's output is what the loop hands the next block, so work the model does
between blocks counts and work after the last block (norm, head,
soft-capping) does not.

Boundary: the hidden state plus the tensor arguments the model passes to each
downstream block (rotary tables, masks, position ids, ALiBi); blocks receiving
the same tensors share one entry. Extra traffic per token over the hidden
state: 2 × head_dim values plus position ids, about 5% for Qwen3-VL-32B
(head_dim 128, hidden 5120). Eager attention also sends a `[rows, 1, seq, seq]`
mask; SDPA sends none. CPU cost of the hooks: about 0.1 ms per stage and
microbatch.

VL inputs: stage-0 inputs may be a dict. Row-shaped values are split per rank
like any tensor; values that are not (`pixel_values`, `image_grid_thw`) are
given as per-row lists and concatenated per rank. Qwen3-VL adds vision features
inside its first blocks (deepstack), so the first cut must come after them.

Per-stage weights: build the driver model with `accelerate.init_empty_weights()`
and pass `rdsp.initialize(..., weights="<HF checkpoint dir>")`. Each stage reads
only its own tensors from the safetensors files; the driver holds no weights.

Not yet supported with `HFModelStage`: TP, SP and EP inside a stage (AutoTP
plan and HF config are not attached); blocks that return more than the hidden
state for the next block to use (GLM-5.3's `(hidden, topk_indices)`).

## Layout

```
train.py, run.sh      training example
demo.py               CPU walkthrough
src/ray_deepspeed_pipeline/
  api.py config.py compiler.py plan.py partition.py     planning
  schedule.py coordinator.py engine.py data.py           driver-side step
  stage_group.py                                         start-up, per-stage client
  stage_worker.py deepspeed_adapter.py p2p.py boundary.py per-GPU worker
  hf_stage.py                                            HF model stages, per-stage weights
  checkpoint.py protocols.py errors.py support_matrix.py
tests/                unit, contract, architecture, integration (GPU tests skip on CPU)
scripts/              Modal harnesses for GPU tests
bench/                Megatron-Core comparison, results
docs/                 ENGINEERING.md, BENCHMARK_RESULTS.md, CODING_STANDARDS.md, presentation/
```
