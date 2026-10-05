# Contracts

What rdsp's public API guarantees, what it raises, and what it does not support. The docstrings in `src/ray_deepspeed_pipeline/` carry the same contracts at the call site.

## Public API

```python
import ray_deepspeed_pipeline as rdsp
```

| Name | Guarantees | Raises |
|---|---|---|
| `rdsp.initialize(model=, config=, loss_fn=, pipeline_config=, weights=None, training_data=None)` | Returns `(engine, None, dataloader_or_None, None)`. Every check runs before any Ray actor starts. With `weights`, each stage loads only its own tensors from the HF checkpoint. | `ValidationError`, `UnsupportedInV1` |
| `rdsp.PipelineConfig` | Lowered once into an immutable plan. The plan's hash binds checkpoints. | `ValidationError` |
| `rdsp.StageOverride(stage, num_gpus, tp, sp, ep, zero_stage, recompute, offload_optimizer, compile, compile_vision)` | One stage's grid: `num_gpus = dp × sp × tp`, with dp derived. Stages without an override get 1 GPU. | `ValidationError` (grid does not divide, ep does not divide `num_gpus`, tp does not divide the key/value heads) |
| `rdsp.ConnectionOverride(source, dest, conversion)` | The conversion on an edge is derived from the two layouts; a declared one must match. | `ValidationError` |
| `rdsp.UniformTransformerBlocks()` | Blocks split as evenly as the counts allow. | `ValidationError` |
| `rdsp.BalancedTransformerBlocks(vision_token_ratio=)` | Cut that minimises the most expensive stage, from parameter counts scaled by tokens processed. Measure the ratio with `partition.vision_token_ratio(config, batch)`. | `ValidationError` |
| `rdsp.ExplicitCuts((c1, ...))` | Stage i+1 starts at block `c_i`. A first cut of 0 gives the vision encoder its own stage. | `ValidationError` |
| `rdsp.ColocatedVision(recompute, compile, encode_per_microbatch)` | The encoder runs on every GPU. fp32 master weights and optimizer state are split over all ranks, and checkpoints hold the full state. | `ValidationError` (encoder output enters the decoder blocks, fp16) |
| `rdsp.TokenMeanLoss(sum_fn, count_fn, per_microbatch=False)` | The step loss is the mean over every counted token of the step. With `per_microbatch=True`, training averages per rank-microbatch (Megatron's default) and the report is still the token mean. | — |
| `rdsp.next_token_loss_sum(logits, labels)` | Summed next-token cross entropy, on full logits or on a vocab-parallel head's shard. As the loss, it lets TP stages skip gathering the 248k-wide logits. | — |
| `engine.train_batch(data_iter)` | Consumes exactly M entries (M = microbatches per step). Returns the step loss. | `StepFailed`, `PipelinePoisoned` |
| `engine.eval_batch(data_iter)` | Forward only. Weights unchanged. | `StepFailed`, `PipelinePoisoned` |
| `engine.save_checkpoint(dir, tag)` | All stages, then the manifest, written last and atomically. A failed save commits nothing. | `CheckpointError` |
| `engine.load_checkpoint(dir, tag)` | Verifies the manifest and every file's SHA-256, rebuilds dead actors, then loads. Clears a poisoned pipeline. | `CheckpointError` |
| `engine.stage_blocks` | Each stage's `[start, stop)` transformer blocks. | — |

## Data

- An entry is `(inputs, labels)`. Inputs are a tensor or a dict of tensors; vision models pass `pixel_values` and `image_grid_thw` as one tensor per row.
- A step reads exactly M entries and leaves the rest of the iterator untouched.
- Microbatches in one step may differ in sequence length. The stages then send each boundary with its shape.
- With `prefetch=True`, the next step's M entries are read while the current step runs. Pass the same iterator to every `train_batch` call. `initialize(training_data=...)` rejects prefetch. `load_checkpoint()` drops the step read ahead.

## Loss

- `loss_fn(outputs, labels)` returns a scalar for one rank's rows of one microbatch. The step loss is the mean over microbatches and data-parallel ranks.
- A loss with `takes_vocab_shards = True` (`next_token_loss_sum`, or a `TokenMeanLoss` built on it) may receive a TP rank's slice of the vocabulary. The logits tensor then carries `.vocab_shard`.

## Failures

| When | Raises | State after |
|---|---|---|
| A stage fails before any stage applies its update | `StepFailed` | Weights unchanged; links rebuilt. A retry discards leftovers and rewinds the RNG, so it is exact. |
| A stage fails while updates are being applied | `PipelinePoisoned` | Every call is refused until `load_checkpoint()`. |
| A checkpoint fails verification | `CheckpointError` | Nothing loaded. |

A checkpoint loads only into a pipeline with the same plan hash: same model split, layouts and microbatches. Settings that change how a step runs but not what a stage saves are left out of the hash: `prefetch`, `recompute`, `compile`, `compile_vision`, `encode_per_microbatch`.

## DeepSpeed settings rdsp changes

Every stage runs a stock DeepSpeed engine, with these settings changed:

- `gradient_accumulation_steps = 1`: the coordinator accumulates over the step's microbatches.
- `gradient_clipping = 0`: DeepSpeed would clip each stage by its own norm, not the global one. A nonzero value is rejected.
- `train_batch_size` is dropped: each stage's DeepSpeed derives it from its own grid.
- `bf16.immediate_grad_update = true` under bf16 with fp32 accumulation and ZeRO-1, unless the config sets it. Each bf16 gradient is added into fp32 as soon as it exists and then freed: 12 bytes of state per parameter instead of 14.

## Supported layouts

Validated against an unsplit model (loss parity, gradient parity, checkpoint round trip, rank-kill recovery). Evidence is in `src/ray_deepspeed_pipeline/support_matrix.py`.

| Row | Stages |
|---|---|
| two-stage-baseline | 1 GPU, 1 GPU (Qwen3-0.6B, bf16) |
| static-boundary | 1 GPU, dp=2, 1 GPU |
| resource-mesh | dp=2, dp=4, dp=2 |
| dp-zero | dp=4 ZeRO-2, dp=2 ZeRO-1, dp=2 ZeRO-0 |
| autotp | 1 GPU, tp=2, tp=2×dp=2 |
| autoep-folding | 1 GPU, ep=4, ep=2 folded with tp=2 |
| sequence-parallel | sp=2, sp=2×dp=2, 1 GPU |

The vision-language layouts benchmarked against Megatron (Qwen3.5-2B/4B) are listed in the README's results.

## Not supported

- TP above, or not dividing, the model's key/value heads (Qwen3.5-2B has 2: no TP4). Raises `ValidationError`.
- Parameters tied across stages. Untie them, or keep both users on one stage.
- Colocated vision for encoders whose features enter the decoder blocks (Qwen3-VL deepstack). The check reads `deepstack_visual_indexes`, so other ways of injecting features are not detected.
- fp16 with colocated vision.
- The 4-node composed layout (`four-stage-mixed`) is validated on one node only.
