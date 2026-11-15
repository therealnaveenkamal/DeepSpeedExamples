# TorchTitan for Qwen3.5-27B: code-reading pass (2026-10-12)

Read at pytorch/torchtitan `3c36a77` (2026-10-12). Nothing run. Paths relative to the repo root.
Claims marked ✔ were re-checked by hand against the code; the rest come from the research pass.

## Normal layout: TP4 x PP2 x DP1, vision on stage 0

| Need | Supported? | Where |
|---|---|---|
| Vision encoder on stage 0 | yes, automatic ✔ | `models/qwen3_5/model.py:309` (`pipeline_first_stage_module_fqns = ("vision_encoder",)`) |
| 32 \| 32 layer split | yes, the default | `distributed/pipeline_parallel.py:515-581` |
| 1F1B, 64 microbatches | yes | `config/parallelism.py:114`, `num_pp_microbatches=64` |
| TP4 (4 KV heads) | yes | `models/qwen3_5/model.py:404-421`, `config/validation.py:73-77` |
| Tested anywhere? | **no**: no dense-model PP test, no TP4 test, nothing at 27B | `tests/integration_tests/models.py:308-320` |

## Settings to match our equal-work runs
- Parallelism: dp_shard 1, TP 4 + sequence parallel, PP 2, 1F1B, 64 microbatches of 2048 tokens.
- Training: fp32 model (master weights), bf16 compute, fp32 gradient reduce; **CUDA graphs off** (on by default; reject varying image sizes).
- AdamW: lr 1e-5, betas 0.9/0.95, eps 1e-8, **weight decay 0** (default 0.1), fused, fp32 moments.
- LR schedule: **warmup 0** (default 200), no decay.
- Clipping: **`max_norm=float("inf")`** ✔. Default 1.0 (`components/optim/optim.py:42`); 0 passes validation (`:47-49`) but clipping always runs (`:117`), and PyTorch's clip coefficient max_norm/(norm+eps) would zero every gradient.
- Loss: already token mean over the whole step ✔ (`components/loss.py:354-355`; trainer counts labels across all microbatches first).
- Recompute: default is selective activation checkpointing; set none to match Bridge/rdsp.
- Weights: `scripts/download_hf_assets.py`, `initial_load_in_hf=True`. The 27B flavor matches HF config field for field; embeddings untied.

## Stage-0 memory per GPU (estimate)
3.48B params per GPU. FSDP is applied even at dp_shard=1; with PP it keeps the bf16 compute copy through forward and backward ✔ (`distributed/fsdp.py:182-185`).

| fp32 master | bf16 copy | fp32 grads | fp32 Adam | states |
|---|---|---|---|---|
| 4 B | 2 B | 4 B | 8 B | 18 B/param = 58.3 GiB |

Plus activations: with sequence parallel ~70-76 GiB total (tight), without ~73-79 GiB (likely out of memory).

## Code we must write (~150-200 lines)
1. CORD sample processor (~60-90 lines): image placeholders + answer, prompt labels -100. The shipped multimodal path is caption-style (masks only image tokens, `components/data/multimodal/mm_datasets.py:97-105`).
2. Dataset entry (~10 lines): HF source `naver-clova-ix/cord-v2`, same pixel budget as the others. TorchTitan resizes images itself (torchvision bicubic), so pixels may differ slightly from the HF processor.
3. Recipe function (~70 lines) with the settings above.

## Differences by design
- GatedDeltaNet kernels: Attention Gym ("fused" chunk backend on H100), not flash-linear-attention.
- Full attention: flex attention (autotuned, slow first compile); SDPA removed.
- Padding is its own "document": flex attention skips real-to-pad blocks (16 of 64 layers are full attention), slightly less attention work than Bridge/rdsp.
- Needs a recent PyTorch nightly ✔ (`README.md:33-35`).

## Vision encoder alone on its own stage
Not supported without changing the model ✔: the encoder runs only inside the embedding step on the stage that holds the token embeddings (`models/qwen3_5/model.py:705-718`); a stage without it passes tokens through (`:612-617`).
