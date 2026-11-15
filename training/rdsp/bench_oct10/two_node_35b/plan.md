# Two nodes, 16x H100: do the single-node conclusions hold for a bigger model?

Model: Qwen3.5-35B-A3B (decided 2026-10-18), replacing two_node_27b/plan.md. Single-node map: "Where
rdsp Wins", https://claude.ai/artifact/GDZSB15b3iMwMHdK84Yq7A; rdsp_fixes_27b/results/results.md.

## The model, and what is new about it

- 36.0B parameters with its next-token-prediction (MTP) layer, ~35B trained: 40 layers (30 Gated
  DeltaNet, 10 full attention), every layer a mixture of experts (MoE): 256 experts, 8 used per
  token, plus a shared expert. Same 27-layer vision encoder as the 27B.
- Does not fit one node: 14 bytes per parameter of training state is 61 GiB per GPU on 8 GPUs
  before activations (the 27B at 44 GiB already peaked at 57-73 GiB). On 16 GPUs: 31 GiB.
- About 3B parameters active per token, ~1/7 of the 27B's language math, with the same vision
  encoder. Estimate from the 27B's ~130 ms per image: vision is a much larger share of each
  step than on the 27B, so where the vision encoder sits may matter more than layer cuts.
  Measured in run 1 (stage busy times), not assumed.

## Settings (as the 27B runs, plus the MoE ones)

- 128 rows of 2,048 per step (Bridge's rows, exported again at 128 per step), one row per GPU
  per microbatch, every row padded to 2,048, token-mean loss, AdamW lr 1e-5 constant, betas
  0.9/0.95, eps 1e-8, no weight decay, no clipping, no warm-up, 50 steps, median of steps 6-50.
- bf16 weights and gradients, fp32 master weights stored as 2-byte remainders, fp32 Adam
  moments (Bridge: precision-aware optimizer, meg_extra_bf16.txt; rdsp: --master-dtype
  remainder). TorchTitan as on one node (its fp32 FSDP master).
- MoE, the same on every system: no token dropping; MTP layer off (Bridge
  model.mtp_num_layers=null; TorchTitan and rdsp never build it); router load-balancing loss
  OFF everywhere. Each system defines it differently (Bridge global_aux_loss, TorchTitan
  batch-wise, Hugging Face per batch, off by default), so only "off" is equal. Over 50 steps
  at lr 1e-5 from the trained model, routing barely moves.
- Network: EFA, 32 cards per node. Software: NeMo 26.08 container (Megatron-Bridge v0.6.0),
  TorchTitan 3c36a77, DeepSpeed 53a2ac4, transformers 5.17.0.

## Runs

| # | Checks | System and layout |
|---|---|---|
| 1 | Best of each (27B: rdsp 725 > TorchTitan 571 > Bridge 492 real tok/s per GPU) | rdsp TP1 x PP8 x DP2, measured cut, no expert parallelism |
| 2 | same | Bridge TP1 x PP8 x DP2: even split, and the nearest Megatron can express to rdsp's measured cut |
| 3 | same | TorchTitan FSDP over 16 |
| 4 | Bridge's ground (27B: Bridge 8% faster at TP4 x DP2) | Bridge's own 2-node recipe for this model: TP2 x EP16, no pipeline (qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config) |
| 5 | same | rdsp on the same layout: one 16-GPU stage, TP2, DeepSpeed AutoEP 16 |
| 6 | TorchTitan's ground (27B: TorchTitan 27% faster than rdsp ZeRO-3) | TorchTitan FSDP16 + EP8 (experts split inside each node) |
| 7 | same | rdsp ZeRO-3 over 16 |
| 8 | Vision on its own GPUs (MIMO's idea) | rdsp: vision encoder on its own stage of k GPUs, language PP x DP2 on the rest. k from run 1's measured times: the k in {2, 4, 8} that minimizes max(vision time / k, language time / (16 - k)) |
| 9 | Prediction: deeper pipeline, bigger rdsp lead | rdsp TP1 x PP16 |

MIMO is out: in NeMo 26.08 its Qwen3.5 example covers dense models only (README line 24), its
checkpoint conversion fails on this MoE model (fixed after the 26.08 cut, in 26.09-beta /
r0.7.0, commit 63761af5), and its launcher has no expert-parallel setting. Running it would mean
another container version.

## Choices made from measurements, not by hand

- rdsp's expert kernel: Liger's fused kernel or Hugging Face's grouped matmuls, whichever is
  faster over 6 steps of the PP8 x DP2 layout (queue35.sh, before run 1). Bridge uses its own
  grouped GEMM, TorchTitan its grouped_mm.
- Bridge's cut in run 2: even (5 per stage), and the cut Megatron can express nearest rdsp's
  measured one (scripts/layout_from_log.py megatron-cut).
- K, the vision stage's GPUs in run 8: from run 1's measured stage times
  (scripts/layout_from_log.py vision-width).
- Recompute: every run starts without; one that stops before step 3 (out of memory) runs
  again with every block recomputed. Same rule for every system.

## Phase 0: code, no GPUs (done 2026-10-18)

- rdsp: stages spanning nodes; the 2-byte master with expert parallelism (DeepSpeed's MoE
  groups); AutoEP's Qwen3.5-MoE preset name; --cuts 0,measured (the vision stage keeps its
  cut, the decoder stages are re-cut); --moe-experts; the driver's settings passed to every
  Ray worker (a cluster started by `ray start` runs workers in its own environment: without
  this, --profile and HF_HOME never reached them, checked on a local `ray start` cluster).
- Tests: tiny Qwen3.5-MoE vision model through the pipeline matches the unsplit model (two
  stages, data-parallel experts, vision stage measured and re-cut); the 2-byte master with
  experts split over 2 ranks, and with TP2 x experts over 4 (folded), is bit-identical to
  DeepSpeed's fp32 master after step 1 (CPU, gloo, pinned DeepSpeed and 0.19.6); the queue's
  decisions on fake logs. Unit 457 pass; integration 100 pass, 34 skipped (need GPUs).
- Every rdsp layout's arguments through train_vl's checks and rdsp's planner on a meta-device
  Qwen3.5-35B-A3B (35.1B parameters): all 128 rows a step, the stage widths as planned.
- Two-node scripts: bench/two_node/{efa_env.sh, ray_cluster.sh, both.sh, nccl_check.*};
  ops/h100/{push2, remote2, pull2}.sh, guard.sh with GUARD_ENV=env2.sh; scripts/{common,
  node_prep, prep}.sh, layout_from_log.py; <layout>/run.sh; queue35.sh. TorchTitan 35B configs
  (router loss off) and torchtitan/scripts/run2.sh.

Not testable without GPUs: AutoEP over two nodes (all-to-all over EFA), Bridge's MoE
checkpoint conversion, TorchTitan loading the MoE checkpoint, every system's speed on MoE.

## Phase 1-2: GPUs

| Phase | What | Time |
|---|---|---|
| 1 | launch2.sh; prep.sh: both nodes' environments, 72 GB download, conversion, export, NCCL check over EFA, Ray cluster | ~1.25 h |
| 2 | Expert-kernel check, then the 9 runs (Bridge's first step compiles for 10-18 min), recompute reruns where needed | ~5.5-7 h |

Total: ~7-8 h on 2x p5.48xlarge (us-west-2c). A pair in one placement group may need on-demand
or a capacity block.
