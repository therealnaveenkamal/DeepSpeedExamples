# AutoTP module injection vs. DeepCompile `autotp` pass — dense Qwen3.5, 4×A100

Benchmark of the DeepCompile AutoTP pass ([PR #8204](https://github.com/deepspeedai/DeepSpeed/pull/8204),
issue [#8104](https://github.com/deepspeedai/DeepSpeed/issues/8104)) against stock AutoTP module
injection on a hybrid-attention model, at two DP/TP splits, plus a compatibility study of the
pass's `fullgraph=True` requirement against the flash-linear-attention / causal-conv1d fast
kernels.

## Summary of findings

- **Finding 1 (confidence: high):** On dense Qwen3.5 (618M, 6 GatedDeltaNet + 2 full-attention
  layers, bf16), the DeepCompile `autotp` pass is **1.31× faster than eager module injection at
  DP2/TP2 and 1.45× at DP1/TP4** (steady-state tokens/s, single 500-step run per cell), at the
  cost of **1.15–1.19× peak memory**. The speedup growing with TP size is consistent with the
  compiler winning most where collectives and replicated fallback compute dominate the step.
- **Finding 2 (confidence: high):** Loss curves for the two modes agree to the bf16 noise floor
  (max |Δloss| 2.35e-2 / 2.68e-2 over 500 steps, ≤0.4% relative after step 400), and step-0
  losses match all four runs at 11.2358 from a bit-identical shared init. This confirms the pass
  computes the same training math *at bf16 resolution*; the sharper fp32 equivalence (~1e-6) was
  previously established on Llama only (see Threats).
- **Finding 3 (confidence: high, direct error observation):** The pass's mandatory
  `fullgraph=True` capture is **incompatible with both fast-kernel libraries as shipped**:
  causal-conv1d 1.6.2 fails Dynamo tracing ("non-contiguous `out=` tensor" inside its
  `autograd.Function`), and fla 0.5.2 wraps `chunk_gated_delta_rule` and `FusedRMSNormGated` in
  `torch.compiler.disable`, which fullgraph turns into a hard error. The pass fails loudly at
  capture time — no silent fallback — which is the designed behavior.
- **Finding 4 (confidence: low — partial data):** With fla's fast recurrence in the *eager* arm
  only (the compile arm cannot use it), eager throughput at DP2/TP2 rose from 1,557 to ~1,640
  tok/s (+5%, measured over 28 steps before the run was intentionally stopped). This suggests the
  compile pass would remain ahead (~1.25×) even against fla-accelerated eager, but the run was not
  completed; treat as preliminary.
- **Bottom line:** the compile pass delivers a reproducible ~1.3–1.45× training speedup over
  module injection on this model with identical-at-bf16 training behavior, but it currently
  forces the traceable torch fallback path for linear-attention kernels; its net advantage over a
  fla-accelerated eager baseline is indicated but not yet fully measured.

## Question and approach

Does moving AutoTP's tensor-parallel collectives from module-level (eager `LinearLayer` /
`LinearAllreduce`) into the compiled FX graph (a) preserve training behavior and (b) improve
throughput, on a model that stresses the new code paths — gathered column-parallel layers
(`colwise_gather_output` on all five GatedDeltaNet projections and the LM head) and replicated
parameters with gradient all-reduce (`q_norm`/`k_norm`)?

Design: 2×2 grid — {DP2/TP2, DP1/TP4} × {`autotp` (module inject), `autotp_compile` (DeepCompile
pass)} — 500 optimizer steps each. All four runs share one random init (`init.pt`, seed 42) and a
deterministic batch schedule indexed by `(step, accum_index, dp_rank)`, so TP peers receive
identical inputs and every setup consumes the same global batch sequence. The only variable
between paired arms is where the collectives execute.

## Setup

| | |
| --- | --- |
| Hardware | 4× NVIDIA A100-PCIE-40GB |
| Software | torch 2.11.0+cu128, transformers 5.14.1, DeepSpeed 0.19.3+e7762c3b (branch `feature/autotp-dev`, commit `4738778e`) — pinned in `environment/requirements.lock` |
| Model | dense Qwen3.5 (`Qwen3_5ForCausalLM`), 8 layers (layers 3 and 7 full attention, rest GatedDeltaNet), hidden 2048, intermediate 5632, 16 heads / 8 KV, head_dim 128, vocab padded to 50304, 617.6M params |
| TP plan | HF `base_model_tp_plan`: colwise/rowwise attention+MLP, `colwise_gather_output` on all linear-attention projections and `lm_head`, `replicated_with_grad_allreduce` on `q_norm`/`k_norm` |
| Data | wikitext-2-raw-v1 train, gpt2 tokenizer, seq_len 1024 |
| Schedule | micro-batch 2, global batch 8 sequences (8,192 tokens/step), AdamW lr 1e-4, bf16, ZeRO-0, seed 42, 500 steps, steps ≥50 in steady-state averages |
| Kernels | torch fallback for GatedDeltaNet in all four main runs (fla/causal-conv1d not installed); see the fla section for the variant |

Runs: 4 main cells (~4 h wall total) + 1 partial fla cell + 3 short smoke cells. One seed per
cell — variance not measured here; see Threats.

## Results

### Throughput / memory / loss (500 steps, bf16)

| setup | mode | steady tok/s | step ms | peak GiB | first step (s) | final loss |
| --- | --- | --- | --- | --- | --- | --- |
| DP2/TP2 | module inject | 1,556.6 | 5,264 | 10.97 | 7.1 | 4.5981 |
| DP2/TP2 | DeepCompile pass | 2,041.7 | 4,013 | 12.60 | 347.4 | 4.5968 |
| DP1/TP4 | module inject | 751.9 | 10,895 | 8.83 | 13.5 | 4.5872 |
| DP1/TP4 | DeepCompile pass | 1,089.1 | 7,531 | 10.52 | 340.4 | 4.5991 |

Ratios (compile / inject): **1.31× / 1.45× throughput**, 1.15× / 1.19× peak memory, max |Δloss|
2.35e-2 / 2.68e-2. Figures: `plots/throughput.png`, `plots/step_latency.png`,
`plots/peak_memory.png`, `plots/loss_curve.png` (loss curves with per-setup |Δloss| panel; blue =
module inject, orange = DeepCompile pass; solid = DP2/TP2, dashed = DP1/TP4).

Sanity checks that anchor these numbers:

- Step-0 loss is 11.2358 in all four runs (and matches an independent earlier run of the same
  config on a previous environment), confirming bit-identical init and data.
- DP2/TP2 and DP1/TP4 loss curves coincide at matched steps (same global batch by construction).
- The compile arms' ~340 s first step is one-time Dynamo/Inductor compilation with a cold cache;
  it is excluded from steady-state numbers.
- An earlier run of the DP2/TP2 pair on the parent commit gave 1,377 / 1,868 tok/s (1.36×): the
  `4738778e` cleanup sped up *both* arms (~13% / ~9%) without changing the loss agreement —
  behavior-preserving, as intended.

### Why the speedup grows with TP size (interpretation)

Per-GPU *sharded* matmul FLOPs are identical at TP2 and TP4 in this grid (half weights × half
batch vs. quarter weights × full batch). What doubles per GPU at TP4 is (a) the replicated
GatedDeltaNet fallback compute — the HF plan gathers every linear-attention projection output, so
the recurrence runs on the full width on every rank, over 4 seqs/GPU at DP2/TP2 but 8 seqs/GPU at
DP1/TP4 — and (b) collective launches (grad-accum doubles from 2 to 4 microbatches, each forward
crossing ~30 gathers), with 4-rank rings also moving 1.5× the bytes per element. These are
exactly the components the compiler attacks (kernel fusion of the memory-bound fallback;
collectives as schedulable graph nodes), so the addressable pool is larger at TP4: 1.31× → 1.45×.
This interpretation is consistent with the eager slowdown pattern (1,557 → 752 tok/s despite
constant sharded FLOPs) but is not yet confirmed by profiling — the planned torch.profiler
three-arm comparison will attribute the win between Inductor fusion and collective placement.

### fla / causal-conv1d compatibility (fullgraph study)

With `flash-linear-attention==0.5.2` and `causal-conv1d==1.6.2.post1` installed, the
`autotp_compile` arm fails at graph capture (the pass sets `fullgraph=True` by design, so any
graph break is a hard, loud error):

| library | failure | nature |
| --- | --- | --- |
| causal-conv1d | `torch._dynamo.exc.Unsupported: non-contiguous out= tensor` in `CausalConv1dFn` (`DaoAILab._causal_conv1d_fwd_cpp`) | Dynamo tracing gap; potentially fixable upstream |
| fla | `chunk_gated_delta_rule` and `FusedRMSNormGated`'s `layer_norm_gated_fwd` wrapped in `torch.compiler.disable` | deliberate opt-out by fla; fullgraph converts the intended graph break into an error |

Eager module injection has no such constraint (it never traces), so the realistic head-to-head is
eager+fla vs. compile+torch-fallback. A 100-step run of that comparison was started (with a
sweep-script shim routing compiled runs to the traceable fallbacks) and intentionally stopped at
step 28 of the first arm: eager+fla reached ~1,640 tok/s at DP2/TP2 (+5% over fallback eager;
`data/fla_run_partial/`). A toy-scale smoke test (4 layers, seq 256) had compile+fallback 1.08×
ahead of eager+fla. Both point the same direction — the pass's win survives fla-accelerated
eager — but neither is a completed, full-size measurement.

## Threats to validity

- **Single run per cell, no seeds.** Throughput on this box varied a few percent between
  identical-config runs on different days (e.g. eager DP2/TP2: 1,377 pre-cleanup vs 1,557
  post-cleanup includes both the commit and box state). The 1.31×/1.45× ratios come from arms
  interleaved on the same box within hours, which cancels most drift, but ±few-% on any single
  number should be assumed. The loss-agreement finding is unaffected (deterministic data/init).
- **bf16 cannot prove collective placement.** A misplaced collective producing ~1e-2 errors would
  hide inside the observed bf16 noise (the step-0 |Δ| of ~2e-4 with zero optimizer state is
  reassuring but not proof). The fp32 equivalence run (~1e-6 floor) that anchors the Llama
  verification has **not** been run for Qwen3.5, which exercises new paths (gather-output layers,
  replicated-grad hooks, hybrid layers). This is the highest-value missing check.
- **Speedup attribution is bundled.** 1.31–1.45× compares eager module injection against
  compiled-with-pass; it includes Inductor's generic kernel fusion, which is large here because
  the GatedDeltaNet torch fallback is fusion-friendly. The `autotp_torchcompile` attribution arm
  (run for Llama in the PR's verification, where the pass contributed beyond plain compile) has
  not been run for Qwen3.5 — pending profiling work.
- **Memory cost is real but unquantified in mechanism:** compile arms hold 1.15–1.19× peak
  memory; not investigated further.
- **Absolute throughput is not representative** (MFU is very low): the main runs use the slow
  torch fallback for 6 of 8 layers' attention. Ratios, not absolute tok/s, are the meaningful
  output.

## Reproducibility

- Harness: `compile_pass_verification/qwen35_dense_sweep.py` (this repo snapshot); DeepSpeed
  branch `feature/autotp-dev` @ `4738778e` (`environment/deepspeed_commit.txt`), packages in
  `environment/requirements.lock`; environment rebuild via `setup_env.sh` (torch cu128 index pin).
- Main grid: `python qwen35_dense_sweep.py --out_dir runs/qwen35_dense_500 --steps 500`
  (add `--setups dp2_tp2` / `dp1_tp4` to run cells separately; plots regenerate with
  `--plot_only`).
- Determinism: shared `init.pt` (seed 42) + deterministic batch indexing make eager arms
  bit-reproducible; compiled arms vary at the ~1e-6 (fp32) / ~1e-2 (bf16) level across
  recompilations because Inductor does not pin reduction order.
- Raw artifacts: `data/<setup>/<mode>/{summary.json,metrics.csv}`, combined ratios in
  `data/combined_summary.json`.

## What's next (planned, not yet run)

1. **torch.profiler three-arm comparison** (`autotp` / `autotp_torchcompile` / `autotp_compile`)
   on this model — attributes the speedup between Inductor fusion and in-graph collectives, and
   measures collective time share directly.
2. **FX graph dump + cross-check** (`graph_dump.py`) on commit `4738778e` — verifies every
   column/row-parallel module got its `copy_to_tp_region` / `reduce_from_tp_region` /
   `gather_from_tp_region` node, and documents what the captured graph contains in the fla
   configuration.
3. **fp32 equivalence run** (~40 steps, DP2/TP2) for Qwen3.5 — closes the strongest remaining
   correctness gap.
4. Completing the 100-step eager+fla vs. compile+fallback comparison at both splits.
