# AutoTP module injection vs. DeepCompile `autotp` pass — dense Qwen3.5, 4×A100

Benchmark of the DeepCompile AutoTP pass ([PR #8204](https://github.com/deepspeedai/DeepSpeed/pull/8204),
issue [#8104](https://github.com/deepspeedai/DeepSpeed/issues/8104)) against stock AutoTP module
injection on a hybrid-attention model, at two DP/TP splits, plus a compatibility study of the
pass's `fullgraph=True` requirement against the flash-linear-attention / causal-conv1d fast
kernels.

## Summary of findings

- **Finding 1 :** On dense Qwen3.5 (618M, 6 GatedDeltaNet + 2 full-attention
layers, bf16), the DeepCompile `autotp` pass is **1.31× faster than eager module injection at
DP2/TP2 and 1.45× at DP1/TP4** (steady-state tokens/s, single 500-step run per cell), at the
cost of **1.15–1.19× peak memory**. The speedup growing with TP size is consistent with the
compiler winning most where collectives and replicated fallback compute dominate the step.
- **Finding 2 :** Loss curves for the two modes agree to the bf16 noise floor  
(max |Δloss| 2.35e-2 / 2.68e-2 over 500 steps, ≤0.4% relative after step 400), and step-0  
losses match all four runs at 11.2358 from a bit-identical shared init. This confirms the pass  
computes the same training math *at bf16 resolution*; the sharper fp32 equivalence (~1e-6) was  
previously established on Llama only (see Threats).
- **Bottom line:** the compile pass delivers a reproducible ~1.3–1.45× training speedup over module injection on this model with identical-at-bf16 training behavior.



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


|          |                                                                                                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hardware | 4× NVIDIA A100-PCIE-40GB                                                                                                                                                                                |
| Software | torch 2.11.0+cu128, transformers 5.14.1, DeepSpeed 0.19.3+e7762c3b (branch `feature/autotp-dev`, commit `4738778e`) — pinned in `environment/requirements.lock`                                         |
| Model    | dense Qwen3.5 (`Qwen3_5ForCausalLM`), 8 layers (layers 3 and 7 full attention, rest GatedDeltaNet), hidden 2048, intermediate 5632, 16 heads / 8 KV, head_dim 128, vocab padded to 50304, 617.6M params |
| TP plan  | HF `base_model_tp_plan`: colwise/rowwise attention+MLP, `colwise_gather_output` on all linear-attention projections and `lm_head`, `replicated_with_grad_allreduce` on `q_norm`/`k_norm`                |
| Data     | wikitext-2-raw-v1 train, gpt2 tokenizer, seq_len 1024                                                                                                                                                   |
| Schedule | micro-batch 2, global batch 8 sequences (8,192 tokens/step), AdamW lr 1e-4, bf16, ZeRO-0, seed 42, 500 steps, steps ≥50 in steady-state averages                                                        |
| Kernels  | torch fallback for GatedDeltaNet in all four main runs (fla/causal-conv1d not installed); see the fla section for the variant                                                                           |


Runs: 4 main cells (~4 h wall total) + 1 partial fla cell + 3 short smoke cells. One seed per
cell — variance not measured here; see Threats.

## Results



### Throughput / memory / loss (500 steps, bf16)


| setup   | mode             | steady tok/s | step ms | peak GiB | first step (s) | final loss |
| ------- | ---------------- | ------------ | ------- | -------- | -------------- | ---------- |
| DP2/TP2 | module inject    | 1,556.6      | 5,264   | 10.97    | 7.1            | 4.5981     |
| DP2/TP2 | DeepCompile pass | 2,041.7      | 4,013   | 12.60    | 347.4          | 4.5968     |
| DP1/TP4 | module inject    | 751.9        | 10,895  | 8.83     | 13.5           | 4.5872     |
| DP1/TP4 | DeepCompile pass | 1,089.1      | 7,531   | 10.52    | 340.4          | 4.5991     |


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


| library       | failure                                                                                                             | nature                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| causal-conv1d | `torch._dynamo.exc.Unsupported: non-contiguous out= tensor` in `CausalConv1dFn` (`DaoAILab._causal_conv1d_fwd_cpp`) | Dynamo tracing gap; potentially fixable upstream                                     |
| fla           | `chunk_gated_delta_rule` and `FusedRMSNormGated`'s `layer_norm_gated_fwd` wrapped in `torch.compiler.disable`       | deliberate opt-out by fla; fullgraph converts the intended graph break into an error |




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

