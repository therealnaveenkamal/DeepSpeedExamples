# 27B 50-step runs on one 8× H100 node: plan (2026-10-12, not yet approved)

Equal work in every run: Bridge's exported CORD-v2 rows, same order, 64 rows per step, 1 row per microbatch, every row padded to 2048, token-mean loss over the step, AdamW lr 1e-5 constant, betas 0.9/0.95, eps 1e-8, no weight decay, no clipping, no warm-up, fp32 master weights and fp32 Adam, no recompute unless stated. Timing: median of steps 6-50, no profiler.

## Layouts

**A. Normal layout (vision shares stage 0), TP4 × PP2 × DP1**
- Stage 0 = GPUs 0-3 (TP4): vision encoder + embedding + layers 0-31.
- Stage 1 = GPUs 4-7 (TP4): layers 32-63 + output layer.
- 1F1B. Split 32 | 32 on all three (Bridge even split, rdsp `--cuts balanced` chose it, TorchTitan's default).

| # | System | Settings beyond the shared ones | Fit check |
|---|---|---|---|
| A1 | Megatron-Bridge | bf16 gradients, precision-aware optimizer (fp32 master as remainders), `scripts/meg_extra_bf16.txt` + `GRAD_REDUCE_FP32=false` | passed, 27.0 s/step |
| A2 | rdsp | `--grad-dtype bf16 --fused-adam --optimizer-group-numel 250000000` | passed, 22.7 s/step (timers on) |
| A3 | TorchTitan | its defaults: fp32 gradients, sequence parallel on; recipe `torchtitan/cord_qwen35.py` | not yet: needs a fit check first |

**B. Vision on its own GPU**

| # | System | Layout | Status |
|---|---|---|---|
| B1 | rdsp | GPU 0 vision; GPUs 1-4 TP4 layers 0-38; GPUs 5-6 TP2 layers 39-57; GPU 7 TP1 layers 58-63 + output; recompute on the language stages + groups; token-mean loss | fit passed, 28.4 s/step |
| B2 | MegatronMIMO | GPU 4 vision; GPUs 0-3 TP4 language (at most 4 GPUs); 3 GPUs idle | does not fit: out of memory in step 1 (`mimo_vs_rdsp/27b_v1/logs/fit-mimo.log`) |
| — | Bridge, TorchTitan | no vision-on-its-own-GPU option | not runnable |

**C. No pipeline, TP4 × DP2 (added 2026-10-16: does 27B need a pipeline at all?)**

At 9B the layout without a pipeline was the fastest on both systems (TP2 × DP4: rdsp 5.84 s, Bridge 6.45 s; TP4 × PP2 × DP1: 14.63 s, 18.24 s). C asks whether that still holds at 27B.

- One stage on all 8 GPUs: TP4 × DP2, ZeRO-1 (optimizer state split over the 2 DP copies), 32 microbatches of 2 rows (1 per DP copy). Vision encoder in the stage, eager.
- Why TP4 and not more: 27B has 4 key/value heads, so TP stops at 4 (rdsp refuses tp=8: `compiler.py`, `_check_tp_divides_heads`). Why not TP2 × DP4: 13.7B parameters per GPU, about 70 GiB with ZeRO-2 before any activations, and DeepSpeed refuses ZeRO-3 with TP.
- Same per-GPU work as A: 32 microbatches × 64 layers = 64 × 32 layer passes, all at TP4 with 1 row. So A against C measures pipeline costs against recompute costs.

Memory estimate, rdsp, one GPU, GiB (75.45 usable, measured on these cards). Not measured yet:

| | No recompute | Full recompute |
|---|---|---|
| Held between steps: 6.87B parameters × 8 bytes (bf16 weight, fp32 master and Adam split over DP2) | 51.2 | 51.2 |
| bf16 gradients | 12.8 | 12.8 |
| Activations: 1 microbatch × 64 layers (equals stage 0's measured 2 × 32), plus the encoder | 17.9 | ~4.5 |
| Logits | ~1 | ~1 |
| **Forward/backward peak** | **82.9: does not fit** | **69.5: fits, ~6 spare** |

The optimizer step also needs `--optimizer-group-numel`: with one group it holds a flat bf16 and an fp32 copy of the gradients at once (+~19 GiB, out of memory). Bridge's estimate is similar (about 57.6 GiB held with remainders, ~70 GiB usable), so both systems run C with full recompute.

| # | System | Settings beyond the shared ones | Fit check |
|---|---|---|---|
| C1 | Megatron-Bridge | TP4, PP1, distributed optimizer, bf16 gradients + precision-aware optimizer (as A1), full recompute, `dataset.pad_to_max_length=true` (without a pipeline Bridge does not force padding) | not yet |
| C2 | rdsp | `--stages 1 --stage "0:gpus=8,tp=4,zero=1,recompute=1" --rows 2 --microbatches 32 --pad-multiple 0 --grad-dtype bf16 --fused-adam --optimizer-group-numel 250000000` | not yet |
| C3 | TorchTitan | TP4 × FSDP over 2 (`cord_qwen35.qwen35_27b_cord_tp4dp2`): FSDP splits weights, gradients and Adam over both copies (Bridge and rdsp split only the optimizer state), so it may fit without recompute (estimate ~51 GiB of states + activations with sequence parallel). Rows interleaved so each step trains on Bridge's 64 rows (`cord_data.py`, `data_parallel=2`; CPU check `torchtitan/checks/tt_dp_check.py`). Falls back to full recompute (`_rc`) if the fit check fails | not yet |

Prediction, to check against the result: recompute adds one forward per microbatch (+33% compute; at TP4 × PP2 it cost +49%: 36.6 s against 24.6 s), while A's pipeline costs at 64 microbatches are small (1.5% bubble). So C should take about 29-34 s against A's ~23 s, which would make 27B the first size where the pipeline pays on one node: it avoids recompute. If C is faster anyway, the pipeline is not needed on one NVLink node at 27B either.

## Order and time on the box (approved 2026-10-16: everything)

One unattended queue, `queue_27b_all.sh`: each run has a timeout, its own nvidia-smi samples and an `=== END <run> exit <code>` marker. TorchTitan runs last (A3, then C3): a 3-step fit check, then 50 steps, or full recompute if the fit check fails. Its environment builds during prep.

| Step | Minutes |
|---|---|
| Prep: setup, 27B download, Bridge's exported rows, Bridge checkpoint | ~45 |
| A1 Bridge TP4×PP2 | ~30 |
| A2 rdsp TP4×PP2 | ~23 |
| B1 rdsp vision 1 \| TP4 \| TP2 \| TP1 | ~28 |
| C2 rdsp TP4×DP2 | ~40 |
| C1 Bridge TP4×DP2 | ~45 |
| B2 MIMO: checkpoint conversion + run (out of memory expected) | ~17 |
| A3 TorchTitan TP4×PP2: fit check + 50 steps | ~42 |
| C3 TorchTitan TP4×FSDP2: fit check + 50 steps | ~42 |
| TorchTitan recompute fallbacks, if needed | 0-80 |
| Pull logs | ~5 |
| **Total** | **~5.3-6.6 h** |

## Added 2026-10-16: PP against DP with recompute everywhere (`queue_27b_more.sh`, starts after `=== ALL DONE`)

A against C mixes two effects: the pipeline lets 27B skip recompute (A fits without it, C does not), and PP against DP traffic and bubble. These runs recompute everywhere, so layouts differ only in how the 8 GPUs are split. Bridge and rdsp each; settings as A1/A2 plus full recompute.

| # | Layout | Folder | Pairs with |
|---|---|---|---|
| A1-rc, A2-rc | TP4 × PP2 + recompute | `27b_tp4pp2dp1` (logs `*-rc-*`) | C1, C2 (TP4 × DP2 + recompute): PP2 against DP2 at TP4 |
| D1, D2 | TP2 × PP2 × DP2 + recompute | `27b_tp2pp2dp2` | E (more PP, no DP) at TP2 |
| E1, E2 | TP2 × PP4 + recompute | `27b_tp2pp4dp1` | D |

Not runnable on 8 GPUs at 27B (estimates, `≈75 GiB usable`): TP8 (4 KV heads), TP2 × DP4, PP4 × DP2, PP2 × DP4, DP8 without full sharding; PP8 only maybe with recompute and an uneven cut.

Time ~3.5 h. **Whole session: ~8.8-10.1 h.**

## Round 3 (approved 2026-10-17: "everything") — `queue_27b_round3.sh`

| Part | Runs | Minutes |
|---|---|---|
| Prep | environment, weights, Bridge's exported rows, Bridge checkpoint; TorchTitan builds alongside | ~45 |
| No-recompute fit checks (3 steps; 50 if it fits) | C1/C2, D1/D2, E1/E2 (Bridge/rdsp) | ~36-60 |
| rdsp D (TP2 × PP2 × DP2 + recompute) | ZeRO-2, 50 steps (frees ~6.5 GiB: gradients split over DP2; CPU check `27b_tp2pp2dp2/results/zero2_check.txt`) | ~20 |
| rdsp D profile | 10 steps with rdsp's timers; 12 under Nsight Systems (40 s window from step 3), sqlite export | ~17 |
| Bridge D partial recompute | block method, recompute 24 of 32 layers per stage (28 if 24 does not fit), fit then 50 | ~28-36 |
| TorchTitan | no-recompute fit checks D3, E3; then D3, E3, C3 with recompute (fit, 50) | ~160 |
| Pull | | ~5 |
| **Total** | | **~5.2-5.7 h** |

Measured headroom with recompute (last round, `nvidia-smi`) says none of the no-recompute checks should fit; they run to measure it.
