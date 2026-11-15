# Blog round results, 2026-11-11 (one p5.48xlarge, us-east-2b)

All numbers: median seconds per step over steps 6-50 of a 50-step run unless marked "trial".
Tokens/s per GPU = 64 x 2,048 tokens a step / step time / GPUs. Logs carry `-nov11`.

## 27B, eight stages of 1 GPU (8 GPUs), every layer recomputed

| System | Step | Tokens/s/GPU | Log |
|---|---|---|---|
| rdsp, measured cut 1\|9\|10\|9\|9\|10\|9\|7 | 10.91 s | 1,502 | `bridge_vs_rdsp/27b_tp1pp8dp1/logs/rdsp-nov11.log` |
| Megatron-Bridge at that cut | 14.45 s | 1,134 | `.../bridge-nov11.log` |

1.32x (blog: 1.28x from two machines). Bridge's settings equal October's except the cut.
Bridge py-spy (60 s, `profiles/pyspy-bridge-nov11-*.txt`, summary `results/pyspy_bridge-nov11.txt`):
stage 0 (vision encoder + 1 layer) forward 37.6% (vision 31.1% = 4.5 s), backward 62.1% (9.0 s),
waiting 0.1%; stages 1-7 spend 53-65% at `get_batch_from_iterator (qwen3_vl_step.py:90)`, waiting
for the stage before. rdsp's stage 0 at one layer is about 9.2 s (from calibration at six layers:
forward 4.4 s, backward 9.0 s, 0.82 s a layer).

## 9B MIMO comparison (5 GPUs: 1 vision GPU + 4 language GPUs)

rdsp runs with `RDSP_DIRECT_GRAD_ADD=1` (one GPU kernel per parameter per microbatch for the fp32
gradient accumulation instead of DeepSpeed's two; same bits). Rows per microbatch changed on both
systems alike (MIMO: `--micro-batch-size`; rdsp: `--rows`, `--microbatches`).

### B: language TP2 x DP2 (`mimo_vs_rdsp/9b_v1_tp2pp1dp2/logs/`)

| Setting | MIMO | rdsp | rdsp / MIMO |
|---|---|---|---|
| 2 rows per microbatch (shipped) | 8.41 (1 run) | 8.39-8.45 without the fix; 7.73 with (50 steps), 7.78, 7.78 (26 steps) | 1.00x; 1.08x with the fix |
| 4 rows per microbatch | 6.08, 6.07, 5.74 | 5.56, 5.58, 5.54 | 1.09x (medians 6.07 / 5.56) |
| 4 rows, rdsp language stage compiled, no Liger | | 5.40 (1 run) | 1.12x |
| 8 rows per microbatch | out of memory | out of memory | |

### A: language TP2 x PP2 (`mimo_vs_rdsp/9b_v1_tp2pp2dp1/logs/`)

| Setting | MIMO | rdsp | rdsp / MIMO |
|---|---|---|---|
| 1 row per microbatch (shipped) | 9.32 (1 run) | about 8.6 (26-step trial, with the fix) | 1.08x |
| 2 rows per microbatch | 6.82 (1 run) | 5.06, 5.06, 5.01 | 1.35x |
| 4 rows per microbatch | 5.76, 6.22, 6.09 | out of memory; 5.67 with recompute_blocks=6 on stage 1 | |
| each at its best | 6.09 (4 rows) | 5.06 (2 rows) | 1.20x |

### What the profiles say

- Shipped microbatches: rdsp's language stage paces (busy about 93% of the step); the October
  traces put its extra backward work in the gradient-accumulation hook (two kernels per parameter
  per microbatch) and about 47,500 small elementwise launches a step, most inside the gated-delta
  layers (origin inferred).
- Larger microbatches: both systems about 30% faster (both launch-bound). rdsp's language stages
  then wait 0.5-0.9 s a step for the vision stage's next forward (`--profile`, logs `*-probe.log`).
- MIMO's step time follows the step's content: the same steps are slow in all three runs
  (correlation 0.68 between runs; 4.5 to 8.8 s), where rdsp's are flat (5.4-5.7 s). MIMO's
  fastest steps (4.5 s) are faster than any of rdsp's.
- torch.compile on the language stage works only without Liger's kernels.

### Open

- Loss drift in two shipped-setting runs (one baseline, one with the fix) and 3-5 slow steps
  (10-15 s) in most rdsp runs after 19:43 UTC: cause unknown.
- 27B MIMO row: needs two nodes; no spot capacity for a pair on 2026-11-11.
- Next rdsp levers: let stage 0 run two microbatches ahead; the small-operation overhead in the
  language backward; Liger + compile.

## 27B MIMO comparison (9 GPUs on two p5.48xlarge, us-west-2a, 2026-11-12 01:04-06:06 UTC)

1 vision GPU + language TP4 x PP2; 128 x 2,048 a step. MIMO: its launcher with our memory settings
(`gap_round/mimo/patch_mimo.py`). rdsp: `gap_round/scripts/rdsp9.sh`'s settings. Logs in
`blog_round/logs27b/` (MIMO's step lines in `*-node2.log`). 10-step runs: median of steps 4-10.

| Setting | MIMO | rdsp | rdsp / MIMO |
|---|---|---|---|
| 1 row per microbatch (shipped), 10 steps | 30.82 s (peak 68.6 GB) | 31.23 s with --profile (peak 78.2 GB) | 0.99x |
| 2 rows per microbatch, 10 steps | 17.27 s (peak 79.7 GB) | out of memory | |
| 2 rows, rdsp recompute_blocks=12, measured cut 32\|32, 10 steps | | 20.40 s | |
| 2 rows, rdsp recompute_blocks=8, `--cuts 0,32`, 10 steps | | 19.78 s (peak 79.2 GB) | |
| 4 rows, MIMO with every language layer recomputed, 10 steps | 17.19 s (peak 68.6 GB) | | |
| **2 rows, 50 steps (median of steps 6-50)** | **16.73 s, 1,742 tokens/s/GPU** | **19.62 s, 1,484 (recompute_blocks=8, `--cuts 0,32`)** | **0.85x** |

Did not work for rdsp at 2 rows: recompute_blocks=16 with the measured cut and compile + recompute_blocks=12
(both out of memory in `StageWorkerActor.bootstrap()` when the engine rebuilds after `--cuts 0,measured`);
compile without recompute (out of memory); language as TP2 x PP4 (out of memory at 1 and 2 rows);
3 decoder layers on the vision GPU (`--cuts 3,33`, out of memory).

rdsp `--profile` at 1 row (cut 31\|33): vision forward 4.1 s, backward 6.0 s, waiting 20 s; language
stage 1 forward 11.2 s, backward 16.5 s, waiting 3 s; stage 2 forward 11.2 s, backward 17.4 s,
waiting 1.4 s. The language stages pace the step.

What it means: MIMO gains 46% from 2 rows per microbatch and still fits; rdsp's saved activations
are larger (78 GB against 69 GB at 1 row), so 2 rows need recompute, and even without recompute its
language stages would take about 18.5 s (12 to 8 recomputed layers saved 0.6 s). Leads for rdsp:
one all-reduce per shared input in the tensor-parallel backward (rdsp 153 a microbatch against
MIMO's 65, counted on 9B), fewer small operations, the engine rebuild's memory after a measured cut.

## MIMO table, both ways (2026-11-11/12)

| Row | At MIMO's shipped microbatch | Each system at its best microbatch |
|---|---|---|
| 9B, TP2 x PP2 | 1.08x (9.32 / 8.62 s; rdsp from a 26-step run) | 1.20x (6.09 / 5.06 s) |
| 9B, TP2 x DP2 | 1.09x (8.41 / 7.73 s, the 50-step run; the two 26-step repeats 7.77 s, 1.08x) | 1.09x (6.07 / 5.56 s) |
| 27B, TP4 x PP2 | 0.99x (30.82 / 31.23 s, 10 steps) | 0.85x (16.73 / 19.62 s) |

Decided 2026-11-12: the blog uses the left column (MIMO's shipped microbatch). Table as tokens/s/GPU,
DeepSpeed / MIMO: 3,039 / 2,814; 3,391 / 3,116; 933 / 945. Paste-ready text and the three things to
settle first (the gradient switch is off by default; the TP2 x PP2 rdsp run is 26 steps; the 27B rdsp
run had no gradient switch and three slow steps of 42-44 s that the median hides, 0.98x over the same
six steps on both systems) are in `docs/blog-comments-2026-11-11.md`.
Since October: MIMO's TP2 x DP2 run measured 6% lower on this machine (3,331 to 3,116 tokens/s/GPU);
rdsp without the switch is unchanged (3,124 to 3,125).

2026-11-12: the direct gradient add is now on by default (`RDSP_DIRECT_GRAD_ADD=0` turns it off). It
only engages with FP32 gradient accumulation, so the 27B and 35B settings (BF16 accumulation) are not
touched. Mac: unit tests 455 passed, 2 skipped (need DeepSpeed) with it on and with it off; ruff clean on the two changed files (8 older
findings in six files from October, seven of them long lines, left alone).
Not run: the integration tests (Ray refuses to upload this folder, now 774 MB against its 512 MB limit,
and the DeepSpeed ones need Linux), and no GPU run with the new default yet.

## Run-ahead trial (`RDSP_FIRST_STAGE_LEAD`, schedule.py), pair node 1, 2026-11-12 05:04-05:53 UTC

Stage 0 runs N more warm-up forwards than 1F1B (off by default). 50 steps each, with the gradient fix.

| Row | Lead 0 | Lead 1 | Lead 2 | MIMO, same machine |
|---|---|---|---|---|
| B (TP2 x DP2), 4 rows | out of memory | 5.57 s | 5.66 s | out of memory at 4 rows |
| A (TP2 x PP2), 2 rows | 5.05 s | | 5.38 s | 5.76 s at 4 rows |

No gain (B: 5.56 s without it on the first node; A: 6% slower). The language stages' wait_fwd is not a
one-microbatch-late vision stage. On this machine both systems ran out of memory at 4 rows on row B
(first node: 80.2 and 80.9 GB peaks), so that setting is at the edge for both. A row on this
machine: 5.76 / 5.05 = 1.14x.
Tests on pair node 2 (CPU): tests/integration test_two_stage_baseline, test_ray_local_pipeline,
test_vl_pipeline give 23 failed / 36 passed with lead 0 and with lead 2, the same 23, and the
three-stage reference test's step-1 loss is identical in both (2.770256280899048; the recorded
reference is 2.7702560424804688, so those failures are that node against the recorded values).
Logs: `blog_round/logs/lead_tests.out`, `nolead_tests.out`.
