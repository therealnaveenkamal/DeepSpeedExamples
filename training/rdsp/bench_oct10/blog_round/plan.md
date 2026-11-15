# Blog round: diagnosis session (2026-11-11)

Approved 2026-11-11. The full reasoning is in
`rdsp/ncompass/profiling-specs/2026-11-11-blog-mimo-and-bridge-27b-gaps.md`.

Why: two review comments on the PyTorch blog need GPU evidence.
- 27B table: is Megatron-Bridge's first stage (vision encoder + 1 layer) its slowest, and how
  long does its encoder take? Today that is inferred from a 9B profile.
- MIMO table: on 9B with language TP2 x DP2 our language GPUs' backward pass is heavier than
  MIMO's (4.5 s against 3.4 s a step, 2026-10-11). Check it still holds on today's code.

Not in this session: code fixes, the final MIMO reruns (three rows, repeated, two nodes), the
35B expert-split run.

## Runs (`queue_profile.sh`, one p5.48xlarge)

| # | Run | Steps | Minutes |
|---|---|---|---|
| E2 | rdsp 27B, eight stages, measured cut | 50 | 12 |
| E2 | Megatron-Bridge 27B at the cut rdsp measured (October's cut if it does not fit) | 50 | 18 |
| E1 | Megatron-Bridge 27B with its own per-stage timers | 15 | 8 |
| E1 | Megatron-Bridge 27B, py-spy on every stage process, 60 s | 15 | 9 |
| E3 | MIMO 9B, 1 vision GPU + TP2 x DP2 | 50 | 10 |
| E3 | rdsp 9B, same layout | 50 | 10 |
| E3 | py-spy of both on that layout, 40 s each | 20 + 20 | 12 |
| O1 | only with `RUN_O1=1`: Nsight traces of both on that layout | 20 + 20 | 15 |

Every log carries today's tag (for example `-nov12`), so October's files stay. Profiled runs are
diagnosis; only the 50-step runs give step times.

## Steps

All commands from `ops/h100/`.

| Step | Who | Command | Time |
|---|---|---|---|
| 1. Launch | you | `SHUTDOWN_MIN=240 ./launch.sh p5.48xlarge us-east-2a us-east-2b us-east-2c` (no capacity: `AWS_REGION=us-west-2 SHUTDOWN_MIN=240 ./launch.sh p5.48xlarge us-west-2c us-west-2a us-west-2b`) | 5 min |
| 2. Code | Claude | `./push.sh` | 1 min |
| 3. Setup | Claude | `./remote.sh ../../rdsp/bench_oct10/blog_round/prep.sh`; done at `=== PREP DONE` in `~/prep.out` | about 60 min |
| 4. Runs | Claude | `./remote.sh ../../rdsp/bench_oct10/blog_round/queue_profile.sh`; done at `=== ALL DONE BLOG` | about 90 min |
| 5. Settings check | Claude | after the first Bridge run: `python3 settings_diff.py` of its log against October's `bridge-gap-cut1.log`; only paths may differ, else stop the queue | 2 min |
| 6. Pull | Claude | `bash ../../rdsp/bench_oct10/blog_round/pull.sh` (safe to rerun) | 2 min |

## Time

About 3 hours of one p5.48xlarge.

## What comes out

- 27B: both step times on one machine; Megatron-Bridge's forward and backward seconds per
  stage; the share of its first stage spent in the vision encoder.
- 9B: both step times; forward, backward and waiting per GPU for both systems.
- A profiling report in `rdsp/ncompass/profiling-reports/`, and the 27B paragraph for the blog.
