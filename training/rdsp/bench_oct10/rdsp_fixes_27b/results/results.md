# rdsp fixes, Qwen3.5-27B, 8x H100 (p5.48xlarge us-west-2c), 2026-10-17

Seconds per step, median of steps 6-50. Bridge and old rdsp: results_27b_20261017.md (same box
type, data, settings).

## D: TP2 x PP2 x DP2

| Run | Master | Cut | Recompute | s/step | Loss mean, steps 41-50 |
|---|---|---|---|---|---|
| Bridge | 2 B (remainder) | 32 \| 32 | all | 17.47 | 0.0067 |
| Bridge | 2 B (remainder) | 32 \| 32 | 24 of 32 | 17.05 | 0.0078 |
| rdsp (17 Oct) | 4 B (fp32) | 32 \| 32 | all | 17.70 | 0.0055 |
| rdsp fix-d1 (this box) | 4 B (fp32) | 32 \| 32 | all | OOM, stage 1, step 1 | - |
| Bridge on rdsp's measured cut | 2 B (remainder) | 27 \| 37 | all | OOM, stage 1 (68.6 GiB by PyTorch, 79.2 GiB in use) | - |
| Bridge sweep, 20 steps (median 6-20; even split over the same steps: 17.76) | 2 B (remainder) | 30 \| 34 | all | 17.44 | - |
| Bridge sweep, 20 steps (median 6-20) | 2 B (remainder) | 29 \| 35 | all | 17.18 | - |
| **rdsp fix-d2** | **2 B (remainder)** | **27 \| 37 (measured)** | **all** | **16.40** (15.95-16.92) | 0.0083 |
| **rdsp fix-d3-rc20** | **2 B (remainder)** | **27 \| 37 (measured)** | **20 of 32 per stage before the re-cut; 15 of 27 \| 25 of 37 after (12 kept on each)** | **14.95** (14.27-15.26) | 0.0063 |

fix-d2's calibration (3 steps at lr 0, cut 32 | 32): stage 0 17.09 s, stage 1 12.15 s busy per
step; peaks 67.3 / 65.3 GiB (4-byte master on 17 Oct: 73.7 / 71.5). Re-cut 27 | 37, as the CPU
simulation from the 17 Oct timings predicted. No allocator retries (17 Oct: one per step).
Bridge's two runs do the same math and differ by 0.0011 in late loss: the spread is noise.

Bridge D recomputing 20 of 32 (block method) at its even cut: out of memory (3-step fit check,
2026-10-17 23:35), as predicted from its recompute-24 run at 80.1-80.3 GB on stage 0.

fix-d3-rc20: calibration stage 0 16.54 s, stage 1 10.91 s busy per step; 53 allocator retries in
50 steps (at the memory edge, as the 17 Oct rdsp run was). Bridge never tried keeping more than 8
layers' activations: its round-3 search tried 24 of 32 first and stopped there. At 24 of 32 its
stage-0 GPUs already reached 80.1-80.3 GB of 81.5 (nvidia-smi, includes the allocator cache), and
its even cut cannot move layers off stage 0, so 20 of 32 likely does not fit for it (not run).

## C: TP4 x DP2, no pipeline, full recompute

| Run | Master | s/step | Loss mean, steps 41-50 |
|---|---|---|---|
| Bridge | 2 B (remainder) | 26.95 | - |
| rdsp (17 Oct) | 4 B (fp32) | 29.26 | 0.0065 |
| rdsp fix-c1 | 2 B (remainder) | 31.12 (steady 30.2-31.9 from step 1) | 0.0072 |

fix-c1 is 6% slower than 17 Oct. The remainder's optimizer step costs ~0.08 s of the 1.85 s
(3.4B parameters per GPU at +35 ms per 1.6B). D's calibration timers also show forward +
backward 2-3% slower than 17 Oct's profile (17.09 vs 16.7 s, 12.15 vs 11.9 s). This box has the
same driver (595.91.07), kernel image and package versions as 17 Oct's. Control queued
(queue6.sh): C with the fp32 master, 12 steps, same box.

## E: TP2 x PP4, full recompute

| Run | Master | Cut (layers per stage) | s/step | Loss mean, steps 41-50 |
|---|---|---|---|---|
| Bridge | 2 B (remainder) | 16 \| 16 \| 16 \| 16 | 24.70 | - |
| rdsp (17 Oct) | 4 B (fp32) | 15 \| 18 \| 17 \| 14 (balanced estimate) | 22.65 | 0.0061 |
| Bridge on rdsp's measured cut | 2 B (remainder) | 5 \| 20 \| 20 \| 19 (num_layers_in_first/last_pipeline_stage) | 19.39 | 0.0060 |
| Bridge sweep, 20 steps (median 6-20) | 2 B (remainder) | 1 \| 22 \| 22 \| 19 | 17.43 | - |
| Bridge sweep, 20 steps (median 6-20) | 2 B (remainder) | 3 \| 21 \| 21 \| 19 | 18.16 | - |
| Bridge sweep, 20 steps (median 6-20) | 2 B (remainder) | 7 \| 19 \| 19 \| 19 | 20.26 | - |
| **rdsp fix-e1** | **2 B (remainder)** | **5 \| 20 \| 20 \| 19 (measured)** | **17.25** (16.26-17.82) | 0.0068 |
| **rdsp fix-e2-rc12** | **2 B (remainder)** | **3 \| 21 \| 21 \| 19 (measured), recompute 12 of 16 before the re-cut (4 layers keep activations per stage)** | **16.10** (15.19-17.11) | 0.0080 |

fix-e1's calibration at 15 | 18 | 17 | 14: stage busy 21.8 / 13.2 / 12.3 / 10.6 s per step; peaks
57.2 / 53.7 / 50.9 / 51.0 GiB. Stage 0 (vision encoder + embedding + 15 layers) took 65% longer
than stage 1 with 18 layers: the re-cut gave it 5 layers.

Per-layer cost from E's calibration (15 | 18 | 17 | 14, full recompute): stage 1 13.23 s / 18 layers
= 0.735 s, stage 2 12.28 s / 17 = 0.723 s per layer per step. Stage 0: 21.82 s - 15 x 0.73 = 10.9 s
beyond its layers = ~170 ms per image (64 rows, one image each). D's calibration gives the same
(17.09 s - 32 x (12.15 s / 32) = 4.9 s over 32 rows per GPU = ~155 ms per image). Stage 3 (output
layer and loss): 10.55 s - 14 x 0.73 = 0.3 s. The fixed per-image cost sits on stage 0 and does not
shrink with more stages, while each stage's share of the layers does.

## A: TP4 x PP2, no recompute

| Run | Master | Cut | s/step | Loss mean, steps 41-50 | Allocator retries |
|---|---|---|---|---|---|
| Bridge | 2 B (remainder) | 32 \| 32 | 25.31 | - | - |
| rdsp (17 Oct) | 4 B (fp32) | 32 \| 32 | 22.78 | 0.0122 | 49 |
| **rdsp fix-a1** | **2 B (remainder)** | **25 \| 39 (measured)** | **18.53** (17.77-19.52) | 0.0061 | 0 |

fix-a1's calibration at 32 | 32: stage 0 21.04 s, stage 1 14.93 s busy per step; peaks 64.0 / 52.9
GiB. Bridge's checkpoint converted alongside this run on GPU 7 (16 CPU cores at nice 19).

E at the same cut (5 | 20 | 20 | 19): Bridge 19.39 s, rdsp 17.25 s. Of rdsp's 7.45 s lead over
Bridge's even split, 5.31 s (71%) is the cut and 2.14 s (rdsp 11% faster) is the runtime. The cut
was measured on rdsp's stage costs; Bridge's own best cut may differ (sweep queued).

C control (same box, steps 6-12 medians): fp32 master 29.74 s (17 Oct box 29.37: +1%, the box
is not slower); remainder master 31.29 s (+5%). The remainder's optimizer step explains ~0.1-0.15 s;
D's per-stage timers show no forward/backward slowdown with it. Profiled pair queued (queue8).

## PP8: TP1 x PP8, full recompute (new layout, 27b_tp1pp8dp1)

| Run | Master | Cut | s/step | Real tok/s per GPU | Loss mean, steps 41-50 |
|---|---|---|---|---|---|
| Bridge, even split | 2 B (remainder) | 8 \| 8 \| 8 \| 8 \| 8 \| 8 \| 8 \| 8 | OOM on stage 0 at step 2 (first iteration 1114 s; 67.0 GiB by PyTorch, 78.7 GiB in use) | - | - |
| Bridge on the nearest cut Megatron can express (equal middle stages) | 2 B (remainder) | 1 \| 9 x 6 \| 9 | OOM on stage 7 at step 2 (68.1 GiB by PyTorch, 77.5 GiB in use) | - | - |
| Bridge, a cut Megatron can express that fits (3rd try) | 2 B (remainder) | 3 \| 9 x 6 \| 7 | 16.51 (15.96-18.64; first iteration 1110 s) | 492 | 0.0071 |
| **rdsp fix-p8** | **2 B (remainder)** | **1 \| 9 \| 10 \| 9 \| 9 \| 9 \| 10 \| 7 (measured)** | **11.20** (10.40-11.92) | **726** | 0.0063 |

Calibration at the balanced estimate 6 | 9 | 9 | 9 | 9 | 9 | 8 | 5: stage 0 13.26 s busy per step, stages
1-5 7.5-7.7 s (0.84 s per layer), stage 6 6.7 s, stage 7 (5 layers + output layer + loss) 6.1 s.
Stage 0 beyond its 6 layers: 13.26 - 6 x 0.84 = 8.2 s, ~130 ms per image at TP1. Peaks 65.5 GiB
(stage 0) and 48-54 GiB.

Bridge's own best E cut (sweep, first stage 1/3/7, last 19): first stage 1 gives 17.43 s (steps 6-20),
better than rdsp's cut (5: 19.23 s over the same steps). Bridge's stage 0 costs more than rdsp's, so
its best cut gives stage 0 fewer layers. At each system's best cut, full recompute: rdsp 17.25 vs
Bridge 17.43 (1%). rdsp found its cut by measuring; Bridge's took a sweep (its default even split:
24.70).

## GPU checks, corrected (scripts/gpu_checks2.py, 2026-10-17 23:36)

- Remainder master: 12.04 vs 14.04 bytes held per parameter (2.00 saved), masters identical after
  5 steps (max difference 0.0), optimizer step 102.9 vs 67.0 ms per 1.61B parameters.
- Vision varlen vs SDPA, both bf16, against SDPA in fp32: one attention call, output error 0.375%
  both, gradient 0.432% both; the whole 27B vision encoder, output error 2.4% (varlen) vs 3.1%
  (SDPA), weight gradients 27% vs 30%. Varlen is as accurate as SDPA. It is not used: no speedup
  at one image per microbatch (151 vs 131 ms forward + backward per microbatch, gpu_checks.py).

PP8 at each system's best: rdsp 11.20 s (726 real tok/s per GPU) vs Bridge 16.51 s (492): rdsp 32%
less time per step, 1.47x the throughput. Bridge needed three tries to find a cut that fits:
Megatron sizes only the first and last stages (the middle ones are equal), its even split and the
nearest cut to rdsp's run out of memory (stage 0, then the last stage), and the cut that fits
leaves 3 layers on stage 0 next to its slower vision encoder.

Best layout per system on one 8x H100 node (27B, 64 rows of 2048 per step): rdsp TP1 x PP8,
11.20 s; Bridge TP1 x PP8 at 3 | 9x6 | 7, 16.51 s (its best before this: TP2 x PP2 x DP2 recompute
24 of 32, 17.05 s).

C profiled pair (same box, 6 steps each, --profile, 2026-10-18 00:14-00:18): steps 1-5 fp32 master
29.0-29.9 s, remainder 28.9-29.5 s; median over ranks: forward 10.1 / 9.5 s, backward 18.9 / 19.0 s,
optimizer step 0.22 / 0.30 s, peak 72 / 66 GiB. The remainder master does not slow C: fix-c1's
31.12 s (slow from step 1) was that run, not the master; the cause is not established (the box was
also doing setup work through that evening). rdsp on C: ~29.3 s either master; Bridge 26.95 s.

## TorchTitan's regime: fully sharded data parallel over 8 GPUs, no TP or PP, full recompute

| Run | s/step | Real tok/s per GPU | Loss mean, steps 41-50 |
|---|---|---|---|
| rdsp, one stage, DeepSpeed ZeRO-3 (27b_dp8), fp32 master | 19.54 (18.33-20.07) | 416 | 0.0069 |
| TorchTitan FSDP over 8 GPUs (no recompute) | OOM (73.1-73.3 GiB by PyTorch) | - | - |
| **TorchTitan FSDP over 8 GPUs, FullAC** | **14.22** (14.13-14.50; from its log timestamps) | **571** (1,152 padded) | 0.0069 |

In its own regime TorchTitan is 27% faster than rdsp's ZeRO-3 (14.22 vs 19.54 s). It is the best
non-pipeline result on this node, ahead of Bridge's best (16.51 s); rdsp's TP1 x PP8 (11.20 s) is
21% faster than it.

## MegatronMIMO's regime: vision encoder on its own GPU (9B, 5 GPUs: vision 1 + language 4)

Same box, 2026-10-18, the 2026-10-11 settings (fp32 gradients and master, per-microbatch loss mean,
every row padded to 2048, no recompute; rdsp's cut 0,balanced). Median of steps 6-50.

| Layout | MIMO | rdsp | 2026-10-11 (MIMO / rdsp) |
|---|---|---|---|
| a: vision 1 GPU + language TP2 x PP2 | 9.78 s | **9.37 s** (1,388 real tok/s per GPU) | 9.52 / 9.11 |
| b: vision 1 GPU + language TP2 x DP2 | **7.87 s** (1,652) | 8.39 s (1,550) | 8.11 / 8.22 |

PP8 with per-stage recompute (recompute_blocks 8, the cut re-measured: 1 | 9 | 10 | 9 | 9 | 10 | 9 | 7):
11.17 s (10.66-11.68), 728 real tok/s per GPU, loss 0.0071. Same as full recompute (11.20 s).

Bridge A (TP4 x PP2, no recompute) first-stage sweep, 20 steps, median of steps 6-20: even 32 | 32
26.07 s (same steps of its 50-step run), 30 | 34 25.49 s, 28 | 36 24.31 s (still improving; smaller
first stages not tried). rdsp at its measured 25 | 39: 18.53 s, 24% faster than Bridge's best tried.

Bridge with Megatron FSDP over 8 GPUs (ddp.use_megatron_fsdp, optim_grads_params, every layer
recomputed): stopped first on "requires the fsdp_dtensor checkpointing format"; with
checkpoint.ckpt_format=fsdp_dtensor, out of memory while starting (77.6 GiB by PyTorch, before
step 1). Not tuned further.

Single-node report: https://claude.ai/artifact/GDZSB15b3iMwMHdK84Yq7A ("Where rdsp Wins").

## TP4 x PP2 gap, profiled (2026-10-18 03:03-03:47, diagnosis only)

Same cut 28 | 36 on both, no recompute. rdsp with its stage timers: ~20.3 s per step (profiled);
stage 0 forward 8.4 s + backward 11.0 s, stage 1 6.3 + 9.5 s (waits 3.9 s). Bridge at the same cut:
24.31 s (20-step run). So at TP4 the gap is per-step work, not the cut. Bridge's per-step timers go
to TensorBoard in this recipe, not its log, so both were traced with Nsight Systems, 40 s of steady
state each (bridge_vs_rdsp/27b_tp4pp2dp1/profiles/*.nsys-rep, scripts/nsys_stage_busy.py,
results_nsys_cut28.txt). Per step (Bridge ~1.66 steps in the window, rdsp ~2): matmul kernels
Bridge 2.3 / 3.0 s (stage 0 / 1) vs rdsp 1.9 / 2.6 s; other kernels 2.7 / 1.8 vs 2.9 / 2.4 s;
stage 0 launches ~263k vs ~295k kernels. NCCL kernels dominate both but include waiting for peers,
so the traces cannot split transfer from waiting. Not attributed further: that needs per-layer
NVTX ranges. (The script's "conv" column also counts the linear-attention layers' causal conv1d.)
