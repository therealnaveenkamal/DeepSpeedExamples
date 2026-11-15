# Gap round: competitor settings the slides left out (2026-10-18)

Approved 2026-10-18 ("okay go ahead!"). Why: an audit of the competitor numbers on the
PyTorchCon slides 14-15 found settings each framework ships or supports that our runs never
used, so the slides may understate Megatron-Bridge and TorchTitan. Nothing goes back on a slide
until these runs are in.

Same equal-work settings as before (Bridge's CORD-v2 rows in Bridge's order, every row padded
to 2048, token-mean loss, AdamW lr 1e-5 constant, 50 steps, median of steps 6-50). Allowed:
parallel layout, cuts, recompute choice, overlap and check flags, gradient reduce dtype where it
matches the others (bf16). Not allowed: forced expert load balancing, other data, batch or
sequence length.

## Runs

| # | Where | System | Run | Gap it closes |
|---|---|---|---|---|
| A1 | node 1 | Bridge 27B TP1 x PP8 | rdsp's measured cut 1\|9\|10\|9\|9\|9\|10\|7 through `model.pipeline_model_parallel_layout` | Megatron takes any per-stage cut (our reports said first/last only) |
| A2 | node 1 | Bridge 27B TP1 x PP8 | 0\|9\|10\|10\|10\|9\|9\|7 (stage 0: vision encoder only) | Bridge's own stage 0 costs more than rdsp's |
| A3 | node 1 | Bridge 27B TP1 x PP8 | rdsp's cut as 2 interleaved chunks per GPU (virtual pipeline 2) | interleaved 1F1B never tried |
| A4 | node 2 | TorchTitan 27B FSDP8 | FullAC, fp32 reduce: the slide's setting on this box | control |
| A5 | node 2 | TorchTitan 27B FSDP8 | FullAC, bf16 reduce | gradients sent in fp32 |
| A6 | node 2 | TorchTitan 27B FSDP8 | only the MLP recomputed (RegionAC), bf16 reduce | full recompute costs ~48% |
| A7 | node 2 | rdsp 27B TP1 x PP8 | --cuts measured, 2-byte master (the slide's 11.20 s) | same-box rerun |
| B1 | both | Bridge 35B TP1 x EP8 x DP16 | experts split inside each node, no TP (NVIDIA's perf recipes) | we ran the SFT recipe's TP2 x EP16 |
| B2 | both | TorchTitan 35B FSDP16 + EP8 | 50 steps (31 ran on 2026-10-18: 9.07 s) | the slide shows TorchTitan without EP |
| B3 | both | TorchTitan 35B FSDP16 + EP8 | bf16 reduce | gradients sent in fp32 |
| B4 | both | rdsp 35B PP8 x DP2, then vision stage | same-box reruns of the slide's 9.60 s and 6.38 s | same-box rerun |
| B5 | both | NCCL | all-reduce and all-to-all, host stack and inside Bridge's container | Bridge's container EFA plugin never measured for all-to-all |
| B6 | both | Bridge 35B TP1 x PP2 x EP8 | NVIDIA's H100 perf layout without its 12 virtual stages (40 layers do not split into 2 x 12), `batch_p2p_comm=false` | perf layout |

Bridge runs A1-A3, B1, B6 also turn off the NaN checks NVIDIA's perf recipes turn off, use 8 data
workers, and (pipelines) drop the sync after every batched send/receive; A3 overlaps its sends.

Not run: NVIDIA's H100 perf recipe as shipped (PP2 x VP12 needs 40 layers to split into 24 equal
chunks; its expert all-to-all overlap needs a schedule plan Qwen3VLModel lacks), forced load
balancing (not equal work).

## Time

Prep ~75 min (both boxes from scratch), phase A ~95 min (both nodes at once), phase B ~3 h.
About 6 h on two p5.48xlarge.

## Added 2026-10-18 (user: "fix all the gaps ... their best config for 1 and 2 nodes"; "also run
## qwen 3.5 27B in 16 gpus"): the other two cells of the grid, scripts/queue2.sh

| # | Where | System | Run |
|---|---|---|---|
| C1 | node 2 | Bridge 35B one node | TP1 x EP8 (NVIDIA's 8-GPU perf layout); every layer recomputed if it does not fit |
| C2 | node 2 | Bridge 35B one node | TP1 x PP2 x EP4; recompute if needed |
| C3 | node 1 | rdsp 35B one node | 8 stages of 1 GPU, measured cut; then the vision stage at the width rdsp picks |
| C4 | node 1 | TorchTitan 35B one node | FSDP8 + EP8, FullAC, bf16 reduce: fit check (fp32 weights and Adam ~70 GB a GPU) |
| D1 | both | Bridge 27B two nodes | NVIDIA's 16-GPU recipe TP4 x PP4, unbatched sends; recompute if needed |
| D2 | both | Bridge 27B two nodes | PP8 x DP2, pipelines inside each node, at its faster phase-A cut |
| D3 | both | TorchTitan 27B two nodes | FSDP16, bf16 reduce: no recompute, else MLP-only, else full |
| D4 | both | rdsp 27B two nodes | PP8 x DP2, measured cut (same-box rerun of 10.19 s) |

About 4-4.5 h more: ~10.5 h in all.

2026-10-19 00:58 UTC: B4 and D4 (two-node rdsp reruns) dropped, the user's call ("no need"): the
slides keep the 2026-10-18 rdsp numbers for two nodes (6.38 s, 9.60 s, 27B 10.19 s). Saves ~50 min.

2026-10-19 01:10 UTC: TorchTitan pipeline added (user: "why torchtitan has fsdp over 8 GPUs? not
PP8?"). TorchTitan's PP never ran past 2 stages: TP2 x PP4 crashed loading weights (its Qwen3.5 HF
loader wants the embedding or the output layer on every stage: state_dict_adapter.py from_hf),
TP2 x PP2 x DP2 ran out of memory even with FullAC, TP4 x PP2 took 66.0 s with FullAC. Now:
A8 (node 2, phase A) TorchTitan TP1 x PP8 at rdsp's cut, weights converted to DCP with its own
convert_from_hf.py; D5 (phase D) TorchTitan PP8 x DP2 the same way. ~+45 min.

2026-10-19 02:00 UTC: D6 Megatron-MIMO on the 27B over 2 nodes, approved as an exception to "never
modify MIMO" (user's choice: "Edit MIMO, 2 nodes"). mimo/patch_mimo.py patches a /tmp copy of the
container's launcher with Bridge's memory settings (bf16 main grads, 2-byte master, bf16 reduce;
MIMO_RECOMPUTE=1 adds full recompute); the diff is mimo/patch.diff. Results labeled "MIMO with our
memory settings". Layouts: language TP4 x PP2 + images DP8 (8 rows a microbatch), and images
TP4 x DP2 (2 rows). Conversion in phase D prep. Shutdown moved to 14:57 UTC.

2026-10-19 02:25 UTC: phase E (scripts/queue3.sh, after queue2), user: "just use their recipe".
NVIDIA's published fine-tuning setup for this architecture (Qwen3.6-35B-A3B "SFT · H100", 2,507
tokens/s per GPU counting every position): MedPix, 32 rows of 4096 a step, Bridge at TP1 x PP2 x
EP8 with every layer recomputed, run by Bridge (that layout if it ran in phase B, and TP1 x EP8),
TorchTitan (FSDP16 + EP8 bf16) and rdsp (PP8 x DP2 measured, then its vision stage). Not identical
to NVIDIA's: Bridge 0.6.0 not nightly, images at native size (no 448x448 resize in 0.6.0),
Qwen3.5 weights, our equal-work precision. ~2.5 h; shutdown moved to ~16:45 UTC.

2026-10-19 02:45 UTC: user: "run all the layouts that we showed in our ppt with MedPix, 32 rows x
4,096 per step, 50 steps". queue3.sh's unread tail rewritten in place:
- phase E (35B, two nodes, MedPix 32 x 4096): Bridge without recompute first (recompute only if it
  does not fit, the rule for every system), PP2 x EP8 (if it ran in phase B), TP1 x EP8, and slide
  15's TP2 x EP16; TorchTitan FSDP16 + EP8 and slide 15's FSDP16, bf16 reduce; rdsp PP8 x DP2,
  then its vision stage.
- phase F (27B, one node, MedPix 32 x 4096; one_node_medpix.sh): each system at its fastest
  phase-A setting (scripts/medians.py --best): Bridge PP8 at its best cut on node 1, TorchTitan's
  best on node 2, then rdsp PP8.
Official 27B reference found: NVIDIA's Qwen3.8-27B page (qwen35_vl_27b recipes, Qwen3.5 class):
Benchmark · H100, 16 GPUs TP4 x PP4, MBS2, 32 x 4096 MockVLM, no recompute: 4.591 s, 1,784
tokens/s per GPU (every position). Shutdown 18:22 UTC.

2026-10-19 03:50 UTC, the user: "remove phase D" (27B on two nodes: repeating), phase C (35B on one
node) only if it makes sense: dropped (the 35B is a two-node model in NVIDIA's own docs, and on one
node every system is memory-bound; the 27B covers one node). queue2.sh now: A8 (TorchTitan PP8 on
the 27B, one node, on node 2) while node 1 converts MIMO's checkpoint, then MIMO 27B two nodes
(patched), against rdsp's 10.19 s. Phase E: each competitor at its fastest phase-B setting
(medians.py --best) plus Bridge's NVIDIA layout if it ran; rdsp PP8 x DP2 then its vision stage.
Phase F unchanged. Reports: tokens/s per GPU counted NVIDIA's way (every position) and real tokens.

2026-10-19 04:20 UTC, the user: stick to 27B on one node and 35B on two nodes; MIMO with the vision
encoder alone on 1 GPU. MIMO 27B now: language TP4 x PP2 on node 1 + vision on 1 GPU of node 2
(9 GPUs, runs.sh MIMO_NPROC0=8 MIMO_NPROC1=1), 1 row a microbatch; as shipped first, then
patch_mimo.py, then recompute. Replaces the vision-on-8-GPUs layouts. Also queued after B6: B6 with
NVIDIA's measured settings (17 | 23 cut, TE fused cross-entropy); phase E takes it when it ran.

2026-10-19 04:25 UTC: a third box for Megatron-MIMO's tutorial layout, as shipped (user: "can we
request a p5.4xlarge with EFA?"): ops/h100/launch3.sh, p5.4xlarge (1x H100, 1 EFA),
same placement group and security group; setup ops/h100/box3_prep.sh. queue2
ends with MIMO over 17 GPUs (bench/two_node/three.sh): language TP4 x PP2 x DP2 on the two 8-GPU
nodes, the vision encoder on the third box.
Result so far: Bridge TP1 x PP2 x EP8 across the nodes ran (no deadlock with
model.batch_p2p_comm=false): 9.28 s.

2026-10-19 04:35 UTC, the user: "add a 17 gpu rdsp run too ... once mimo is done". queue2 now ends: MIMO 17 GPUs, then the 27B's rows at 128 a step on all three boxes,
the 3-node Ray cluster (bench/two_node/ray_cluster3.sh), rdsp over 17 GPUs (scripts/rdsp17.sh,
driver on the third box: vision encoder on its GPU, language 8 stages x 2 on the 8-GPU nodes;
every block recomputed only if needed).
Phase B results: Bridge TP1 x PP2 x EP8 even 9.28 s; with NVIDIA's settings (17 | 23, TE fused
cross-entropy) 8.67 s, Bridge's best; TP1 x EP8 x DP16 with recompute 10.68 s.
