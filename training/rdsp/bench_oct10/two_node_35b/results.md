# Two nodes, Qwen3.5-35B-A3B: results (2026-10-18)

2x p5.48xlarge (16 H100 80 GB), us-west-2c, NCCL over EFA (all-reduce 1 GiB over 16 GPUs:
435 GB/s bus bandwidth, provider efa). Settings: plan.md ("Settings"). Logs:
box_logs_20261018/ (pulled from both nodes).

Step = median of steps 6-50. Real tokens per step: 129,997 (mean over steps 6-50, every run
the same rows). Per-GPU rate = real tokens per step / step / 16.

## Runs that finished

| System, layout | Recompute | Step | Real tok/s per GPU | Total real tok/s |
|---|---|---|---|---|
| rdsp, vision encoder on its own stage of 4 GPUs + 6 language stages of 2 (K=4, --cuts 0,measured) | every block (out of memory without) | **6.38 s** | **1,273** | 20,369 |
| rdsp TP1 x PP8 x DP2 (--cuts measured: 1 \| 6x6 \| 3) | stage 0's vision encoder | 9.60 s | 846 | 13,541 |
| TorchTitan FSDP over 16 | none | 11.84 s | 686 | 10,983 |
| rdsp ZeRO-3 over 16 | every block (out of memory without) | 18.33 s | 443 | 7,093 |
| rdsp TP1 x PP16 (--cuts measured) | stage 0's vision encoder | 18.24 s | 445 | 7,127 |
| Bridge TP2 x EP16, its own recipe for this model (expert parallelism: not in the 27B comparison) | none | 14.94 s | 544 | 8,701 |
| (the same, first run, with runs.sh's earlier multi-node flags that replaced the container's LD_LIBRARY_PATH) | none | 15.72 s | 517 | 8,270 |
| Bridge TP1 x PP8 x DP2 at rdsp's cut (1 \| 6x6 \| 3), each pipeline inside one node (dist.use_tp_pp_dp_mapping=true) | none | 20.50 s | 396 | 6,341 |
| Bridge TP1 x PP8 x DP2 at rdsp's cut, pipeline across the nodes, NCCL between the nodes over TCP (see below) | none | 20.27 s | 401 | 6,413 |

## Dense against sparse: the same two rdsp layouts on Qwen3.5-27B

The dense 27B on the same two nodes, settings and layouts (its rows exported at 128 a step:
logs-27b/ in each layout directory). K for the vision stage from the 27B's own PP8 run, as for
the 35B (scripts/layout_from_log.py vision-width): 4 again.

| rdsp, 16 H100 | Dense 27B | Sparse 35B-A3B |
|---|---|---|
| TP1 x PP8 x DP2, --cuts measured (stage 0's vision encoder recomputed) | **10.19 s**, 798 per GPU | 9.60 s, 846 per GPU |
| vision encoder on its own stage of 4 GPUs (every block recomputed: out of memory without) | 12.06 s, 674 per GPU | **6.38 s**, 1,273 per GPU |
| vision stage against PP8 | -16% throughput | +50% throughput |

rdsp's calibration at 5-6 layers a stage, PP8 x DP2 (seconds of busy time per step):

| | Dense 27B | Sparse 35B-A3B |
|---|---|---|
| stage 0: vision encoder + embedding + layers | 12.7 (6 layers) | 11.9 (5 layers) |
| a middle stage | ~6.0 (9 layers) | ~3.4 (5 layers) |

With the vision stage the language model has 12 GPUs instead of 14 and full recompute. The
27B's language stages then take 8.3-9.6 s and set the pace; the 35B's take 3.7-4.9 s (a token
touches about 3B of its 35B parameters), against 4.6 s for the vision stage. A separate vision
stage pays where the language model is light per token and the vision encoder is not: sparse
(MoE) vision-language models.

rdsp's choice of K weighs measured time only, not the memory that forced full recompute: for the
27B it chose 4 where keeping the encoder on stage 0 (PP8) was faster.

## Where TorchTitan's step goes (Nsight Systems)

TorchTitan FSDP16, steps ~7-10 of a 15-step rerun (step times 11.7-11.9 s, as in the 50-step
run), node 1's 8 GPUs, a 36 s window (fsdp16/logs/nsys-titan-busy.txt; the report and its
sqlite export are on the box):

| Per GPU, seconds of the 36 s window | |
|---|---|
| any kernel running | 31.7-32.5 |
| NCCL kernels running (communication, including waiting for a peer) | 29.2-30.0 (about 83%) |
| matrix multiplies | 2.2 (6%) |

FSDP gathers each layer's full weights on every GPU for every microbatch: all 35B parameters,
every one of the 256 experts, while a token uses 8 experts (about 3B active parameters). The
step is communication; TorchTitan's own MFU line says 2.7%. rdsp's pipeline keeps each stage's
weights on its GPUs and sends activations between stages.

## Runs that did not finish

| System, layout | What happened |
|---|---|
| Bridge TP1 x PP8 x DP2, pipeline across the nodes (Megatron's default rank order), NCCL over EFA | Deadlocked across the node boundary, twice (the second time with the container's own library path and NCCL INFO logs: EFA selected, NCCL 2.30.7). Every rank in Megatron's batched send/receive (batch_isend_irecv); stages 3 (node 1) and 4 (node 2) each inside their matching exchange with the other. Bridge's cross-node collectives worked (TP2 x EP16); within one node its PP8 worked (27B). |
| Bridge TP4 x DP4, no expert parallelism | Transformer Engine: "Maximum number of tensors reached: 20321" in the optimizer's fused step (multi_tensor_apply): every rank holds all 256 experts of all 40 layers as separate tensors. The same with Bridge's fp32 master (MASTER=fp32). With expert parallelism (16 experts a rank) Bridge runs. |
| rdsp TP2 x EP16 (Bridge's layout) | NCCL "unhandled cuda error" in DeepSpeed AutoEP's first all-to-all across the 16 ranks. Expert parallelism is out of scope; not investigated. |
| rdsp TP4 x DP4 | Not run: rdsp's tensor parallelism leaves the routed experts whole, so each GPU would hold all 32B of them. |
| MegatronMIMO, Qwen3.5-35B-A3B | Not runnable. NeMo 26.08 (and 26.08.01): its MoE checkpoint conversion fails (fixed after the cut, in Megatron-Bridge git tag 26.09-beta.rc1; no container has it; nvcr.io/nvidia/nemo:dev is from April 2025). Even then its launcher (26.08 and Bridge main) hard-codes fp32 gradients and fp32 master weights (about 18 bytes per parameter, about 630 GB for the language model), has no recompute and no expert parallelism, and the language model takes at most TP2 (2 key/value heads): every layout that leaves GPUs for the images needs more than 80 GB a GPU (TP2 x PP4 on 8 GPUs: about 79 GB before activations). |
| MegatronMIMO, dense Qwen3.5-27B, its own regime: language TP4 x PP2 on node 1, images DP8 on node 2 (mimo27b/) | Out of memory on rank 0 (language stage 0) before the first step finished (79.2 GB in use); the same launcher limits. On one node it did not fit the 27B either. |

## The 27B claims at two nodes

| 27B claim (one node, 8 GPUs) | 35B-A3B, two nodes |
|---|---|
| Best of each: rdsp 725 > TorchTitan 571 > Bridge 492 real tok/s per GPU | Holds, wider: rdsp 1,273 > TorchTitan 686 > Bridge 544 (its own EP recipe; without EP, PP8 x DP2: 396-401). |
| Same pipeline layout: rdsp PP8 11.20 s vs Bridge PP8 16.51 s on one node (1.47x) | Holds, wider: rdsp PP8 x DP2 9.60 s vs Bridge PP8 x DP2 20.50 s at the same cut (2.1x), Bridge's pipelines inside a node, rdsp's across the nodes. |
| TorchTitan's ground: TorchTitan FSDP 27% faster than rdsp ZeRO-3 | Holds, wider: TorchTitan 55% faster (11.84 s vs 18.33 s). rdsp's ZeRO-3 needed full recompute, TorchTitan's FSDP did not. |
| Bridge's ground (TP + DP): Bridge 8% faster | No comparison: on this MoE model neither runs TP x DP without expert parallelism (rdsp: experts not split by TP, out of memory; Bridge: Transformer Engine's tensor limit). |
| Mechanism: stage 0 carries the vision encoder; rdsp's measured cut moves layers off it | Holds, stronger. Calibration at 5 layers a stage: stage 0 11.9 s, the next stages 3.4 s. The cut went to 1 layer and stage 0 still set the pace; the vision stage of its own (4 GPUs) took the step from 9.60 s to 6.38 s (+50% throughput). |
| Two-node prediction: deeper pipeline, bigger rdsp lead | Fails: PP16 is 18.24 s, the vision encoder on one GPU. |

## Bridge's pipeline across the nodes

Over EFA it deadlocked twice in Megatron's batched send/receive (above). A third attempt mounted
the host's aws-ofi-nccl 1.21.1 over the container's 1.17.3 (built for NCCL 2.28, beside NCCL
2.30.7) and set NCCL_P2P_PXN_LEVEL=0 (the hang's last NCCL lines show stage 3 sending through
another stage's NIC, PXN). The host plugin did not load (the host's libfabric needs a newer
libefa than the container's: "version `EFA_1.7' not found"), so NCCL fell back to TCP sockets
between the nodes: 50 steps, 20.27 s, no deadlock; with each pipeline inside a node, 20.50 s.
The cause over EFA is not isolated (the container's plugin version, or PXN). No published fix
found (Megatron-LM after cd4afffa, Megatron-Bridge issues, NCCL 2.30-2.32 and aws-ofi-nccl
1.17-1.21 release notes); a similar open report: NVIDIA-NeMo/Megatron-Bridge#5006.

## Problems found on the way (fixed unless noted)

- Liger's fused MoE kernel: the first step took 350 s against 24 s with Hugging Face's grouped
  matmuls (compiling and benchmarking Triton kernels), past rdsp's 5-minute NCCL wait: the run
  crashed. rdsp now defaults to Hugging Face's experts (--moe-experts hf).
- rdsp PP8 x DP2: stage 0 out of memory at step 1 (Adam's state); stage 0's vision encoder is
  now recomputed by default (recompute_blocks=0).
- Ray workers on a cluster started with `ray start` do not see the driver's environment
  (RDSP_PROFILE, HF_HOME, ...): train_vl.py passes it (found before the runs).
- Prep: rsync into a missing parent directory on node 2; an old Ray cluster blocking a new one.
- runs.sh's multi-node flags replaced the NeMo container's LD_LIBRARY_PATH, dropping its CUDA
  forward-compatibility library (container CUDA 13.3 on a 13.2 driver); removed. Not the cause
  of Bridge's PP deadlock.
