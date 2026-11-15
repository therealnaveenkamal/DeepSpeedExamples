# Two nodes, 16x H100: do the single-node conclusions hold at scale?

Approved 2026-10-18 ("yes to both, 128 rows, go"). Single-node map: "Where rdsp Wins",
https://claude.ai/artifact/GDZSB15b3iMwMHdK84Yq7A; rdsp_fixes_27b/results/results.md.

No new search for best layouts or cuts: each system's single-node configuration, scaled to 16 GPUs.

## Settings

- Qwen3.5-27B (same model as the single-node map; ~30B vision-language as the blog direction asks).
- 128 rows of 2,048 per step: each GPU does the same work as on one node, so per-GPU throughput
  compares directly with the single-node numbers. Bridge's rows exported again at 128 per step.
- Everything else as on one node: token-mean loss, AdamW lr 1e-5 constant, bf16 weights and
  gradients, fp32 master weights in 2 bytes, 50 steps, median of steps 6-50.
- Network: EFA, 32 cards per node (3,200 Gbps). TCP not in this round.

## Runs

| Single-node result to re-check | 16-GPU run |
|---|---|
| Best of each: rdsp 725 > TorchTitan 571 > Bridge 492 real tok/s per GPU | rdsp TP1 x PP8 x DP2 (measured cut); TorchTitan FSDP over 16; Bridge TP1 x PP8 x DP2 at 3 \| 9x6 \| 7 |
| Bridge's ground (TP4 x DP2): Bridge 8% faster | Bridge and rdsp TP4 x DP4, every layer recomputed |
| TorchTitan's ground (FSDP over 8): TorchTitan 27% faster | TorchTitan FSDP over 16 (above); rdsp ZeRO-3 over 16 |
| MIMO does not fit 27B on one node | MIMO: vision on its own GPU + language on 8 (fit check); rdsp on the same layout if it fits |
| Prediction: deeper pipeline, bigger rdsp lead | rdsp TP1 x PP16 (pipeline across both nodes) |

In TP1 x PP8 x DP2 both systems put each stage's two data-parallel copies on one node and stages
0-3 on the first node, 4-7 on the second (Megatron's default rank order; rdsp's stages fill the
driver's node first): the pipeline crosses the network once.

## Phase 0: code, no GPUs

1. rdsp: a stage needing more GPUs than a node has spans nodes (TP4 x DP4 and ZeRO-3 over 16 are
   single 16-GPU stages), with its tensor-parallel groups kept inside a node.
2. rdsp: the Ray cluster over both nodes (head on node 1, worker on node 2), NCCL over EFA in the
   workers' environment; the driver joins the running cluster.
3. Bridge and MIMO: launch on both nodes at once (runs.sh NNODES mode, ssh from node 1 to node 2);
   MIMO's launcher gets the same mode.
4. TorchTitan: torchrun over 2 nodes; FSDP16 config.
5. Data and checkpoints: Bridge's rows exported at 128 per step; the Bridge and MIMO checkpoints
   converted once and copied to node 2; HF weights on both nodes.
6. Layout scripts per run (bench_oct10/two_node_27b/<layout>/), a queue, and a 2-node guard.
7. CPU tests for 1 and 2; a 2-process gloo test of a stage spanning "nodes".

## Phase 1-2: GPUs

| Phase | What | Time |
|---|---|---|
| 1 | launch2.sh, setup both nodes, NCCL bandwidth check over EFA, 3-step smoke run of each system | ~1.5 h |
| 2 | The 9 runs above | ~4 h |

Total: ~5.5 h on 2x p5.48xlarge (us-west-2c). A pair in one
placement group may need on-demand or a capacity block.
