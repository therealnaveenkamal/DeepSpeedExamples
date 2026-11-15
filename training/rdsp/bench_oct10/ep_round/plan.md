# EP round: rdsp's 35B two-node layouts with expert parallelism, no recompute

2026-10-19. The slide's rdsp row (vision 4 GPUs + 6 language stages × 2, 6.38 s per step,
128 × 2,048 receipt rows) recomputes every block: without, it ran out of memory before step 0,
because each language GPU holds all 256 experts of its layers (DP2, no expert splitting).
Megatron-Bridge (EP8) and TorchTitan (EP8) recompute nothing. The user approved splitting rdsp's
experts too and choosing the layout again from measured times.

## Runs (scripts/queue_ep.sh)

| Run | Layout | Recompute |
|---|---|---|
| R0 | PP8 × DP2, EP2 on every stage, measured cut | none (fallback: encoder only) |
| R1 | vision K2 + language stages of 2, EP2 (K2 from R0) | none (fallback: every block) |
| R2 | vision K4 + language stages of 4, EP4 (K4 from R0) | none (fallback: every block) |
| R3 | the slide's run on these boxes: vision 4 + 6 × 2, no EP | every block |

K comes from `two_node_35b/scripts/layout_from_log.py vision-width <R0 log> 16 <stage GPUs>`, as
the slide's K=4 came from the PP8 × DP2 log on 2026-10-18. Each stage sits inside one node
(STRICT_PACK), so EP's token exchange stays on NVLink; AutoEP across both nodes crashed on
2026-10-18 (TP2 × EP16) and is not used. Everything else as the slide: same rows, AdamW 1e-5,
bf16 gradients, 2-byte master, 50 steps, median of steps 6-50.

Before the runs: AutoEP's GPU integration tests (`tests/integration/test_heterogeneous_pipeline.py
-k autoep`, 7 GPUs) on node 1.

## Launch

`ops/h100/go_ep.sh <spot|on-demand>`: launches 2 × p5.48xlarge (us-east-2, us-west-2, us-east-1
in turn), copies the code, starts the queue on node 1; self-shutdown after 300 minutes. Watch with
`GUARD_ENV=env2.sh bash guard.sh queue_ep.out 60 12 20 5`. Prep about 60-90 minutes (35B weights,
NeMo container), runs about 40-60 minutes.
