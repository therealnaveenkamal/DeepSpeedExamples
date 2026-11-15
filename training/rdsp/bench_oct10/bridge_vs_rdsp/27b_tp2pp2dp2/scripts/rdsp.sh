#!/usr/bin/env bash
# rdsp, Qwen3.5-27B, TP2 x PP2 x DP2 (2 pipeline stages of TP2 x DP2) on 8x H100. 2 row(s) per microbatch (one per DP copy), 32
# microbatches per step, as Megatron runs this layout. Every row padded to 2048. Recompute every
# block (27B does not fit this layout without: plan_27b_50step.md), bf16 gradient accumulation,
# fused AdamW, optimizer groups of 250M. Vision encoder eager. Extra arguments go to train_vl.py.
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../../../.." && pwd)
# RECOMPUTE=0: no recompute (a fit check); ZERO: ZeRO stage (default 1); PROFILER: a command
# prefix such as an nsys profile call
RC=",recompute=1"; [ "${RECOMPUTE:-1}" = 0 ] && RC=""
# RC_BLOCKS=N: recompute only each stage's first N decoder layers (and the vision encoder)
[ -n "${RC_BLOCKS:-}" ] && RC=",recompute_blocks=$RC_BLOCKS"
# COMPILE_VISION=1: torch.compile the vision encoder's blocks on stage 0 (rdsp's recipe; off at TP4,
# where 4304 / 4 = 1076-wide rows are not 16-byte aligned and the compiled kernel crashed)
CV=""; [ "${COMPILE_VISION:-0}" = 1 ] && CV=",compile_vision=1"
exec ${PROFILER:-} python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-27B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 2 --cuts balanced --stage 0:gpus=4,tp=2,zero=${ZERO:-1}$RC$CV --stage 1:gpus=4,tp=2,zero=${ZERO:-1}$RC \
    --rows 2 --microbatches 32 --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype bf16 --fused-adam --optimizer-group-numel 250000000 \
    --loss token-mean --liger --prefetch --sharded-loss --drop-padding-mask \
    "$@"
