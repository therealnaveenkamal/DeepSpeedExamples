#!/usr/bin/env bash
# rdsp, Qwen3.5-27B, no pipeline: one stage of TP4 x DP2 on 8x H100, ZeRO-1 (fp32 master
# and Adam split over the 2 DP copies). Two rows per microbatch (one per DP copy), 32
# microbatches per step, as Megatron runs this layout. Every row padded to 2048. Recompute
# every block (it does not fit without, estimate in plan_27b_50step.md), bf16 gradient
# accumulation, fused AdamW, optimizer groups of 250M. Vision encoder eager. Extra
# arguments go to train_vl.py.
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../../../.." && pwd)
# RECOMPUTE=0: no recompute (a fit check); ZERO: ZeRO stage (default 1); PROFILER: a command
# prefix such as an nsys profile call
RC=",recompute=1"; [ "${RECOMPUTE:-1}" = 0 ] && RC=""
# RC_BLOCKS=N: recompute only each stage's first N decoder layers (and the vision encoder)
[ -n "${RC_BLOCKS:-}" ] && RC=",recompute_blocks=$RC_BLOCKS"
exec ${PROFILER:-} python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-27B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 1 --cuts even \
    --stage 0:gpus=8,tp=4,zero=${ZERO:-1}$RC \
    --rows 2 --microbatches 32 --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype bf16 --fused-adam --optimizer-group-numel 250000000 \
    --loss token-mean --liger --prefetch --sharded-loss --drop-padding-mask \
    "$@"
