#!/usr/bin/env bash
# rdsp, Qwen3.5-9B, no pipeline: one stage of TP2 x DP4 on 8x H100. Four rows per
# microbatch (one per DP copy), 16 microbatches per step, as Megatron runs this layout.
# Every row padded to 2048 (Megatron-Bridge is told to do the same). The vision
# encoder runs eager, as in the TP4xPP2 run. Extra arguments go to train_vl.py.
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../../../.." && pwd)
exec python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-9B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 1 --cuts even \
    --stage 0:gpus=8,tp=2,zero=1 \
    --rows 4 --microbatches 16 --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype fp32 --loss token-mean --liger \
    --prefetch --sharded-loss --drop-padding-mask \
    "$@"
