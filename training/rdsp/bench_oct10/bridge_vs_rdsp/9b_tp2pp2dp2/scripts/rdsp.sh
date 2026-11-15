#!/usr/bin/env bash
# rdsp, Qwen3.5-9B, 2 pipeline stages of TP2 x DP2 on 8x H100. Two rows per
# microbatch (one per DP copy), 32 microbatches per step, as Megatron runs this layout.
# Every row padded to 2048 (equal work with Megatron-Bridge under PP). The vision
# encoder runs eager, as in the TP4xPP2 run. Extra arguments go to train_vl.py.
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../../../.." && pwd)
exec python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-9B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 2 --cuts balanced \
    --stage 0:gpus=4,tp=2,zero=1 \
    --stage 1:gpus=4,tp=2,zero=1 \
    --rows 2 --microbatches 32 --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype fp32 --loss token-mean --liger \
    --prefetch --sharded-loss --drop-padding-mask \
    "$@"
