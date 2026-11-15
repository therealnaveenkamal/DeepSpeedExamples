#!/usr/bin/env bash
# rdsp, Qwen3.5-27B, data parallel 8 with ZeRO-3 on 8x H100: one stage, weights, gradients and
# optimizer state split eight ways (DeepSpeed's fully sharded mode; TorchTitan runs this regime as
# FSDP over 8 GPUs, torchtitan/cord_qwen35.py qwen35_27b_cord_fsdp8). 8 rows per microbatch (one
# per GPU), 8 microbatches per step = 64 rows, every row padded to 2048. Recompute every block,
# bf16 gradient accumulation, fp32 master weights (the remainder master needs ZeRO 1 or 2), fused
# AdamW. Extra arguments go to train_vl.py.
set -euo pipefail
RDSP=$(cd "$(dirname "$0")/../../../.." && pwd)
RC=",recompute=1"; [ "${RECOMPUTE:-1}" = 0 ] && RC=""
exec ${PROFILER:-} python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-27B --dataset "${DATA:?set DATA=exported:<dir>}" \
    --stages 1 --cuts even \
    --stage 0:gpus=8,tp=1,zero=${ZERO:-3}$RC \
    --rows 8 --microbatches 8 --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-50}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype bf16 --fused-adam \
    --loss token-mean --liger --prefetch --sharded-loss --drop-padding-mask \
    "$@"
