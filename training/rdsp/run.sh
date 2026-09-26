#!/usr/bin/env bash
# Qwen3-0.6B on 4 GPUs: stage 0 data-parallel (ZeRO-2) on 2 GPUs, stage 1
# tensor-parallel on 2 GPUs. Extra arguments are passed to train.py.
set -euo pipefail
cd "$(dirname "$0")"
python train.py \
    --model Qwen/Qwen3-0.6B \
    --stages 2 \
    --stage 0:gpus=2,zero=2 \
    --stage 1:gpus=2,tp=2 \
    --microbatches 8 --rows 4 --seq 512 \
    --steps 100 --lr 1e-5 \
    "$@"
