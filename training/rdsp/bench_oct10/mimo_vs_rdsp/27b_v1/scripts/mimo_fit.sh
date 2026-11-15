#!/bin/bash
# MegatronMIMO, Qwen3.5-27B: vision encoder alone on 1 GPU (rank 4) + language TP4 on ranks 0-3.
# The language model can have at most 4 GPUs here: TP divides 4 KV heads, PP divides 64 layers,
# DP divides the 64-row batch, so TP x PP x DP is a power of 2 and 8 would leave no GPU for vision.
# MIMO's own script, as shipped; flags as bench/mimo/runs.sh (MIMO_TRAIN), STEPS (default 3).
#   convert   HF -> MegatronMIMO checkpoint, in the fit layout (language TP4 on ranks 0-3, images
#             on rank 4): runs.sh's language=tp=1 puts all 27B on one GPU and runs out of memory
#   fit       the 3-step run
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../../../.." && pwd)
LOGS=$HERE/../logs; mkdir -p "$LOGS"
MODEL=Qwen3.5-27B
container() {  # as bench/mimo/runs.sh
  sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
    -v "$HOME/hf:/root/.cache/huggingface" -v "$HOME/mimo:/workspace" -v "$RDSP:/rdsp" \
    -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge \
    nvcr.io/nvidia/nemo:26.08 bash -c "$*"
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo"
}
case "${1:-}" in
  convert)
    container "python -m torch.distributed.run --nproc_per_node=5 examples/conversion/convert_megatron_mimo.py \
      import --hf-model Qwen/$MODEL --megatron-path /workspace/$MODEL-mimo \
      --component language=tp=4,dp=1,rank_offset=0 --component images=tp=1,dp=1,rank_offset=4 \
      --torch-dtype bfloat16" 2>&1 | tee "$LOGS/convert-mimo.log" ;;
  fit)
    container "python -m torch.distributed.run --nproc_per_node=5 \
      examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py --hf-model Qwen/$MODEL --dataset-name cord_v2 \
      --seq-length 2048 --global-batch-size 64 --train-iters ${STEPS:-3} --dataloader-type single \
      --pretrained-checkpoint /workspace/$MODEL-mimo \
      --lr 1e-5 --min-lr 1e-5 --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.0 \
      --start-weight-decay 0.0 --end-weight-decay 0.0 --clip-grad 0.0 --lr-warmup-iters 0 \
      --log-interval 1 --experiment-root /workspace/runs --micro-batch-size 1 --run-name fit-mimo \
      --component language=tp=4,pp=1,dp=1,rank_offset=0 --component images=tp=1,pp=1,dp=1,rank_offset=4" \
      2>&1 | tee "$LOGS/fit-mimo${NAME:-}.log" ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
