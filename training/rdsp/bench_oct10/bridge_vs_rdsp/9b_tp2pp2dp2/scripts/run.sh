#!/bin/bash
# run.sh <step>: one step of the 9B TP2xPP2xDP2 comparison on the H100 box, after
# bench/setup_node.sh and the 9B checkpoint + export (9b_tp4pp2dp1/scripts/run.sh prep).
#
#   bridge   Megatron-Bridge, 50 steps
#   rdsp     rdsp, 50 steps
#
# Megatron-Bridge pads every row to 2048 itself whenever PP > 1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../../../.." && pwd)
export MODEL=Qwen3.5-9B RECIPE=qwen35_vl_9b_sft_4gpu_h100_bf16_config
export LOGS=$HERE/../logs; mkdir -p "$LOGS"
RUNS=$RDSP/bench/mimo/runs.sh
DATA=exported:$HOME/mimo/cord_steps-$MODEL

case "${1:-}" in
  bridge)
    MEGATRON_EXTRA="" bash "$RUNS" shared-megatron
    mv "$LOGS/shared-megatron.log" "$LOGS/bridge.log" ;;
  rdsp)
    (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
     env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
         DATA=$DATA bash "$HERE/rdsp.sh") 2>&1 | tee "$LOGS/rdsp.log" ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
