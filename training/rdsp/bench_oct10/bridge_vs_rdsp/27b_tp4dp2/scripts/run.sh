#!/bin/bash
# run.sh <step>: 27B TP4 x DP2 (no pipeline) on the H100 box, after bench/setup_node.sh and the
# 27B checkpoint + export (27b_tp4pp2dp1/scripts/run.sh convert / export).
#
#   bridge   Megatron-Bridge, 50 steps
#   rdsp     rdsp, 50 steps
#
# Both recompute every layer and accumulate gradients in bf16 (fp32 master weights and Adam),
# as in the TP4xPP2 runs. Without a pipeline Megatron-Bridge does not pad to 2048 on its
# own: dataset.pad_to_max_length=true makes the work equal to rdsp's. STEPS (default 50).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../../../.." && pwd)
export MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config
export LOGS=$HERE/../logs; mkdir -p "$LOGS"
RUNS=$RDSP/bench/mimo/runs.sh
DATA=exported:$HOME/mimo/cord_steps-$MODEL
# MEG_RC: Bridge's recompute overrides (default: every layer); RECOMPUTE=0: none
MEG_RC=${MEG_RC:-"model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"}
[ "${RECOMPUTE:-1}" = 0 ] && MEG_RC=""
MEG_BF16=$(cat "$RDSP/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/meg_extra_bf16.txt")

case "${1:-}" in
  bridge)
    GRAD_REDUCE_FP32=false MEGATRON_EXTRA="dataset.pad_to_max_length=true $MEG_RC $MEG_BF16" \
      bash "$RUNS" tp4dp2-megatron || rc=$?
    mv "$LOGS/tp4dp2-megatron.log" "$LOGS/bridge${NAME:-}.log"  # kept when the run fails
    exit "${rc:-0}" ;;
  rdsp)
    (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
     env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
         DATA=$DATA bash "$HERE/rdsp.sh" ${RDSP_ARGS:-}) 2>&1 | tee "$LOGS/rdsp${NAME:-}.log" ;;
  *) sed -n 2,11p "$0"; exit 1 ;;
esac
