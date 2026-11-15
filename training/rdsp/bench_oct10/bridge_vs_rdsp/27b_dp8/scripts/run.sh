#!/bin/bash
# run.sh rdsp: 27B data parallel 8, ZeRO-3, on the H100 box, after bench/setup_node.sh and the 27B
# export (27b_tp4pp2dp1/scripts/run.sh export). rdsp only: Megatron-Bridge's recipe has no fully
# sharded mode here; TorchTitan's FSDP over 8 GPUs is the comparison (torchtitan/scripts/run.sh).
# STEPS (default 50), NAME (log suffix), RDSP_ARGS (more train_vl.py flags).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$HERE/../logs; mkdir -p "$LOGS"
DATA=exported:$HOME/mimo/cord_steps-Qwen3.5-27B
case "${1:-}" in
  rdsp)
    (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
     env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
         DATA=$DATA bash "$HERE/rdsp.sh" ${RDSP_ARGS:-}) 2>&1 | tee "$LOGS/rdsp${NAME:-}.log" ;;
  bridge)  # Megatron-Bridge with Megatron FSDP (weights, gradients and optimizer state split 8 ways)
    RDSP=$(cd "$HERE/../../../.." && pwd)
    MEG_BF16=$(cat "$RDSP/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/meg_extra_bf16.txt")
    export MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config LOGS
    GRAD_REDUCE_FP32=false MEGATRON_EXTRA="model.recompute_granularity=full model.recompute_method=uniform \
      model.recompute_num_layers=1 $MEG_BF16 ddp.use_megatron_fsdp=true \
      ddp.data_parallel_sharding_strategy=optim_grads_params dist.distributed_timeout_minutes=60 ${EXTRA:-}" \
      bash "$RDSP/bench/mimo/runs.sh" dp8-megatron || rc=$?
    mv "$LOGS/dp8-megatron.log" "$LOGS/bridge${NAME:-}.log"
    exit "${rc:-0}" ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac
