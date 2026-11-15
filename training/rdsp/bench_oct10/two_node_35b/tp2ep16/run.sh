#!/bin/bash
# run.sh <bridge|rdsp>: Qwen3.5-35B-A3B on Bridge's own two-node layout for it (plan.md runs 4-5):
# TP2 x DP8, no pipeline, the 256 experts split over all 16 GPUs (16 per GPU), every expert
# layer's tokens sent to their experts with all-to-all, across both nodes. 16 microbatches of
# 8 rows (one per data-parallel copy). On node 1, after ../scripts/prep.sh; rdsp needs the Ray
# cluster up.
#   bridge   qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config as shipped: TP2, EP16, sequence parallel
#   rdsp     one stage of 16 GPUs: AutoTP 2, DeepSpeed AutoEP 16 folded onto the TP ranks
# STEPS (default 50), RECOMPUTE=1 (every block), NAME (log suffix). Logs in logs/.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../scripts/common.sh"
export LOGS=$HERE/logs$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; RC=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc RC=",recompute=1"
case "${1:-}" in
  bridge)
    bridge tp2ep16-megatron "bridge$TAG${NAME:-}" model.expert_model_parallel_size=16 ;;
  rdsp)
    rdsp "rdsp$TAG${NAME:-}" --stages 1 --stage "0:gpus=16,tp=2,ep=16,fold=1,zero=1$RC" \
      --rows 8 --microbatches 16 ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
