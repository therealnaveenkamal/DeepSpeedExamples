#!/bin/bash
# run.sh bridge: Qwen3.5-35B-A3B on Bridge's ground without expert parallelism, as the 27B runs
# had it (TP4 x DP2 on one node; the 27B two-node plan's TP4 x DP4): TP4 x DP4, no pipeline, the
# experts split over the 4 TP ranks like every other layer (expert TP 4, EP 1; the recipe's
# expert TP 1 / EP 16 overridden). 32 microbatches of 4 rows (one per data-parallel copy).
# rdsp has no run here: its tensor parallelism leaves the routed experts whole, so each GPU would
# hold all 32B of them. On node 1, after ../scripts/prep.sh.
# MASTER=fp32: Bridge's standard fp32 master in its distributed optimizer, not the 2-byte one
# (precision-aware optimizer): with every rank holding all 256 experts of all 40 layers (no EP),
# that optimizer's fused step needs more tensor handles than Transformer Engine allows ("Maximum
# number of tensors reached: 20321", 2026-10-18). Gradients stay bf16.
# STEPS (default 50), RECOMPUTE=1 (every block), NAME (log suffix). Logs in logs/.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../scripts/common.sh"
export LOGS=$HERE/logs$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc
case "${1:-}" in
  bridge)
    fp32=""
    [ "${MASTER:-}" = fp32 ] && fp32="optimizer.use_precision_aware_optimizer=false \
      optimizer.store_param_remainders=false optimizer.main_grads_dtype=fp32"
    bridge tp4dp2-megatron "bridge$TAG${MASTER:+-$MASTER}${NAME:-}" \
      model.expert_model_parallel_size=1 model.expert_tensor_parallel_size=4 $fp32 ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
