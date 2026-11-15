#!/bin/bash
# On the H100 box, after queue_27b_all.sh ("=== ALL DONE"): the PP-against-DP runs, everything
# recomputed so the layouts differ only in how the 8 GPUs are split (plan_27b_50step.md).
#   A-rc  TP4 x PP2 + recompute (against C, TP4 x DP2 + recompute: same TP, PP2 against DP2)
#   D     TP2 x PP2 x DP2 + recompute
#   E     TP2 x PP4 + recompute
# Each run: Bridge and rdsp, 50 steps, timeout, nvidia-smi samples, "=== END <run> exit <code>".
set -x
while ! grep -q "^=== ALL DONE" ~/queue_27b_all.out 2>/dev/null; do sleep 30; done
B=~/rdsp/bench_oct10/bridge_vs_rdsp
A=$B/27b_tp4pp2dp1
D=$B/27b_tp2pp2dp2
E=$B/27b_tp2pp4dp1
G="--optimizer-group-numel 250000000"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  (. ~/venv/bin/activate && ray stop --force >/dev/null 2>&1)
  sudo docker ps -q | xargs -r sudo docker kill >/dev/null 2>&1
  sleep 10
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > "$csv" &
  local smi=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi; cleanup
  mark "END $name exit $rc"
}

run A1-rc 60 $A/logs/nvsmi-A1rc-bridge-50.csv env RECOMPUTE=1 GRAD_REDUCE_FP32=false NAME=-bf16grads-50 \
  EXTRA="$(cat $A/scripts/meg_extra_bf16.txt)" bash $A/scripts/run.sh bridge
run A2-rc 55 $A/logs/nvsmi-A2rc-rdsp-50.csv env RECOMPUTE=1 NAME=-bf16-groups-50 \
  RDSP_ARGS="--grad-dtype bf16 --fused-adam $G" bash $A/scripts/run.sh rdsp
run D2 55 $D/logs/nvsmi-D2-rdsp-50.csv env NAME=-50 bash $D/scripts/run.sh rdsp
run D1 60 $D/logs/nvsmi-D1-bridge-50.csv env NAME=-50 bash $D/scripts/run.sh bridge
run E2 55 $E/logs/nvsmi-E2-rdsp-50.csv env NAME=-50 bash $E/scripts/run.sh rdsp
run E1 60 $E/logs/nvsmi-E1-bridge-50.csv env NAME=-50 bash $E/scripts/run.sh bridge
mark "MORE DONE"
