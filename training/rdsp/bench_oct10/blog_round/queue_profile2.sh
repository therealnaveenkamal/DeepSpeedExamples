#!/bin/bash
# queue_profile2.sh, on the node after queue_profile.sh: its three profiling steps that failed on
# 2026-11-11 (py-spy was not installed on the node; the container's nsys could not read the
# host's report). Same tag, same cut as the 50-step Bridge run. Last marker: "=== ALL DONE BLOG2".
set -uo pipefail
cd ~/rdsp
B=~/rdsp/bench_oct10
P8=$B/bridge_vs_rdsp/27b_tp1pp8dp1
M=$B/mimo_vs_rdsp
L9=9b_v1_tp2pp1dp2
TAG=${TAG:-nov11}
CUT=${CUT:-"Et|t*9|t*10|t*9|t*9|t*10|t*9|t*7L"}
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]orchrun"; pkill -f "[t]rain_vl.py"; sudo pkill -f "[p]y-spy record"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1
  sleep 10
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  $SMI > "$csv" &
  local smi=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi
  cleanup
  mark "END $name exit $rc"
}
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
layout() { echo "\"model.pipeline_model_parallel_layout='$1'\""; }

mark "QUEUE2 START tag $TAG"
[ -x ~/venv/bin/py-spy ] || (export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
  uv pip install --python ~/venv/bin/python py-spy || { ~/venv/bin/python -m ensurepip && ~/venv/bin/python -m pip install py-spy; })
[ -x ~/venv/bin/py-spy ] || { mark "FAILED no py-spy"; exit 1; }
~/venv/bin/py-spy --version
# what the failed attempts left behind
rm -f "$P8"/profiles/pyspy-bridge-"$TAG"-* "$M/$L9/profiles-$TAG"/pyspy-* "$M/$L9/profiles-$TAG"/rdsp-bt.* \
      "$M/$L9/profiles-$TAG"/steady-*.s

run bridge-27b-pyspy 30 "$P8/logs/nvsmi-pyspy-bridge-$TAG.csv" env NAME="-$TAG" \
  MEG_RC="$RC_ALL $(layout "$CUT") $FAST model.batch_p2p_sync=false" bash "$P8/scripts/run.sh" pyspy
mark "BRIDGE PYSPY files: $(ls "$P8"/profiles/pyspy-bridge-"$TAG"-*.txt 2>/dev/null | wc -l)"
run pyspy-9b 40 "$M/$L9/logs/nvsmi-pyspy-$TAG.csv" env TAG="-$TAG" bash "$M/scripts/run.sh" pyspy $L9
mark "9B PYSPY files: $(ls "$M/$L9/profiles-$TAG"/pyspy-*.txt 2>/dev/null | wc -l)"
run bt-9b 25 "$M/$L9/logs/nvsmi-bt-$TAG.csv" env TAG="-$TAG" bash "$M/scripts/run.sh" backtrace $L9
ls -la "$M/$L9/profiles-$TAG"/rdsp-bt.* 2>/dev/null
mark "ALL DONE BLOG2"
