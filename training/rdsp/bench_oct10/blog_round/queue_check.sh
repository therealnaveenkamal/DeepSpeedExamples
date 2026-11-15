#!/bin/bash
# queue_check.sh, on the node: waits for queue_profile2.sh, then repeats the 9B gradient-add trial
# in short runs (26 steps): baseline once, direct add twice. Why: on 2026-11-11 the first direct-add
# run was 8% faster but its loss ran above the baseline from step 4 (up to +0.025 at step 10), where
# three unchanged runs agree within 0.002, and its steps 4-7 were slow. The add itself is
# bit-identical in isolation, so this tells a one-off from an effect of the switch.
# Last marker: "=== ALL DONE CHECK".
set -uo pipefail
cd ~/rdsp
M=~/rdsp/bench_oct10/mimo_vs_rdsp
L9=9b_v1_tp2pp1dp2
TAG=${TAG:-nov11}
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  pkill -f "[t]rain_vl.py"; bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1; sleep 10
}
until grep -qE "^=== (ALL DONE BLOG2|FAILED no py-spy)" ~/queue_profile2.out 2>/dev/null; do sleep 10; done
mark "CHECK START"
# First the Megatron-Bridge 27B py-spy run, which failed twice (py-spy missing, then a script bug)
P8=~/rdsp/bench_oct10/bridge_vs_rdsp/27b_tp1pp8dp1
CUT="Et|t*9|t*10|t*9|t*9|t*10|t*9|t*7L"
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
mark "START bridge-27b-pyspy"
$SMI > "$P8/logs/nvsmi-pyspy-bridge-$TAG.csv" &
smi=$!
rm -f "$P8"/profiles/pyspy-bridge-"$TAG"-*
timeout 1500 env NAME="-$TAG" MEG_RC="$RC_ALL \"model.pipeline_model_parallel_layout='$CUT'\" $FAST model.batch_p2p_sync=false" \
  bash "$P8/scripts/run.sh" pyspy; rc=$?
kill $smi
sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1; sudo pkill -f "[p]y-spy record"; sleep 10
mark "END bridge-27b-pyspy exit $rc; stack files $(ls "$P8"/profiles/pyspy-bridge-"$TAG"-*.txt 2>/dev/null | wc -l)"
cat "$P8/profiles/pyspy-bridge-$TAG-ranks.txt" 2>/dev/null
one() {  # one <log suffix> [env assignment]
  mark "START rdsp-9b-$1"
  $SMI > "$M/$L9/logs/nvsmi-rdsp-$TAG-$1.csv" &
  local smi=$!
  timeout 900 env ${2:-RDSP_NOOP=1} bash "$M/scripts/run.sh" rdsp $L9 26; local rc=$?
  kill $smi; cleanup
  mv "$M/$L9/logs/rdsp.log" "$M/$L9/logs/rdsp-$TAG-$1.log"
  mark "END rdsp-9b-$1 exit $rc; direct-add lines $(grep -c "direct gradient add on" "$M/$L9/logs/rdsp-$TAG-$1.log")"
  grep -a "^step " "$M/$L9/logs/rdsp-$TAG-$1.log" | awk '{printf "%s:%s/%s ", $2, $4, $5} END {print ""}'
}
one base2
one directadd2 RDSP_DIRECT_GRAD_ADD=1
one directadd3 RDSP_DIRECT_GRAD_ADD=1
mark "ALL DONE CHECK"
