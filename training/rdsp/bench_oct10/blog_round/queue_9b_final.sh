#!/bin/bash
# queue_9b_final.sh, on the node (started with ops/h100/remote.sh, never after a push): the two 9B
# MIMO-comparison layouts with repeats on one machine (approved 2026-11-11, "extend and run").
#   A = 9b_v1_tp2pp2dp1 (1 vision GPU + language TP2 x PP2)
#   B = 9b_v1_tp2pp1dp2 (1 vision GPU + language TP2 x DP2); B already has mimo-nov11.log and
#       rdsp-nov11-directadd.log (50 steps each) from queue_profile.sh
# MIMO runs unchanged. rdsp runs with the environment in ~/rdsp_env.txt (one line, read at each
# run; default RDSP_DIRECT_GRAD_ADD=1), so it can be changed without restarting the queue.
# The order comes from ~/9b_order.txt if it exists (one "<layout letter> <mimo|fix|base> <n>" a line).
# Logs: <layout>/logs/{mimo-nov11-rN,rdsp-nov11-fix-rN,rdsp-nov11-base-rN}.log. Last marker
# "=== ALL DONE 9B FINAL".
set -uo pipefail
cd ~/rdsp
M=~/rdsp/bench_oct10/mimo_vs_rdsp
TAG=nov11
MED="python3 $HOME/rdsp/bench_oct10/gap_round/scripts/medians.py"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
[ -f ~/rdsp_env.txt ] || echo "RDSP_DIRECT_GRAD_ADD=1" > ~/rdsp_env.txt
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]rain_vl.py"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1
  sleep 10
}
one() {  # one <A|B> <mimo|fix|base> <n>
  local L sys=$2 n=$3 name kind src rc
  case $1 in A) L=9b_v1_tp2pp2dp1 ;; B) L=9b_v1_tp2pp1dp2 ;; *) return 1 ;; esac
  case $sys in
    mimo) name=mimo-$TAG-r$n; kind=bridge; src=mimo.log ;;
    fix)  name=rdsp-$TAG-fix-r$n; kind=rdsp; src=rdsp.log ;;
    base) name=rdsp-$TAG-base-r$n; kind=rdsp; src=rdsp.log ;;
    *) return 1 ;;
  esac
  mkdir -p "$M/$L/logs"
  mark "START $1 $name"
  $SMI > "$M/$L/logs/nvsmi-$name.csv" &
  local smi=$!
  case $sys in
    mimo) timeout 1500 bash "$M/scripts/run.sh" mimo $L; rc=$? ;;
    fix)  timeout 1200 env $(cat ~/rdsp_env.txt) bash "$M/scripts/run.sh" rdsp $L; rc=$? ;;
    base) timeout 1200 bash "$M/scripts/run.sh" rdsp $L; rc=$? ;;
  esac
  kill $smi; cleanup
  mv "$M/$L/logs/$src" "$M/$L/logs/$name.log"
  mark "END $1 $name exit $rc median $($MED $kind "$M/$L/logs/$name.log" | awk '{print $1}') s env [$([ $sys = fix ] && cat ~/rdsp_env.txt)] direct-add lines $(grep -c "direct gradient add on" "$M/$L/logs/$name.log")"
}
[ -f ~/9b_order.txt ] || cat > ~/9b_order.txt <<'ORDER'
A mimo 1
A fix 1
B mimo 2
B fix 2
A mimo 2
A fix 2
A mimo 3
A fix 3
A base 1
ORDER
until grep -q "^=== ALL DONE PROBE" ~/queue_wait_probe.out 2>/dev/null; do sleep 5; done  # the timing probe first
mark "9B FINAL START"
i=1
while :; do   # the order file is re-read before every run, so later lines can be edited
  line=$(sed -n "${i}p" ~/9b_order.txt)
  [ -n "$line" ] || break
  one $line
  i=$((i + 1))
done
mark "ALL DONE 9B FINAL"
