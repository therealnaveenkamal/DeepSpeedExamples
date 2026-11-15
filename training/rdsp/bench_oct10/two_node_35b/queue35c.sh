#!/bin/bash
# On node 1 after queue35b.sh: Bridge TP4xDP4 without expert parallelism, with its fp32 master
# (tp4dp4/run.sh MASTER=fp32: the 2-byte master hits Transformer Engine's tensor limit here).
set -x
cd ~/rdsp
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
run() {  # run <name> <timeout minutes> <command...>
  local name=$1 minutes=$2; shift 2
  mark "START $name"
  $SMI > "$T/tp4dp4/logs/nvsmi-$name-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$T/tp4dp4/logs/nvsmi-$name-node2.csv" &
  local smi2=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  bash bench/two_node/both.sh --cleanup
  mark "END $name exit $rc"
}
run tp4dp4-bridge-fp32 80 env MASTER=fp32 bash "$T/tp4dp4/run.sh" bridge
L=$T/tp4dp4/logs/bridge-fp32.log
if grep -qaE "OutOfMemoryError|CUDA out of memory" "$L"; then
  mark "NOFIT tp4dp4-bridge-fp32: again with every block recomputed"
  run tp4dp4-bridge-fp32-rc 80 env MASTER=fp32 RECOMPUTE=1 bash "$T/tp4dp4/run.sh" bridge
fi
mark "ALL DONE"
