#!/bin/bash
# On node 1: Bridge TP1 x PP8 x DP2 on the 35B-A3B with each 8-stage pipeline inside one node and
# data parallelism across the nodes (dist.use_tp_pp_dp_mapping=true; Megatron's default order put
# the pipeline across the nodes, which deadlocked twice in batched send/recv), at the cut nearest
# rdsp's measured one (1 | 6x6 | 3). One that runs out of memory runs again with every layer
# recomputed. Markers for ops/h100/guard.sh.
set -x
cd ~/rdsp
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
P=$T/pp8dp2
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
run() {  # run <name> <timeout minutes> <command...>
  local name=$1 minutes=$2; shift 2
  mark "START $name"
  $SMI > "$P/logs/nvsmi-$name-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$P/logs/nvsmi-$name-node2.csv" &
  local smi2=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  bash bench/two_node/both.sh --cleanup
  mark "END $name exit $rc"
}
X="dist.use_tp_pp_dp_mapping=true"
run pp8dp2-bridge-ppdp 80 env NAME=-ppdp-cut1-3 CUT="1 3" BRIDGE_EXTRA="$X" bash "$P/run.sh" bridge
L=$P/logs/bridge-ppdp-cut1-3.log
if grep -qaE "OutOfMemoryError|CUDA out of memory" "$L"; then
  mark "NOFIT pp8dp2-bridge-ppdp: again with every layer recomputed"
  run pp8dp2-bridge-ppdp-rc 80 env RECOMPUTE=1 NAME=-ppdp-cut1-3 CUT="1 3" BRIDGE_EXTRA="$X" \
    bash "$P/run.sh" bridge
fi
mark "ALL DONE"
