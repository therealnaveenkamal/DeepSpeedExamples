#!/bin/bash
# On node 1, after queue35f.sh (Bridge PP8 x DP2, pipelines inside each node), in order:
#   1. MegatronMIMO checkpoint of the dense 27B, language over 4 GPUs (on one GPU it ran out of
#      memory), copied to node 2; MIMO on both nodes: language TP4 x PP2 on node 1, images DP8
#      on node 2 (runs.sh twonode-mimo). MIMO cannot run the 35B-A3B (results.md).
#   2. Bridge TP1 x PP8 x DP2 on the 35B with the pipeline across the nodes (rdsp's placement),
#      the host's aws-ofi-nccl 1.21.1 mounted (HOST_OFI=1) and NCCL_P2P_PXN_LEVEL=0.
#   3. Bridge's own recipe (TP2 x EP16) again with the container's own library path.
set -x
cd ~/rdsp
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
P=$T/pp8dp2
M=$T/mimo27b
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
run() {  # run <name> <timeout minutes> <nvidia-smi csv prefix> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  $SMI > "$csv-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$csv-node2.csv" &
  local smi2=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  bash bench/two_node/both.sh --cleanup
  mark "END $name exit $rc"
}
mkdir -p "$M/logs"
while pgrep -f "[b]ash queue35f.sh" > /dev/null; do sleep 30; done

# ---- 1. MIMO, dense 27B
rm -rf ~/mimo/Qwen3.5-27B-mimo
mark "START convert-mimo-27b"
env MODEL=Qwen3.5-27B CONVERT_TP=4 LOGS="$M/logs" bash bench/mimo/runs.sh convert-mimo > /dev/null 2>&1
mark "END convert-mimo-27b"
if grep -qaE "Traceback|ChildFailedError" "$M/logs/convert-mimo.log" || [ ! -d ~/mimo/Qwen3.5-27B-mimo ]; then
  mark "FAILED MIMO 27B conversion"
else
  $SSH2 "rm -rf ~/mimo/Qwen3.5-27B-mimo; mkdir -p ~/mimo" < /dev/null
  rsync -a -e "ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no" ~/mimo/Qwen3.5-27B-mimo/ \
    "ubuntu@$PRIV2:mimo/Qwen3.5-27B-mimo/"
  run mimo27b 80 "$M/logs/nvsmi-mimo27b" bash bench/two_node/both.sh env MODEL=Qwen3.5-27B \
    GBS=128 STEPS=50 LOGS="$M/logs" bash bench/mimo/runs.sh twonode-mimo
fi

# ---- 2. Bridge PP8 x DP2 across the nodes, plugin and PXN fix
run pp8dp2-bridge-xnode-fix 60 "$P/logs/nvsmi-bridge-xnode-fix" env NAME=-xnode-fix-cut1-3 \
  CUT="1 3" HOST_OFI=1 NCCL_P2P_PXN_LEVEL=0 NCCL_DEBUG=INFO bash "$P/run.sh" bridge
grep -aoE "Initializing aws-ofi-nccl [0-9.]+|NET/OFI Selected provider is [a-z]+" \
  "$P/logs/bridge-xnode-fix-cut1-3.log" | sort | uniq -c

# ---- 3. Bridge's own recipe, clean flags
run tp2ep16-bridge-fixed 60 "$T/tp2ep16/logs/nvsmi-bridge-fixed" env NAME=-fixed \
  bash "$T/tp2ep16/run.sh" bridge
mark "ALL DONE"
