#!/bin/bash
# On node 1, after queue35g.sh (MIMO 27B): Bridge TP1 x PP8 x DP2 on the 35B-A3B with the pipeline
# across the nodes (Megatron's default rank order, rdsp's placement), which deadlocked twice in
# batched send/recv, now with the host's aws-ofi-nccl 1.21.1 mounted (HOST_OFI=1) and
# NCCL_P2P_PXN_LEVEL=0; NCCL INFO logs show the plugin version. Cut 1 | 6x6 | 3.
set -x
cd ~/rdsp
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
P=$T/pp8dp2
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
while pgrep -f "[b]ash queue35g.sh" > /dev/null; do sleep 30; done
mark "START pp8dp2-bridge-xnode-fix"
$SMI > "$P/logs/nvsmi-bridge-xnode-fix-node1.csv" &
S1=$!
$SSH2 "$SMI" < /dev/null > "$P/logs/nvsmi-bridge-xnode-fix-node2.csv" &
S2=$!
timeout $((60 * 60)) env NAME=-xnode-fix-cut1-3 CUT="1 3" HOST_OFI=1 NCCL_P2P_PXN_LEVEL=0 \
  NCCL_DEBUG=INFO bash "$P/run.sh" bridge; rc=$?
kill $S1 $S2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
bash bench/two_node/both.sh --cleanup
mark "END pp8dp2-bridge-xnode-fix exit $rc"
grep -aoE "Initializing aws-ofi-nccl [0-9.]+|NET/OFI Selected provider is [a-z]+" \
  "$P/logs/bridge-xnode-fix-cut1-3.log" | sort | uniq -c
mark "ALL DONE"
