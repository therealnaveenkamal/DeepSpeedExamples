#!/bin/bash
# On node 1, after queue35f.sh (Bridge PP8 x DP2): MegatronMIMO on the dense Qwen3.5-27B over
# both nodes, its own regime (vision on GPUs of its own): language TP4 x PP2 on node 1, images
# DP8 on node 2, global microbatch 8 rows (16 a step of 128), 50 steps (runs.sh twonode-mimo).
# MIMO cannot run the 35B-A3B (no released container has its MoE checkpoint fix; its launcher has
# no expert parallelism). Its launcher has no recompute option: a fit check as much as a run.
set -x
cd ~/rdsp
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
M=$T/mimo27b
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
mkdir -p "$M/logs"
while pgrep -f "[b]ash queue35f.sh" > /dev/null; do sleep 30; done   # Bridge's run first
while pgrep -f "[c]onvert_megatron_mimo" > /dev/null; do sleep 30; done
[ -d ~/mimo/Qwen3.5-27B-mimo ] || { mark "FAILED MIMO 27B conversion"; exit 1; }
$SSH2 "mkdir -p ~/mimo" < /dev/null
rsync -a -e "ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no" ~/mimo/Qwen3.5-27B-mimo/ \
  "ubuntu@$PRIV2:mimo/Qwen3.5-27B-mimo/" || { mark "FAILED copy MIMO checkpoint"; exit 1; }
mark "START mimo27b"
$SMI > "$M/logs/nvsmi-mimo27b-node1.csv" &
S1=$!
$SSH2 "$SMI" < /dev/null > "$M/logs/nvsmi-mimo27b-node2.csv" &
S2=$!
timeout $((80 * 60)) bash bench/two_node/both.sh env MODEL=Qwen3.5-27B GBS=128 STEPS=50 \
  LOGS="$M/logs" bash bench/mimo/runs.sh twonode-mimo; rc=$?
kill $S1 $S2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
bash bench/two_node/both.sh --cleanup
mark "END mimo27b exit $rc"
mark "ALL DONE"
