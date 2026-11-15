#!/bin/bash
# On node 1, after queue35h.sh: Bridge's own recipe for the 35B-A3B (TP2 x EP16) again, now with
# the container's own library path (the first run had the multi-node flags that replaced it,
# dropping its CUDA forward-compatibility library): Bridge's best shipped number, clean.
set -x
cd ~/rdsp
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
while pgrep -f "[b]ash queue35h.sh" > /dev/null; do sleep 30; done
mark "START tp2ep16-bridge-fixed"
$SMI > "$T/tp2ep16/logs/nvsmi-bridge-fixed-node1.csv" &
S1=$!
$SSH2 "$SMI" < /dev/null > "$T/tp2ep16/logs/nvsmi-bridge-fixed-node2.csv" &
S2=$!
timeout $((60 * 60)) env NAME=-fixed bash "$T/tp2ep16/run.sh" bridge; rc=$?
kill $S1 $S2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
bash bench/two_node/both.sh --cleanup
mark "END tp2ep16-bridge-fixed exit $rc"
mark "ALL DONE"
