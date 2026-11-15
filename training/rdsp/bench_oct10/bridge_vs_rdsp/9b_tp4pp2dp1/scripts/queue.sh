#!/bin/bash
# On the H100 box: 9B checkpoint + samples, 3-step preflights, then the unprofiled 50-step runs.
set -x
cd ~/rdsp
R=bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/scripts/run.sh
L=bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/logs; mkdir -p $L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $L/hardware.txt
bash $R prep
bash $R preflight
grep -q "iteration        3/       3" $L/preflight-bridge.log || { echo "STOP: bridge preflight failed"; exit 1; }
grep -q "^step 2 loss" $L/preflight-rdsp.log || { echo "STOP: rdsp preflight failed"; exit 1; }
echo "PREFLIGHT OK"
bash $R bridge
bash $R rdsp
echo "RUNS DONE"
