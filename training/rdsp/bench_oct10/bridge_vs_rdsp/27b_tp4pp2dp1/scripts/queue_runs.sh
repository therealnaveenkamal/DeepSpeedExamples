#!/bin/bash
# On the H100 box: 3-step preflights with recompute, then the unprofiled 50-step runs.
set -x
cd ~/rdsp
R=bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/run.sh
L=bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/logs
bash $R preflight
grep -q "iteration        3/       3" $L/preflight-bridge-rc.log || { echo "STOP: bridge preflight failed"; exit 1; }
grep -q "^step 2 loss" $L/preflight-rdsp-rc.log || { echo "STOP: rdsp preflight failed"; exit 1; }
echo "PREFLIGHT OK"
bash $R bridge
bash $R rdsp
bash $R rdsp-own
echo "RUNS DONE"
