#!/bin/bash
# On the H100 box: environment, hardware record, 27B checkpoint + samples, 3-step preflights.
set -x
cd ~/rdsp
L=~/rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/logs; mkdir -p $L
bash bench/setup_node.sh > ~/setup.log 2>&1 || { echo "SETUP FAILED"; exit 1; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi topo -m; lscpu | head -20; } > $L/hardware.txt 2>&1
bash bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/run.sh prep
bash bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/run.sh preflight
echo "PREP DONE"
