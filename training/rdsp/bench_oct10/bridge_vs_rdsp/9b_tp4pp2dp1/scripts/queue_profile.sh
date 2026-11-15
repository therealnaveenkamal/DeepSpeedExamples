#!/bin/bash
# On the H100 box, after queue.sh: diagnosis runs (not timing results).
# 1) per-phase timers + GPU utilization, 2) Nsight Systems window on steady steps.
# Set NSYS_DELAY_BRIDGE, NSYS_DELAY_RDSP and NSYS_DURATION (seconds) from the
# unprofiled logs' load and warm-up times before starting.
set -x
cd ~/rdsp
R=bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/scripts/run.sh
P=bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/profiles
until grep -q "RUNS DONE" ~/queue.out; do sleep 20; done
bash $R timers
NSYS_DELAY_BRIDGE=${NSYS_DELAY_BRIDGE:?} NSYS_DELAY_RDSP=${NSYS_DELAY_RDSP:?} bash $R nsys
. ~/venv/bin/activate
for r in bridge rdsp; do
  [ -f $P/$r.nsys-rep ] && nsys export --type sqlite --force-overwrite true -o $P/$r.sqlite $P/$r.nsys-rep
done
echo "PROFILE DONE"
