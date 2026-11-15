#!/bin/bash
# On the H100 box, after queue_mimo.sh: rdsp with optimizer groups of at most 250M elements
# (--optimizer-group-numel), (1) vision-1 + recompute on the language stages, (2) 27B TP4xPP2
# with bf16 gradients + fused AdamW, no recompute. 3 steps each, nvidia-smi every 0.5 s.
set -x
while ! grep -q "QUEUE DONE" ~/queue_mimo.out 2>/dev/null; do sleep 10; done
D=~/rdsp/bench_oct10/mimo_vs_rdsp/27b_v1
T=~/rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1
G="--optimizer-group-numel 250000000"
cleanup() { (. ~/venv/bin/activate && ray stop --force >/dev/null 2>&1); }
peaks() { awk -F', ' '{if ($3 > m[$2]) m[$2] = $3} END {for (g in m) print "GPU", g, "max used MiB", m[g]}' $1 | sort -k2n; }

nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > $D/logs/nvsmi-rdsp-fit-rc-groups.csv & SMI=$!
RECOMPUTE=1 NAME=-rc-groups EXTRA="$G" timeout 900 bash $D/scripts/rdsp_fit.sh; echo "rdsp v1 exit $?"
kill $SMI; cleanup
grep -E "rdsp_profile|^step|OutOfMemory|StepFailed" $D/logs/fit-rdsp-rc-groups.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240 | tail -16
peaks $D/logs/nvsmi-rdsp-fit-rc-groups.csv
echo "V1 DONE"

nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > $T/logs/nvsmi-rdsp-bf16-groups.csv & SMI=$!
RECOMPUTE=0 COMPILE_VISION=0 NAME=-bf16grads-fused-groups EXTRA="--grad-dtype bf16 --fused-adam $G" \
  timeout 900 bash $T/scripts/run.sh preflight-rdsp; echo "rdsp tp4pp2 exit $?"
kill $SMI; cleanup
grep -E "rdsp_profile|^step|OutOfMemory|StepFailed" $T/logs/preflight-rdsp-bf16grads-fused-groups.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240 | tail -16
peaks $T/logs/nvsmi-rdsp-bf16-groups.csv
echo "QUEUE DONE"
