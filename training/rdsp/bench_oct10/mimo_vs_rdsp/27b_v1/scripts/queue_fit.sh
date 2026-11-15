#!/bin/bash
# On the H100 box: rdsp 27B vision-1 fit check (3 steps), with nvidia-smi sampling.
set -x
D=~/rdsp/bench_oct10/mimo_vs_rdsp/27b_v1
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > $D/logs/nvsmi-rdsp-fit.csv &
SMI=$!
bash $D/scripts/rdsp_fit.sh
kill $SMI
grep -E "rdsp_profile|^step|OutOfMemory|cuts|layers" $D/logs/fit-rdsp.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260 | tail -40
awk -F', ' '{if ($3 > m[$2]) m[$2] = $3} END {for (g in m) print "GPU", g, "max used MiB", m[g]}' $D/logs/nvsmi-rdsp-fit.csv | sort -k2n
echo "FIT DONE"
