#!/bin/bash
# On the H100 box: (1) rdsp vision-1 with recompute on the language stages, (2) the MegatronMIMO
# checkpoint, (3) MIMO vision-1 3-step fit check (no recompute: its script has no such option).
# Each run under a 15-minute timeout (a stage that runs out of memory can leave the rest waiting).
set -x
D=~/rdsp/bench_oct10/mimo_vs_rdsp/27b_v1
smi() { nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > $D/logs/nvsmi-$1.csv & }
peaks() { awk -F', ' '{if ($3 > m[$2]) m[$2] = $3} END {for (g in m) print "GPU", g, "max used MiB", m[g]}' $D/logs/nvsmi-$1.csv | sort -k2n; }
cleanup() { (. ~/venv/bin/activate && ray stop --force >/dev/null 2>&1); sudo docker ps -q | xargs -r sudo docker kill >/dev/null; }

smi rdsp-fit-rc; SMI=$!
RECOMPUTE=1 NAME=-rc timeout 900 bash $D/scripts/rdsp_fit.sh; echo "rdsp exit $?"
kill $SMI; cleanup
grep -E "rdsp_profile|^step|OutOfMemory" $D/logs/fit-rdsp-rc.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240 | tail -25
peaks rdsp-fit-rc
echo "RDSP RC DONE"

[ -d ~/mimo/Qwen3.5-27B-mimo ] || timeout 1800 bash $D/scripts/mimo_fit.sh convert
echo "MIMO CONVERT DONE"
smi mimo-fit; SMI=$!
timeout 900 bash $D/scripts/mimo_fit.sh fit; echo "mimo exit $?"
kill $SMI; cleanup
grep -E "iteration +[0-9]+/|OutOfMemory|memory \(GB\)|Error" $D/logs/fit-mimo.log | grep -v "Triton kernels" | cut -c1-400 | tail -20
peaks mimo-fit
echo "QUEUE DONE"
