#!/bin/bash
# On the H100 box: MegatronMIMO 27B checkpoint (TP4 layout), then the vision-1 3-step fit check.
set -x
D=~/rdsp/bench_oct10/mimo_vs_rdsp/27b_v1
timeout 1800 bash $D/scripts/mimo_fit.sh convert; echo "convert exit $?"
du -sh ~/mimo/Qwen3.5-27B-mimo
echo "MIMO CONVERT DONE"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > $D/logs/nvsmi-mimo-fit.csv &
SMI=$!
timeout 900 bash $D/scripts/mimo_fit.sh fit; echo "mimo exit $?"
kill $SMI; sudo docker ps -q | xargs -r sudo docker kill >/dev/null
grep -E "iteration +[0-9]+/|OutOfMemory|memory \(GB\)" $D/logs/fit-mimo.log | cut -c1-400 | tail -20
awk -F', ' '{if ($3 > m[$2]) m[$2] = $3} END {for (g in m) print "GPU", g, "max used MiB", m[g]}' $D/logs/nvsmi-mimo-fit.csv | sort -k2n
echo "QUEUE DONE"
