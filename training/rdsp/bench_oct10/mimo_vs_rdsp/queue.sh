#!/bin/bash
# On the box: MIMO checkpoint, layouts a and b, the vision sweep, then c and d with
# MIMO's fastest vision layout. 50 steps per full run, no profiler. A failed run is
# logged and the queue moves on.
set -x
D=~/rdsp/bench_oct10/mimo_vs_rdsp; R="bash $D/scripts/run.sh"
[ -d ~/mimo/Qwen3.5-9B-mimo ] || $R convert
for L in 9b_v1_tp2pp2dp1 9b_v1_tp2pp1dp2; do
  for s in mimo rdsp; do $R $s $L || echo "RUN FAILED: $L $s"; echo "RUN END: $L $s"; done
done
for L in 9b_vtp2_tp2pp2dp1 9b_vtp4_tp2pp2dp1 9b_vdp2_tp2pp1dp2 9b_vtp2_tp2pp1dp2 9b_vtp2dp2_tp2pp1dp2; do
  $R sweep $L; echo "SWEEP END: $L"
done
for lang in tp2pp2dp1 tp2pp1dp2; do
  best=$($R best $lang 2> $D/sweep/best_$lang.txt); cat $D/sweep/best_$lang.txt
  echo "BEST $lang: $best"
  case $best in 9b_v1_*) echo "1-GPU vision is fastest for $lang: c/d = a/b"; continue ;; esac
  for s in mimo rdsp; do $R $s $best || echo "RUN FAILED: $best $s"; echo "RUN END: $best $s"; done
done
echo "QUEUE DONE"
