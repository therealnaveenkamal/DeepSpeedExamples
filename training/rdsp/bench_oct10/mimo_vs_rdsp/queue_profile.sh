#!/bin/bash
# On the box, after queue.sh: profile layouts a and b (diagnosis runs, not results).
# py-spy first (untraced, also measures when each system is steady), then Nsight.
set -x
D=~/rdsp/bench_oct10/mimo_vs_rdsp; R="bash $D/scripts/run.sh"
for L in 9b_v1_tp2pp2dp1 9b_v1_tp2pp1dp2; do
  $R pyspy $L || echo "PYSPY FAILED: $L"; echo "PYSPY END: $L"
  $R nsys $L || echo "NSYS FAILED: $L"; echo "NSYS END: $L"
done
echo "PROFILE DONE"
