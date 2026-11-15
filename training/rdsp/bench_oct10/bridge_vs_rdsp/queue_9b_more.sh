#!/bin/bash
# On the H100 box: 9B TP2xPP2xDP2, then 9B TP2xPP1xDP4, Megatron-Bridge then rdsp,
# 50 unprofiled steps each. A failed run is logged and the queue moves on.
set -x
cd ~/rdsp/bench_oct10/bridge_vs_rdsp
for L in 9b_tp2pp2dp2 9b_tp2pp1dp4; do
  for s in bridge rdsp; do
    bash $L/scripts/run.sh $s || echo "RUN FAILED: $L $s"
    echo "RUN END: $L $s"
  done
done
echo "QUEUE DONE"
