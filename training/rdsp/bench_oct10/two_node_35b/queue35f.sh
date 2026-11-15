#!/bin/bash
# On node 1: Bridge's 35B checkpoint again (removed for disk room before the 27B runs; prep.sh
# converts it and copies it to node 2, skipping what is in place), then queue35e.sh (Bridge PP8 x
# DP2, each pipeline inside one node). The 27B's exported rows go first: its runs are done.
cd ~/rdsp
rm -rf ~/mimo/cord_steps-Qwen3.5-27B
bash ~/rdsp/bench_oct10/two_node_35b/scripts/prep.sh || exit 1
bash ~/rdsp/bench_oct10/two_node_35b/queue35e.sh
