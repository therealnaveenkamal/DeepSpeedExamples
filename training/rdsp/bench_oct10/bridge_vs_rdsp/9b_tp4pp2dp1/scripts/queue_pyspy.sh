#!/bin/bash
# On the H100 box: install py-spy, then the untraced py-spy pass for both systems.
set -x
$HOME/.local/bin/uv pip install --python $HOME/venv/bin/python py-spy && $HOME/venv/bin/py-spy --version
cd ~/rdsp && bash bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/scripts/run.sh pyspy
ls -la bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/profiles/ | grep pyspy
echo "PYSPY DONE"
