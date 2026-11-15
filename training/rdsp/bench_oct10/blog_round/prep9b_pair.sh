#!/bin/bash
# prep9b_pair.sh, on node 1 of the two-node pair (ops/h100/remote2.sh): the 9B pieces for the
# one-node MIMO-comparison runs, on a node that prep2.sh already set up (2026-11-12, to try the
# run-ahead schedule on the 9B rows): 9B weights, Megatron-Bridge's 9B checkpoint and CORD-v2
# rows at 64 a step, MegatronMIMO's 9B checkpoint. Markers "--- START/END", "=== PREP9B DONE|FAILED".
set -uo pipefail
RDSP=~/rdsp; B=$RDSP/bench_oct10
LOGS=$B/blog_round/prep9b_logs; mkdir -p "$LOGS" ~/mimo/std
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
step() { echo "--- $* $(date -u +%H:%M:%S)"; }
bash $RDSP/bench/two_node/ray_cluster.sh down $(. ~/env2.sh; echo $PRIV1 $PRIV2) > /dev/null 2>&1
step "START 9B weights"
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-9B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > "$LOGS/download9b.log" 2>&1 || { mark "PREP9B FAILED download"; exit 1; }
step "END 9B weights"
step "START 9B convert and export"
MODEL=Qwen3.5-9B
if [ ! -f "$HOME/mimo/std/$MODEL.done" ] || [ ! -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ]; then
  GBS=64 STEPS=50 bash "$B/bridge_vs_rdsp/9b_tp4pp2dp1/scripts/run.sh" prep > "$LOGS/prep9b.log" 2>&1; rc=$?
  [ $rc = 0 ] && [ -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ] || { mark "PREP9B FAILED Bridge checkpoint or rows exit $rc"; exit 1; }
  touch "$HOME/mimo/std/$MODEL.done"
fi
cat "$HOME/mimo/cord_steps-$MODEL/stats.json"; echo
[ -d "$HOME/mimo/$MODEL-mimo" ] || bash "$B/mimo_vs_rdsp/scripts/run.sh" convert > "$LOGS/convert9b-mimo.log" 2>&1 \
  || { mark "PREP9B FAILED MIMO checkpoint"; exit 1; }
step "END 9B convert and export"
df -h ~ | tail -1
mark "PREP9B DONE"
