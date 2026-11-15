#!/bin/bash
# prep.sh, on the node after ops/h100/push.sh (run it with ops/h100/remote.sh): one p5.48xlarge
# ready for queue_profile.sh (plan.md).
#   1. bench/setup_node.sh: environment, NeMo container
#   2. weights: Qwen3.5-27B and Qwen3.5-9B
#   3. 27B: Megatron-Bridge's checkpoint and its CORD-v2 rows at 64 a step
#   4. 9B: Megatron-Bridge's checkpoint and rows at 64 a step, MegatronMIMO's checkpoint
# Safe to rerun: finished parts are skipped. Markers: "--- START/END <part>", then
# "=== PREP DONE" or "=== PREP FAILED <what>".
set -uo pipefail
RDSP=~/rdsp
B=$RDSP/bench_oct10
LOGS=$B/blog_round/prep_logs; mkdir -p "$LOGS" ~/mimo/std
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
step() { echo "--- $* $(date -u +%H:%M:%S)"; }
fail() { mark "PREP FAILED $*"; exit 1; }
CONTAINER="sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
  -v $HOME/hf:/root/.cache/huggingface -v $HOME/mimo:/workspace -v $RDSP:/rdsp \
  -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge nvcr.io/nvidia/nemo:26.08"

step "START setup"
if [ ! -f ~/setup.done ]; then
  bash "$RDSP/bench/setup_node.sh" > ~/setup.log 2>&1 || fail "setup (~/setup.log)"
  touch ~/setup.done
fi
{ nvidia-smi; nvidia-smi topo -m
  ~/venv/bin/python -c "import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"
} > "$LOGS/hardware.txt" 2>&1
grep -c "H100" "$LOGS/hardware.txt" | sed 's/^/H100 lines in nvidia-smi: /'
step "END setup"

step "START weights"
for M in Qwen3.5-27B Qwen3.5-9B; do
  bash -c "unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c \"import huggingface_hub as h; \
h.snapshot_download('Qwen/$M', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])\"" \
    > "$LOGS/download-$M.log" 2>&1 || fail "download $M"
done
step "END weights"

step "START 27B convert and export"
MODEL=Qwen3.5-27B
if [ ! -f "$HOME/mimo/std/$MODEL.done" ]; then
  $CONTAINER bash -c "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
    --megatron-path /workspace/std/$MODEL" > "$LOGS/convert27b.log" 2>&1; rc=$?
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo"
  [ $rc = 0 ] || fail "27B convert exit $rc"
  touch "$HOME/mimo/std/$MODEL.done"
fi
if [ ! -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ]; then
  env MODEL=$MODEL RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config GBS=64 STEPS=50 LOGS="$LOGS" \
    bash "$RDSP/bench/mimo/runs.sh" export > /dev/null 2>&1; rc=$?
  mv "$LOGS/export.log" "$LOGS/export27b.log" 2>/dev/null
  [ $rc = 0 ] && [ -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ] || fail "27B export exit $rc"
fi
cat "$HOME/mimo/cord_steps-$MODEL/stats.json"; echo   # 65,003 real tokens a step on 2026-10-17
step "END 27B convert and export"

step "START 9B convert and export"
MODEL=Qwen3.5-9B
if [ ! -f "$HOME/mimo/std/$MODEL.done" ] || [ ! -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ]; then
  GBS=64 STEPS=50 bash "$B/bridge_vs_rdsp/9b_tp4pp2dp1/scripts/run.sh" prep > "$LOGS/prep9b.log" 2>&1; rc=$?
  [ $rc = 0 ] && [ -f "$HOME/mimo/cord_steps-$MODEL/stats.json" ] || fail "9B Bridge checkpoint or rows exit $rc"
  touch "$HOME/mimo/std/$MODEL.done"
fi
cat "$HOME/mimo/cord_steps-$MODEL/stats.json"; echo
if [ ! -d "$HOME/mimo/$MODEL-mimo" ]; then
  bash "$B/mimo_vs_rdsp/scripts/run.sh" convert > "$LOGS/convert9b-mimo.log" 2>&1 || fail "9B MIMO checkpoint"
fi
ls "$HOME/mimo/$MODEL-mimo" > /dev/null || fail "9B MIMO checkpoint missing"
step "END 9B convert and export"
df -h ~ | tail -1
mark "PREP DONE"
