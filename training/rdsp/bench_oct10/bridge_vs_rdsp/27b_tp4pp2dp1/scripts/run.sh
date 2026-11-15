#!/bin/bash
# run.sh <step>: one step of the 27B TP4xPP2xDP1 comparison, on the H100 box,
# after bench/setup_node.sh. Logs go to ../logs (and ~/mimo for the container).
#
#   prep        HF -> Megatron checkpoint, then export Bridge's samples (pad 2048)
#   export      only export Bridge's samples
#   convert     only the HF -> Megatron checkpoint
#   preflight-bridge  Bridge only, 3 steps (EXTRA: overrides, NAME: log suffix)
#   preflight   3 steps of each system: fits in memory? (peak memory in the logs)
#   bridge      Megatron-Bridge, 50 steps
#   rdsp        rdsp, equal work (every row padded to 2048), 50 steps
#   rdsp-own    rdsp with its own padding (each microbatch to its longest row)
#
# RECOMPUTE (default 1) recomputes every layer on BOTH systems (Megatron: full,
# uniform, 1 layer per chunk; rdsp: recompute=1 on both stages). COMPILE_VISION
# (default 0) is rdsp's torch.compile of the vision encoder.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../../../.." && pwd)
export MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config
export LOGS=$HERE/../logs; mkdir -p "$LOGS"
RUNS=$RDSP/bench/mimo/runs.sh
DATA=exported:$HOME/mimo/cord_steps-$MODEL
# Decided 2026-10-11: without recompute both systems ran out of memory at step 2 (Adam's
# state appears after step 1), so both recompute every layer; the compiled vision encoder
# crashed under TP4 (MLP width 4304/4 = 1076, rows not 16-byte aligned), so it runs eager.
export RECOMPUTE=${RECOMPUTE:-1} COMPILE_VISION=${COMPILE_VISION:-0}
MEG_RC=""
[ "${RECOMPUTE:-0}" = 1 ] && MEG_RC="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
TAG=""; [ "${RECOMPUTE:-0}" = 1 ] && TAG=-rc

container() {  # the same NeMo container runs.sh uses
  sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
    -v "$HOME/hf:/root/.cache/huggingface" -v "$HOME/mimo:/workspace" -v "$RDSP:/rdsp" \
    -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge \
    nvcr.io/nvidia/nemo:26.08 bash -c "$*"
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo"
}
rdsp() {  # rdsp <log name> [env...]
  local name=$1; shift
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       DATA=$DATA "$@" bash "$HERE/rdsp.sh" ${RDSP_ARGS:-}) 2>&1 | tee "$LOGS/$name.log"
}

case "${1:-}" in
  prep)  # Bridge's checkpoint only: this layout has no MegatronMIMO run
    container "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
      --megatron-path /workspace/std/$MODEL" 2>&1 | tee "$LOGS/convert.log"
    bash "$RUNS" export ;;
  convert)  # HF -> Megatron checkpoint only
    container "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
      --megatron-path /workspace/std/$MODEL" 2>&1 | tee "$LOGS/convert${NAME:-}.log" ;;
  preflight-bridge)  # Bridge only, 3 steps; NAME suffixes the log; EXTRA: more Bridge overrides
    STEPS=3 MEGATRON_EXTRA="$MEG_RC ${EXTRA:-}" bash "$RUNS" tp4pp2-megatron || true
    mv "$LOGS/tp4pp2-megatron.log" "$LOGS/preflight-bridge$TAG${NAME:-}.log" ;;
  export)  # Bridge's samples only (rdsp-only sessions need no Megatron checkpoint)
    bash "$RUNS" export ;;
  preflight)
    STEPS=3 MEGATRON_EXTRA="$MEG_RC" bash "$RUNS" tp4pp2-megatron || true
    mv "$LOGS/tp4pp2-megatron.log" "$LOGS/preflight-bridge$TAG.log"
    # --profile: per-phase timers and peak memory per GPU (a 3-step memory check, not timing)
    RDSP_ARGS=--profile rdsp "preflight-rdsp$TAG" STEPS=3 RECOMPUTE="${RECOMPUTE:-0}" || true ;;
  preflight-rdsp)  # rdsp only; NAME suffixes the log, e.g. -nocompile; EXTRA: more train_vl.py flags
    RDSP_ARGS="--profile ${EXTRA:-}" rdsp "preflight-rdsp$TAG${NAME:-}" STEPS=3 RECOMPUTE="${RECOMPUTE:-0}" \
      COMPILE_VISION="${COMPILE_VISION:-1}" || true ;;
  bridge)  # NAME suffixes the log; EXTRA: more Bridge overrides
    MEGATRON_EXTRA="$MEG_RC ${EXTRA:-}" bash "$RUNS" tp4pp2-megatron
    mv "$LOGS/tp4pp2-megatron.log" "$LOGS/bridge$TAG${NAME:-}.log" ;;
  rdsp)     rdsp "rdsp-equal$TAG${NAME:-}" PAD=equal RECOMPUTE="${RECOMPUTE:-0}" ;;  # RDSP_ARGS: more flags
  rdsp-own) rdsp "rdsp-own$TAG" PAD=own RECOMPUTE="${RECOMPUTE:-0}" ;;
  *) sed -n 2,15p "$0"; exit 1 ;;
esac
