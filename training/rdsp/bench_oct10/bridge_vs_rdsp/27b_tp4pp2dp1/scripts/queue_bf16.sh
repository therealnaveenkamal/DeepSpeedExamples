#!/bin/bash
# On the H100 box (2026-10-11): rdsp only, 27B TP4xPP2xDP1, no recompute, 3 steps, with
# bf16 gradient accumulation + fused AdamW (fp32 master and Adam kept). Does it fit?
set -x
cd ~/rdsp
S=~/rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts
L=~/rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/logs; mkdir -p $L
bash bench/setup_node.sh > ~/setup.log 2>&1 || { echo "SETUP FAILED"; exit 1; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi topo -m; lscpu | head -20; } > $L/hardware-bf16.txt 2>&1
# the 27B weights download while the container exports Bridge's samples
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > ~/download.log 2>&1 &
bash $S/run.sh export
wait
echo "EXPORT DONE"
RECOMPUTE=0 COMPILE_VISION=0 NAME=-bf16grads-fused EXTRA="--grad-dtype bf16 --fused-adam" bash $S/run.sh preflight-rdsp
grep -E "rdsp_profile|^step|OutOfMemory" $L/preflight-rdsp-bf16grads-fused.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240 | tail -40
echo "QUEUE DONE"
