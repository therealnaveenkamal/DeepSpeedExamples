#!/bin/bash
# queue_wait_probe.sh, on the node: 12 steps of rdsp on the 9B MIMO layout (1 vision GPU + language
# TP2 x DP2) with --profile (each stage's forward, backward and waiting per step) and the direct
# gradient add on. Why: py-spy shows rdsp's language GPUs waiting 1.7 s a step; this says on what.
# Diagnosis only. Log: logs/rdsp-nov11-waitprobe.log. Last marker "=== ALL DONE PROBE".
set -uo pipefail
cd ~/rdsp
M=~/rdsp/bench_oct10/mimo_vs_rdsp; L=$M/9b_v1_tp2pp1dp2/logs
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
mark "START probe"
(unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
 env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RDSP_DIRECT_GRAD_ADD=1 \
 timeout 600 python ~/rdsp/train_vl.py --model Qwen/Qwen3.5-9B --dataset "exported:$HOME/mimo/cord_steps-Qwen3.5-9B" \
   --stages 2 --cuts 0 --stage 0:gpus=1,zero=1 --stage 1:gpus=4,tp=2,zero=1 \
   --rows 2 --microbatches 32 --seq 2048 --pad-multiple 0 \
   --steps 12 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
   --grad-dtype fp32 --loss microbatch-mean --liger \
   --prefetch --sharded-loss --drop-padding-mask --profile) > "$L/rdsp-nov11-waitprobe.log" 2>&1
rc=$?
pkill -f "[t]rain_vl.py"; bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1; sleep 5
mark "END probe exit $rc"
grep -a "^step " "$L/rdsp-nov11-waitprobe.log" | tail -4
mark "ALL DONE PROBE"
