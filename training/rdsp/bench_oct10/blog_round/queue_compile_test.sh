#!/bin/bash
# queue_compile_test.sh, on the node: one short trial between the final 9B runs (2026-11-11).
# Why: the timing probe shows rdsp's language stage is the pacing one (busy ~93% of the step), and
# its backward pass is half launch-bound. torch.compile on that stage (stage key compile=1) fuses
# small operations, but crashed on 2026-10-05 (4B, "Output 0 of ApplyTemplate is a view ...
# modified inplace"). Trial 1: today's settings + compile=1. Only if that fails early, trial 2:
# the same without --liger (Liger's kernels are custom autograd functions, a suspect).
# 12 steps each, strict timeouts. Then the rest of the final queue starts.
set -uo pipefail
cd ~/rdsp
M=~/rdsp/bench_oct10/mimo_vs_rdsp; L=$M/9b_v1_tp2pp1dp2/logs
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
until grep -q "^=== ALL DONE 9B FINAL" ~/queue_9b_final.out 2>/dev/null; do sleep 5; done
cp ~/queue_9b_final.out ~/queue_9b_final.part1.out
trial() {  # trial <name> <timeout s> <extra train_vl args...>
  local name=$1 t=$2; shift 2
  mark "START $name"
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RDSP_DIRECT_GRAD_ADD=1 \
   timeout "$t" python ~/rdsp/train_vl.py --model Qwen/Qwen3.5-9B --dataset "exported:$HOME/mimo/cord_steps-Qwen3.5-9B" \
     --stages 2 --cuts 0 --stage 0:gpus=1,zero=1 --stage 1:gpus=4,tp=2,zero=1,compile=1 \
     --rows 2 --microbatches 32 --seq 2048 --pad-multiple 0 \
     --steps 12 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
     --grad-dtype fp32 --loss microbatch-mean \
     --prefetch --sharded-loss --drop-padding-mask "$@") > "$L/rdsp-nov11-$name.log" 2>&1
  local rc=$?
  pkill -f "[t]rain_vl.py"; bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1; sleep 8
  mark "END $name exit $rc steps $(grep -a -c "^step " "$L/rdsp-nov11-$name.log")"
  grep -a "^step " "$L/rdsp-nov11-$name.log" | awk '{printf "%s:%s/%s ", $2, $4, $5} END {print ""}'
  grep -a -o "InternalTorchDynamoError.\{0,160\}\|Error: .\{0,160\}" "$L/rdsp-nov11-$name.log" | sort | uniq -c | sort -rn | head -3
}
trial compile-liger 330 --liger
if ! grep -a -q "^step 3 " "$L/rdsp-nov11-compile-liger.log"; then
  trial compile-noliger 300
fi
mark "ALL DONE COMPILE TEST"
cat > ~/9b_order.txt <<'ORDER'
A fix 1
B mimo 2
B fix 2
A mimo 2
A fix 2
A mimo 3
A fix 3
A base 1
ORDER
nohup bash ~/queue_9b_final.sh > ~/queue_9b_final.out 2>&1 < /dev/null &
mark "final queue restarted"
