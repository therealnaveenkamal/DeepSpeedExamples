#!/bin/bash
# queue_tests3.sh, on the node after queue_tests2.sh (2026-11-11): the same trials for layout A
# of the two levers that could move the 9B MIMO rows further, after the MIMO run that is going.
#   T1  rdsp B with the language stage compiled (compile=1); if it fails early, again without --liger
#   T2  rdsp B with twice the rows per microbatch (4 rows x 16 microbatches; shipped: 2 x 32)
#   T3  MIMO B with twice the rows per microbatch (MIMO_MBS=4), the same change for MIMO
#   T4  rdsp A (language TP2 x PP2) with the gradient fix, shipped microbatches (1 row x 64)
#   T5  rdsp A with 2 rows x 32 microbatches
# B = 9b_v1_tp2pp1dp2, A = 9b_v1_tp2pp2dp1. All rdsp trials have RDSP_DIRECT_GRAD_ADD=1.
# Short runs (12-26 steps): they size the levers; they are not blog numbers.
# Last marker "=== ALL DONE TESTS". The final queue is NOT restarted from here.
set -uo pipefail
cd ~/rdsp
M=~/rdsp/bench_oct10/mimo_vs_rdsp
LB=$M/9b_v1_tp2pp1dp2/logs; LA=$M/9b_v1_tp2pp2dp1/logs; mkdir -p "$LA" "$LB"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]rain_vl.py"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1; sleep 8
}
STAGES_B="--stages 2 --cuts 0 --stage 0:gpus=1,zero=1 --stage 1:gpus=4,tp=2,zero=1"
STAGES_A="--stages 3 --cuts 0,balanced --stage 0:gpus=1,zero=1 --stage 1:gpus=2,tp=2,zero=1 --stage 2:gpus=2,tp=2,zero=1"
rdsp() {  # rdsp <log> <timeout s> <steps> <rows> <microbatches> <stage args...and extra args>
  local log=$1 t=$2 steps=$3 rows=$4 mb=$5; shift 5
  mark "START $(basename "$log" .log)"
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RDSP_DIRECT_GRAD_ADD=1 \
   timeout "$t" python ~/rdsp/train_vl.py --model Qwen/Qwen3.5-9B --dataset "exported:$HOME/mimo/cord_steps-Qwen3.5-9B" \
     "$@" --rows "$rows" --microbatches "$mb" --seq 2048 --pad-multiple 0 \
     --steps "$steps" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
     --grad-dtype fp32 --loss microbatch-mean \
     --prefetch --sharded-loss --drop-padding-mask) > "$log" 2>&1
  local rc=$?
  cleanup
  mark "END $(basename "$log" .log) exit $rc steps $(grep -a -c "^step " "$log")"
  grep -a "^step " "$log" | awk '{printf "%s:%s/%s ", $2, $4, $5} END {print ""}'
  grep -a -o "InternalTorchDynamoError.\{0,140\}\|OutOfMemoryError.\{0,80\}\|Error: .\{0,140\}" "$log" | sort | uniq -c | sort -rn | head -3
}
until grep -q "^=== ALL DONE TESTS2" ~/queue_tests2.out 2>/dev/null; do sleep 5; done
mimo_a() {  # mimo_a <rows per microbatch>
  mark "START mimo-nov11-a-mbs$1"
  env MODEL=Qwen3.5-9B LOGS="$LA" STEPS=20 MIMO_NAME=mimo-nov11-a-mbs$1 MIMO_GPUS=5 MIMO_MBS=$1 \
    MIMO_LANGUAGE="tp=2,pp=2,dp=1,rank_offset=0" MIMO_IMAGES="tp=1,pp=1,dp=1,rank_offset=4" \
    timeout 900 bash ~/rdsp/bench/mimo/runs.sh mimo-layout > /dev/null 2>&1; rc=$?
  cleanup
  mark "END mimo-nov11-a-mbs$1 exit $rc median $(MEDIAN_FIRST=6 MEDIAN_MIN=8 python3 ~/rdsp/bench_oct10/gap_round/scripts/medians.py bridge "$LA/mimo-nov11-a-mbs$1.log" | awk '{print $1}') s errors $(grep -a -c "OutOfMemoryError\|CUDA out of memory" "$LA/mimo-nov11-a-mbs$1.log")"
}
# T9   MIMO A with 2 rows per microbatch (shipped: 1); rdsp A at 2 rows ran 5.2 s against 8.6 s
mimo_a 2
# T10  rdsp A with 4 rows x 16 microbatches
rdsp "$LA/rdsp-nov11-rows4.log" 420 26 4 16 $STAGES_A --liger
# T11  MIMO A with 4 rows per microbatch, if the 2-row trial ran
grep -a -q "iteration  *10/" "$LA/mimo-nov11-a-mbs2.log" 2>/dev/null && mimo_a 4
# T12  rdsp A with 2 rows and both language stages compiled, without Liger
rdsp "$LA/rdsp-nov11-rows2-compile.log" 480 14 2 32 --stages 3 --cuts 0,balanced --stage 0:gpus=1,zero=1 --stage 1:gpus=2,tp=2,zero=1,compile=1 --stage 2:gpus=2,tp=2,zero=1,compile=1
mark "ALL DONE TESTS3"
