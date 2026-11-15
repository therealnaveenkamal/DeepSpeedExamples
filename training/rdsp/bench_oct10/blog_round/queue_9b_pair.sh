#!/bin/bash
# queue_9b_final2.sh, on the node (ops/h100/remote.sh; never after a push): 50-step runs of the two
# 9B MIMO-comparison layouts at a chosen number of rows per microbatch, both systems, with repeats.
#   A = 9b_v1_tp2pp2dp1 (1 vision GPU + language TP2 x PP2; shipped: 1 row x 64 microbatches)
#   B = 9b_v1_tp2pp1dp2 (1 vision GPU + language TP2 x DP2; shipped: 2 rows x 32 microbatches)
# The order comes from ~/order9b_pair.txt, re-read before every run (so later lines can be edited):
#   <A|B> mimo <rows per microbatch> <run number>
#   <A|B> rdsp <rows per microbatch> <run number> [compile] [rcN] [leadN]
# leadN: RDSP_FIRST_STAGE_LEAD=N, the vision stage runs N more microbatches ahead (schedule.py).
# This copy (queue_9b_pair.sh) runs on node 1 of the two-node pair, on its own 8 GPUs, after prep9b_pair.sh.
# rcN: recompute the first N decoder layers of the first language stage (recompute_blocks=N),
# which frees activation memory there for more rows per microbatch (layout A ran out of memory
# at 4 rows in its 18-layer stage; MIMO's launcher has no such option).
# MIMO: its launcher unchanged, MIMO_MBS = rows per microbatch. rdsp: RDSP_DIRECT_GRAD_ADD=1 always;
# "compile" compiles the language stages and runs without Liger's kernels (compile crashes with them).
# Logs: <layout>/logs/mimo-nov11-<R>rows-run<n>.log, rdsp-nov11-<R>rows[-compile]-run<n>.log, and
# nvsmi-<same>.csv (memory used per GPU). Each END line carries the median over steps 6-50.
# Last marker "=== ALL DONE 9B FINAL2".
set -uo pipefail
cd ~/rdsp
M=~/rdsp/bench_oct10/mimo_vs_rdsp
MED="python3 $HOME/rdsp/bench_oct10/gap_round/scripts/medians.py"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]rain_vl.py"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1; sleep 10
}
one() {  # one <A|B> <mimo|rdsp> <rows> <n> [compile] [rcN] [leadN]
  local lay=$1 sys=$2 rows=$3 n=$4; shift 4
  local opt="" o L lang stages c="" liger="--liger" name kind rc peak rc1="" lead=0
  local mb=$((64 / rows))
  for o in "$@"; do
    opt="${opt:+$opt-}$o"
    case $o in
      compile) c=",compile=1"; liger="" ;;
      rc[0-9]*) rc1=",recompute_blocks=${o#rc}" ;;
      lead[0-9]*) lead=${o#lead} ;;   # RDSP_FIRST_STAGE_LEAD: the vision stage runs that many more microbatches ahead
    esac
  done
  case $lay in
    A) L=$M/9b_v1_tp2pp2dp1/logs; lang="tp=2,pp=2,dp=1"
       stages="--stages 3 --cuts 0,balanced --stage 0:gpus=1,zero=1 --stage 1:gpus=2,tp=2,zero=1$c$rc1 --stage 2:gpus=2,tp=2,zero=1$c" ;;
    B) L=$M/9b_v1_tp2pp1dp2/logs; lang="tp=2,pp=1,dp=2"
       stages="--stages 2 --cuts 0 --stage 0:gpus=1,zero=1 --stage 1:gpus=4,tp=2,zero=1$c$rc1" ;;
    *) return 1 ;;
  esac
  mkdir -p "$L"
  case $sys in
    mimo) name=mimo-nov11-${rows}rows-run$n; kind=bridge ;;
    rdsp) name=rdsp-nov11-${rows}rows${opt:+-$opt}-run$n; kind=rdsp ;;
    *) return 1 ;;
  esac
  mark "START $lay $name"
  $SMI > "$L/nvsmi-$name.csv" &
  local smi=$!
  if [ $sys = mimo ]; then
    env MODEL=Qwen3.5-9B LOGS="$L" STEPS=50 MIMO_NAME=$name MIMO_GPUS=5 MIMO_MBS=$rows \
      MIMO_LANGUAGE="$lang,rank_offset=0" MIMO_IMAGES="tp=1,pp=1,dp=1,rank_offset=4" \
      timeout 1500 bash ~/rdsp/bench/mimo/runs.sh mimo-layout > /dev/null 2>&1; rc=$?
  else
    (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
     env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RDSP_DIRECT_GRAD_ADD=1 RDSP_FIRST_STAGE_LEAD=$lead \
     timeout 1200 python ~/rdsp/train_vl.py --model Qwen/Qwen3.5-9B --dataset "exported:$HOME/mimo/cord_steps-Qwen3.5-9B" \
       $stages --rows "$rows" --microbatches "$mb" --seq 2048 --pad-multiple 0 \
       --steps 50 --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
       --grad-dtype fp32 --loss microbatch-mean $liger \
       --prefetch --sharded-loss --drop-padding-mask) > "$L/$name.log" 2>&1; rc=$?
  fi
  kill $smi; cleanup
  peak=$(awk -F', ' '{if ($3 > m) m = $3} END {print m}' "$L/nvsmi-$name.csv")
  mark "END $lay $name exit $rc median $($MED $kind "$L/$name.log" | awk '{print $1}') s peak ${peak} MiB errors $(grep -a -c "OutOfMemoryError\|CUDA out of memory\|InternalTorchDynamoError" "$L/$name.log")"
}
until grep -qE "^=== PREP9B (DONE|FAILED)" ~/prep9b_pair.out 2>/dev/null; do sleep 5; done
mark "9B PAIR START"
i=1
while :; do
  line=$(sed -n "${i}p" ~/order9b_pair.txt 2>/dev/null)
  [ -n "$line" ] || break
  one $line
  i=$((i + 1))
done
mark "ALL DONE 9B PAIR"
