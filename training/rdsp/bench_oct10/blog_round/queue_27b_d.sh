#!/bin/bash
# queue_27b.sh, on node 1 of the two-node pair after prep2.sh (start it with ops/h100/remote2.sh;
# never push code while it runs): the dense 27B on Megatron-MIMO's 9-GPU layout, 1 vision GPU +
# language TP4 x PP2, CORD-v2 rows, 128 x 2048 a step, both systems.
# The order comes from ~/order27b_d.txt, re-read before every run (later lines can be edited):
#   mimo <rows per microbatch> <steps> <run number> [rcfull]
#   rdsp <rows per microbatch> <steps> <run number> [profile] [rcN] [tp2pp4] [compile] [cuts=A,B]
# cuts=A,B (rdsp only): the first language stage starts at layer A and the second at layer B.
# A > 0 puts A decoder layers on the vision GPU, which is idle about half of each step.
# tp2pp4 (rdsp only): the language model as four stages of 2 GPUs on the same 8 language GPUs,
# where MIMO's layout has two stages of 4: half the layers per GPU, two-way tensor parallelism.
# MIMO: its example launcher with our memory settings (gap_round/mimo/patch_mimo.py: bf16 gradients,
# exact fp32 master as 2-byte remainders; as shipped it runs out of memory), MIMO_MBS = rows per
# microbatch; language on node 1's 8 GPUs, the vision GPU on node 2 (its step lines print there).
# rdsp: gap_round/scripts/rdsp9.sh's settings (bf16 gradients, 2-byte master, measured cut after
# the vision-only stage, MIMO's per-microbatch loss mean), the vision stage and the first language
# stage on node 1, the second on node 2. profile: --profile (each stage's forward, backward and
# waiting per step; diagnosis). rcN: recompute the first N decoder layers of each language stage.
# Logs in blog_round/logs27b/: <name>.log, MIMO's step lines in <name>-node2.log, nvsmi-<name>-node{1,2}.csv.
# Medians: steps 6-50 of a 50-step run; steps 4-10 of a 10-step run. Last marker "=== ALL DONE 27B".
set -uo pipefail
cd ~/rdsp
. ~/env2.sh
G=~/rdsp/bench_oct10/blog_round
GL=$G/logs27b; mkdir -p "$GL"
RUNS=~/rdsp/bench/mimo/runs.sh
BOTH=~/rdsp/bench/two_node/both.sh
RAY=~/rdsp/bench/two_node/ray_cluster.sh
MED="python3 $HOME/rdsp/bench_oct10/gap_round/scripts/medians.py"
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() { bash "$BOTH" --cleanup; pkill -f "[t]rain_vl.py"; $SSH2 "pkill -f '[t]rain_vl.py'" < /dev/null; sleep 10; }
med() {  # med <kind> <log> <steps>
  if [ "$3" -ge 40 ]; then $MED "$1" "$2" | awk '{print $1}'
  else MEDIAN_FIRST=4 MEDIAN_MIN=5 $MED "$1" "$2" | awk '{print $1}'; fi
}
peak() { awk -F', ' '{if ($3 > m) m = $3} END {print m + 0}' "$1" 2>/dev/null; }
run() {  # run <name> <timeout minutes> <log to watch for an out-of-memory error, or -> <command...>
  local name=$1 minutes=$2 log=$3; shift 3
  $SMI > "$GL/nvsmi-$name-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$GL/nvsmi-$name-node2.csv" &
  local smi2=$!
  "$@" &
  local pid=$! t=0
  while kill -0 $pid 2>/dev/null; do
    if [ "$log" != - ] && grep -qaE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null; then
      sleep 20; mark "OOM $name"; break   # rdsp hangs after one
    fi
    [ $t -ge $((minutes * 60)) ] && { mark "TIMEOUT $name"; break; }
    sleep 15; t=$((t + 15))
  done
  kill -0 $pid 2>/dev/null && { cleanup; bash "$RAY" down "$PRIV1" "$PRIV2" > /dev/null 2>&1; }
  wait $pid; RC=$?
  kill $smi1 $smi2 2>/dev/null; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  cleanup
}
one() {  # one <mimo|rdsp> <rows> <steps> <n> [profile] [rcN]
  local sys=$1 rows=$2 steps=$3 n=$4; shift 4
  local prof="" rc="" tag="" o mrc=0 lay=tp4pp2 comp="" cuts="0,measured"
  for o in "$@"; do
    case $o in
      profile) prof="--profile"; tag="$tag-profile" ;;
      rc[0-9]*) rc=",recompute_blocks=${o#rc}"; tag="$tag-$o" ;;
      rcfull) mrc=1; tag="$tag-rcfull" ;;
      tp2pp4) lay=tp2pp4; tag="$tag-tp2pp4" ;;
      compile) comp=",compile=1"; tag="$tag-compile" ;;
      cuts=*) cuts=${o#cuts=}; tag="$tag-cuts${cuts//,/_}" ;;   # rdsp only: explicit stage cuts (no calibration, no engine rebuild), e.g. cuts=0,32 or cuts=3,33   # rdsp only: language stages compiled, run without Liger (compile crashes with it)   # rdsp only: language as four stages of 2 GPUs (TP2 x PP4)   # MIMO only: every language layer recomputed (the patched launcher's MIMO_RECOMPUTE=1)
    esac
  done
  local minutes=$(( steps >= 40 ? 50 : 26 ))
  if [ "$sys" = mimo ]; then
    local name=mimo-27b-${rows}rows-${steps}steps$tag-run$n
    mark "START $name"
    bash "$RAY" down "$PRIV1" "$PRIV2" > /dev/null 2>&1
    run "$name" $minutes - bash "$BOTH" env MODEL=Qwen3.5-27B GBS=128 STEPS=$steps LOGS="$GL" MIMO_PATCH=1 MIMO_RECOMPUTE=$mrc \
      MIMO_NAME=$name MIMO_GPUS=9 MIMO_MBS=$rows MIMO_NPROC0=8 MIMO_NPROC1=1 \
      MIMO_LANGUAGE=tp=4,pp=2,dp=1,rank_offset=0 MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=8 bash "$RUNS" mimo-layout
    $SSH2 'cat "$(ls -t ~/both-*.log | head -1)"' < /dev/null > "$GL/$name-node2.log"
    mark "END $name exit $RC median $(med bridge "$GL/$name-node2.log" "$steps") s peak node1 $(peak "$GL/nvsmi-$name-node1.csv") node2 $(peak "$GL/nvsmi-$name-node2.csv") MiB oom $(cat "$GL/$name.log" "$GL/$name-node2.log" 2>/dev/null | grep -a -c "OutOfMemoryError\|CUDA out of memory")"
  else
    local name=rdsp-27b-${rows}rows-${steps}steps$tag-run$n
    local stages="--stages 3 --cuts $cuts --stage 0:gpus=1,tp=1,zero=1 --stage 1:gpus=4,tp=4,zero=1$rc$comp --stage 2:gpus=4,tp=4,zero=1$rc$comp"
    [ $lay = tp2pp4 ] && stages="--stages 5 --cuts $cuts --stage 0:gpus=1,tp=1,zero=1 --stage 1:gpus=2,tp=2,zero=1$rc$comp --stage 2:gpus=2,tp=2,zero=1$rc$comp --stage 3:gpus=2,tp=2,zero=1$rc$comp --stage 4:gpus=2,tp=2,zero=1$rc$comp"
    mark "START $name"
    bash "$RAY" up "$PRIV1" "$PRIV2" > "$GL/ray-$name.log" 2>&1 || mark "ray cluster did not come up (see ray-$name.log)"
    rm -f "$GL/$name.log"
    run "$name" $minutes "$GL/$name.log" bash -c "
      export MODEL=Qwen3.5-27B EXPORT_NAME=cord_steps-Qwen3.5-27B-128 GBS=128 STEPS=$steps
      . ~/rdsp/bench_oct10/two_node_35b/scripts/common.sh
      export LOGS=$GL
      [ -z '$comp' ] || RDSP_COMMON=\${RDSP_COMMON/--liger/}
      rdsp $name $stages \
        --rows $rows --microbatches $((128 / rows)) --loss microbatch-mean $prof > /dev/null 2>&1"
    bash "$RAY" down "$PRIV1" "$PRIV2" > /dev/null 2>&1
    mark "END $name exit $RC median $(med rdsp "$GL/$name.log" "$steps") s peak node1 $(peak "$GL/nvsmi-$name-node1.csv") node2 $(peak "$GL/nvsmi-$name-node2.csv") MiB oom $(grep -a -c "OutOfMemoryError\|CUDA out of memory" "$GL/$name.log") cut $(grep -a "^stages (blocks)" "$GL/$name.log" | tail -1 | cut -c17-60)"
  fi
}
mark "27B QUEUE D START"
i=1
while :; do
  line=$(sed -n "${i}p" ~/order27b_d.txt 2>/dev/null)
  [ -n "$line" ] || break
  one $line
  i=$((i + 1))
done
mark "ALL DONE 27B-D"
