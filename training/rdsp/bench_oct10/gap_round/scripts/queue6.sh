#!/bin/bash
# queue6.sh, on node 1 (ops/h100/remote2.sh): the 27B on Megatron-MIMO's 9-GPU layout, MIMO then
# rdsp, 10 steps each (2026-10-19, the user: "just 10 steps ... make it faster"). The vision
# encoder alone on 1 GPU of node 2, language TP4 x PP2 on node 1; receipt rows, 128 x 2048 a step.
# MIMO with our memory settings (gap_round/mimo/patch_mimo.py; as shipped it ran out of memory,
# 04:50 UTC); rdsp: rdsp9.sh. Medians over steps 4-10 (MEDIAN_FIRST=4). MIMO prints its
# iteration lines on node 2 (the vision rank): copied to logs/mimo-9gpu-10-node2.log.
set -x
cd ~/rdsp
. ~/env2.sh
G=~/rdsp/bench_oct10/gap_round
GL=$G/logs
RUNS=~/rdsp/bench/mimo/runs.sh
BOTH=~/rdsp/bench/two_node/both.sh
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() { bash "$BOTH" --cleanup; pkill -f "[t]rain_vl.py"; $SSH2 "pkill -f '[t]rain_vl.py'" < /dev/null; sleep 10; }
run() {  # run <name> <timeout minutes> <log or -> <command...>: a two-node run, stopped early
  # when <log> shows an out-of-memory error (rdsp hangs after one)
  local name=$1 minutes=$2 log=$3; shift 3
  mark "START $name"
  $SMI > "$GL/nvsmi-$name-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$GL/nvsmi-$name-node2.csv" &
  local smi2=$!
  "$@" &
  local pid=$! t=0
  while kill -0 $pid 2>/dev/null; do
    if [ "$log" != - ] && grep -qaE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null; then
      sleep 20; mark "OOM $name"; break
    fi
    [ $t -ge $((minutes * 60)) ] && { mark "TIMEOUT $name"; break; }
    sleep 15; t=$((t + 15))
  done
  kill -0 $pid 2>/dev/null && { cleanup; bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"; }
  wait $pid; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  cleanup
  mark "END $name exit $rc"
}

mark "QUEUE6 START"
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
scp -i "$HOME/key2.pem" -o StrictHostKeyChecking=no "$G/scripts/medians.py" "ubuntu@$PRIV2:rdsp/bench_oct10/gap_round/scripts/"

run mimo-27b-9gpu-10 25 - bash "$BOTH" env MODEL=Qwen3.5-27B GBS=128 STEPS=10 LOGS="$GL" MIMO_PATCH=1 \
  MIMO_NAME=mimo-9gpu-10 MIMO_GPUS=9 MIMO_MBS=1 MIMO_NPROC0=8 MIMO_NPROC1=1 \
  MIMO_LANGUAGE=tp=4,pp=2,dp=1,rank_offset=0 MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=8 bash "$RUNS" mimo-layout
$SSH2 'cat "$(ls -t ~/both-*.log | head -1)"' < /dev/null > "$GL/mimo-9gpu-10-node2.log"
mark "MIMO 9 GPUs, median s over steps 4-10: $(MEDIAN_FIRST=4 MEDIAN_MIN=5 python3 "$G/scripts/medians.py" bridge "$GL/mimo-9gpu-10-node2.log" | awk '{print $1}')"

bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"
rm -f "$GL/rdsp-9gpu.log"
run rdsp-27b-9gpu-10 20 "$GL/rdsp-9gpu.log" bash "$G/scripts/rdsp9.sh"
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
mark "RDSP 9 GPUs, median s over steps 4-10: $(MEDIAN_FIRST=4 MEDIAN_MIN=5 python3 "$G/scripts/medians.py" rdsp "$GL/rdsp-9gpu.log" | awk '{print $1}')"
mark "ALL DONE 6"
