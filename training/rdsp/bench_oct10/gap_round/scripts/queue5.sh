#!/bin/bash
# queue5.sh, on node 1 (ops/h100/remote2.sh): the receipt control (2026-10-19, the user): phases
# E and F again with only the images changed. Receipt rows (CORD-v2, 39% image tokens at 2048)
# exported in NVIDIA's MedPix shape, 32 rows of 4096 a step, 10 steps; every system at its
# MedPix setting, so a change in speed against phases E/F comes from the images.
#   35B, both nodes: Bridge TP1 x PP2 x EP8 (NVIDIA's layout 'Et*17|t*23L', TE cross entropy, no
#   recompute), TorchTitan FSDP16 + EP8 (bf16 reduce), rdsp PP8 x DP2 (measured cut, every block
#   recomputed: without, it ran out of memory on MedPix)
#   27B, one node each: Bridge PP8 on node 1; TorchTitan FSDP8, then rdsp PP8 on node 2
#   (one_node_c4.sh)
# Logs: gap_round/logs/cord4096/ (35B), the 27B's in one_node_c4.sh. Markers ("=== ") for guard.sh.
set -x
cd ~/rdsp
. ~/env2.sh
B=~/rdsp/bench_oct10
G=$B/gap_round
T=$B/two_node_35b
GL=$G/logs/cord4096; mkdir -p "$GL"
RUNS=~/rdsp/bench/mimo/runs.sh
BOTH=~/rdsp/bench/two_node/both.sh
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
RSYNC2="rsync -a -e 'ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no'"
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
reached() {  # reached <log>: 3 steps done (rdsp "step 3", Bridge "iteration 3/" or a third
  # "Step Time" line on its two-node logs, TorchTitan "step: 3")
  grep -qaE "^step 3 |iteration +3/|step: +3 " "$1" 2>/dev/null || [ "$(grep -ac '^Step Time :' "$1" 2>/dev/null)" -ge 3 ]
}
nofit() { ! reached "$1" && grep -qaE "OutOfMemoryError|CUDA out of memory|out of memory" "$1" 2>/dev/null; }
export_rows() {  # export_rows: Bridge's rows for MODEL/RECIPE/GBS/SEQ/DATASET into ~/mimo/$EXPORT_NAME, both nodes
  mark "START export $EXPORT_NAME"
  if [ ! -f "$HOME/mimo/$EXPORT_NAME/stats.json" ]; then
    env LOGS="$GL" bash "$RUNS" export > /dev/null 2>&1
    mv "$GL/export.log" "$GL/export-$EXPORT_NAME.log" 2>/dev/null
  fi
  if [ -f "$HOME/mimo/$EXPORT_NAME/stats.json" ]; then
    cat "$HOME/mimo/$EXPORT_NAME/stats.json"; echo
    eval "$RSYNC2 $HOME/mimo/$EXPORT_NAME/ ubuntu@$PRIV2:mimo/$EXPORT_NAME/" || mark "FAILED copy $EXPORT_NAME to node 2"
  else
    mark "FAILED export $EXPORT_NAME"
  fi
  mark "END export $EXPORT_NAME"
}

mark "QUEUE5 START"
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
rsync -a -e "ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no" "$G/scripts/" "ubuntu@$PRIV2:rdsp/bench_oct10/gap_round/scripts/"

# ---- the rows: both models, receipts, 32 x 4096, 10 steps
export MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config GBS=32 SEQ=4096 \
  DATASET=cord_v2 STEPS=10 EXPORT_NAME=cord4096_steps-Qwen3.5-27B
export_rows
export MODEL=Qwen3.5-35B-A3B RECIPE=qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config \
  EXPORT_NAME=cord4096_steps-Qwen3.5-35B-A3B
export_rows

# ---- 35B, both nodes
. "$T/scripts/common.sh"   # RDSP_COMMON with --seq 4096; GBS 32, STEPS 10 kept
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
NV="\"model.pipeline_model_parallel_layout='Et*17|t*23L'\" model.cross_entropy_fusion_impl=te"
bridge35() {  # bridge35 <log name> [overrides...]: Bridge TP1 x PP2 x EP8 on the 35B, receipt rows
  local name=$1; shift
  run "bridge-35b-$name" 30 - bash "$BOTH" env MODEL=$MODEL RECIPE=$RECIPE GBS=32 SEQ=4096 \
    DATASET=cord_v2 STEPS=10 LOGS="$GL" NET=efa GRAD_REDUCE_FP32=false CUSTOM_NAME="bridge-$name" \
    CUSTOM_TP=1 CUSTOM_PP=2 MEGATRON_EXTRA="$MEG_COMMON $MEG_TP1 $FAST model.expert_model_parallel_size=8 \
    model.batch_p2p_comm=false model.batch_p2p_sync=false $NV $*" bash "$RUNS" custom-megatron
}
bridge35 cord4096-pp2ep8-nv
if nofit "$GL/bridge-cord4096-pp2ep8-nv.log"; then
  mark "NOFIT bridge-35b-cord4096-pp2ep8-nv: again with every layer recomputed"
  bridge35 cord4096-pp2ep8-nv-rc $RC_ALL
fi

run titan-35b-cord4096 25 - bash "$BOTH" env MODEL=$MODEL EXPORT_NAME=$EXPORT_NAME ROWS=32 SEQ=4096 \
  LOGS="$GL" STEPS=10 NAME=-cord4096 bash "$B/torchtitan/scripts/run2.sh" run qwen35_35b_a3b_cord_fsdp16_ep8_bf16

bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"
rm -f "$T/pp8dp2/logs/rdsp-rc-cord4096.log"
run rdsp-35b-cord4096-pp8dp2-rc 25 "$T/pp8dp2/logs/rdsp-rc-cord4096.log" \
  env NAME=-cord4096 RECOMPUTE=1 bash "$T/pp8dp2/run.sh" rdsp
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
mark "35B DONE"

# ---- 27B, one node each (one_node_c4.sh)
$SSH2 "bash ~/rdsp/bench_oct10/gap_round/scripts/one_node_c4.sh others 2>&1 | tee ~/one_node_c4-others.out \
  | grep --line-buffered '^=== '" < /dev/null | sed -u 's/^=== /=== node2 /' &
N2=$!
bash "$G/scripts/one_node_c4.sh" bridge
wait $N2
mark "27B DONE"
mark "ALL DONE 5"
