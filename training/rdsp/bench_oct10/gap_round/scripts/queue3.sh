#!/bin/bash
# queue3.sh, on node 1 (ops/h100/remote2.sh), after queue2.sh: phase E, NVIDIA's published
# fine-tuning setup for this model (Megatron-Bridge docs, Qwen3.6-35B-A3B "SFT · H100": recipe
# qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config, TP1 x PP2 x EP8, MedPix, 32 rows of 4096 a step,
# every layer recomputed; 2,507 tokens/s per GPU on 16 H100), run by every system on the 35B over
# both nodes. Differences from NVIDIA's run: Megatron-Bridge 0.6.0 (theirs: nightly), images at
# their own size (0.6.0's MedPix preset has no 448x448 resize), Qwen3.5-35B-A3B (same
# architecture as Qwen3.6-35B-A3B), our equal-work settings (bf16 gradients, 2-byte master, no
# router balancing loss), 50 steps.
#   E0  Bridge's MedPix rows exported (32 a step, 4096 long) for rdsp and TorchTitan
#   E1  Bridge TP1 x PP2 x EP8 (NVIDIA's layout; only if PP2 x EP8 ran across the nodes in
#       phase B) and TP1 x EP8, every layer recomputed as in NVIDIA's run
#   E2  TorchTitan FSDP16 + EP8, bf16 reduce
#   E3  rdsp PP8 x DP2 (measured cut), then the vision stage at the width rdsp picks
# Markers ("=== ") for ops/h100/guard.sh.
set -x
cd ~/rdsp
. ~/env2.sh
B=~/rdsp/bench_oct10
G=$B/gap_round
T=$B/two_node_35b
GL=$G/logs; mkdir -p "$GL"
RUNS=~/rdsp/bench/mimo/runs.sh
BOTH=~/rdsp/bench/two_node/both.sh
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
RSYNC2="rsync -a -e 'ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no'"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() { bash "$BOTH" --cleanup; pkill -f "[t]rain_vl.py"; sleep 10; }
run() {  # run <name> <timeout minutes> <command...>: a two-node run
  local name=$1 minutes=$2; shift 2
  mark "START $name"
  $SMI > "$GL/nvsmi-$name-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$GL/nvsmi-$name-node2.csv" &
  local smi2=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  cleanup
  mark "END $name exit $rc"
}
reached() { grep -qaE "^step 3 |iteration +3/|step: +3 " "$1" 2>/dev/null; }
nofit() { ! reached "$1" && grep -qaE "OutOfMemoryError|CUDA out of memory|out of memory" "$1" 2>/dev/null; }

while pgrep -f "[b]ash queue2.sh" > /dev/null; do sleep 30; done
mark "QUEUE3 START"
reached() {  # reached <log>: 3 steps done (rdsp "step 3", Bridge "iteration 3/" or a third
  # "Step Time" line on its two-node logs, TorchTitan "step: 3")
  grep -qaE "^step 3 |iteration +3/|step: +3 " "$1" 2>/dev/null || [ "$(grep -ac '^Step Time :' "$1" 2>/dev/null)" -ge 3 ]
}
nofit() { ! reached "$1" && grep -qaE "OutOfMemoryError|CUDA out of memory|out of memory" "$1" 2>/dev/null; }
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
export MODEL=Qwen3.5-35B-A3B RECIPE=qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config GBS=32 SEQ=4096 \
  DATASET=medpix STEPS=50 EXPORT_NAME=medpix_steps-Qwen3.5-35B-A3B
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

# ---- E0: the 35B's MedPix rows
export_rows

. "$T/scripts/common.sh"   # after the exports above: RDSP_COMMON with --seq 4096, GBS 32
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
bridgeE1() {  # bridgeE1 <log name> <tp> <pp> [overrides...]: one Bridge run on the 35B, MedPix
  local name=$1 tp=$2 pp=$3; shift 3
  local tp1=""; [ "$tp" = 1 ] && tp1=$MEG_TP1
  run "bridge-35b-$name" 80 bash "$BOTH" env MODEL=$MODEL RECIPE=$RECIPE GBS=32 SEQ=4096 \
    DATASET=medpix STEPS=50 LOGS="$GL" NET=efa GRAD_REDUCE_FP32=false CUSTOM_NAME="bridge-$name" \
    CUSTOM_TP=$tp CUSTOM_PP=$pp MEGATRON_EXTRA="$MEG_COMMON $tp1 $FAST $*" bash "$RUNS" custom-megatron
}
bridgeE() {  # bridgeE <name> <tp> <pp> [overrides...]: no recompute first (the rule for every
  # system), every layer recomputed (as NVIDIA's run) only if that does not fit
  local name=$1; shift
  bridgeE1 "medpix-$name" "$@"
  if nofit "$GL/bridge-medpix-$name.log"; then
    mark "NOFIT bridge-35b-medpix-$name: again with every layer recomputed"
    bridgeE1 "medpix-$name-rc" "$@" $RC_ALL
  fi
}

# ---- E1, E2 (2026-10-19 03:45 UTC, the user: "all frameworks with finalized best configs"): each
# competitor at its fastest phase-B setting (scripts/medians.py --best), plus NVIDIA's own layout
# for Bridge (TP1 x PP2 x EP8) when it ran across the nodes in phase B
best=$(python3 "$G/scripts/medians.py" --best bridge "$GL/bridge-tp1ep8-again.log" "$GL/bridge-tp1ep8-rc.log" \
  "$GL/bridge-pp2ep8.log" "$GL/bridge-pp2ep8-rc.log" "$GL/bridge-pp2ep8-nv.log" "$GL/bridge-pp2ep8-nv-rc.log" | awk '{print $2}')
mark "BRIDGE 35B BEST IN PHASE B: ${best:-none}"
case "$best" in
  *pp2ep8.log) ;;
  *tp1ep8*)    bridgeE tp1ep8 1 1 model.expert_model_parallel_size=8 ;;
  *)           bridgeE tp2ep16 2 1 model.expert_model_parallel_size=16 ;;   # the 0.6.0 recipe's own
esac
NV="\"model.pipeline_model_parallel_layout='Et*17|t*23L'\" model.cross_entropy_fusion_impl=te"
if reached "$GL/bridge-pp2ep8-nv.log" || reached "$GL/bridge-pp2ep8-nv-rc.log"; then
  bridgeE pp2ep8-nv 1 2 model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false $NV
elif reached "$GL/bridge-pp2ep8.log" || reached "$GL/bridge-pp2ep8-rc.log"; then
  bridgeE pp2ep8 1 2 model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false
else
  mark "SKIP bridge-35b-medpix-pp2ep8: Bridge PP2 x EP8 did not run across the nodes in phase B"
fi
best=$(python3 "$G/scripts/medians.py" --best titan "$T/fsdp16/logs/run-qwen35_35b_a3b_cord_fsdp16_ep8-gap-node0.log" \
  "$T/fsdp16/logs/run-qwen35_35b_a3b_cord_fsdp16_ep8_bf16-gap-node0.log" | awk '{print $2}')
cfg=$(basename "${best:-run-qwen35_35b_a3b_cord_fsdp16_ep8_bf16-gap-node0.log}" -gap-node0.log); cfg=${cfg#run-}
mark "TITAN 35B BEST IN PHASE B: $cfg"
run "titan-35b-medpix-$cfg" 40 bash "$BOTH" env MODEL=$MODEL EXPORT_NAME=$EXPORT_NAME ROWS=32 SEQ=4096 \
  LOGS="$GL" STEPS=50 NAME=-medpix bash "$B/torchtitan/scripts/run2.sh" run "$cfg"

# ---- E3: rdsp PP8 x DP2 (measured cut), then the vision stage at the width rdsp picks
bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"
run rdsp-35b-medpix-pp8dp2 40 env NAME=-medpix bash "$T/pp8dp2/run.sh" rdsp
LOG=$T/pp8dp2/logs/rdsp-medpix.log
if nofit "$LOG"; then
  mark "NOFIT rdsp-35b-medpix-pp8dp2: again with every layer recomputed"
  bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"
  run rdsp-35b-medpix-pp8dp2-rc 40 env NAME=-medpix RECOMPUTE=1 bash "$T/pp8dp2/run.sh" rdsp
  LOG=$T/pp8dp2/logs/rdsp-rc-medpix.log
fi
K=$(python3 "$T/scripts/layout_from_log.py" vision-width "$LOG" 16)
mark "VISION MEDPIX K=$K"
if [ -n "$K" ]; then
  bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"
  run rdsp-35b-medpix-vision 40 env NAME=-medpix K="$K" RECOMPUTE=1 bash "$T/vision_stage/run.sh" rdsp
fi
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
mark "PHASE E DONE"

# ---- phase F: the dense 27B on one node at the same setup, slide 14's layouts (one_node_medpix.sh)
export MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config EXPORT_NAME=medpix_steps-Qwen3.5-27B
export_rows
$SSH2 "bash ~/rdsp/bench_oct10/gap_round/scripts/one_node_medpix.sh others 2>&1 | tee ~/one_node_medpix-others.out \
  | grep --line-buffered '^=== '" < /dev/null | sed -u 's/^=== /=== node2 /' &
N2=$!
bash "$G/scripts/one_node_medpix.sh" bridge
wait $N2
$SSH2 "cat ~/one_node_medpix-others.out" < /dev/null > "$GL/one_node_medpix-others-node2.out"
mark "PHASE F DONE"
mark "ALL DONE 3"
