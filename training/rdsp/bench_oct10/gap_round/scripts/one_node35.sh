#!/bin/bash
# one_node35.sh <rdsp-titan|bridge>: Qwen3.5-35B-A3B on one node (8 GPUs, 64 rows a step), phase C
# of ../plan.md. queue2.sh starts "rdsp-titan" on node 1 and "bridge" on node 2 at the same time;
# cleanup touches this node only. A run that runs out of memory before step 3 runs again with
# every layer recomputed. Markers ("=== ") as in the queue.
#   rdsp-titan  rdsp 8 stages of 1 GPU (--cuts measured), then the vision stage at the width rdsp
#               picks from it; TorchTitan FSDP8 + EP8 (fp32 weights and Adam: a fit check first)
#   bridge      Megatron-Bridge TP1 x EP8 (NVIDIA's 8-GPU perf layout for this model), then
#               TP1 x PP2 x EP4
set -uo pipefail
cd ~/rdsp
B=~/rdsp/bench_oct10
G=$B/gap_round
GL=$G/logs; mkdir -p "$GL"
. "$B/two_node_35b/scripts/common.sh"   # MEG_COMMON, MEG_TP1, MEG_RC, RDSP_COMMON, vision_first, stage_flags
export MODEL=Qwen3.5-35B-A3B GBS=64 EXPORT_NAME=cord_steps-Qwen3.5-35B-A3B-64
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]orchrun"; pkill -f "[t]rain_vl.py"
  sleep 10
}
run() {  # run <name> <timeout minutes> <command...>
  local name=$1 minutes=$2; shift 2
  mark "START $name"
  $SMI > "$GL/nvsmi-$name.csv" &
  local smi=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi
  cleanup
  mark "END $name exit $rc"
}
nofit() {  # nofit <log>: ran out of memory before step 3
  ! grep -qaE "^step 3 |iteration +3/|step: +3 " "$1" 2>/dev/null &&
    grep -qaE "OutOfMemoryError|CUDA out of memory|out of memory" "$1" 2>/dev/null
}
bridge1() {  # bridge1 <name> <tp> <pp> [overrides...]: Bridge on this node
  local name=$1 tp=$2 pp=$3; shift 3
  run "bridge-35b-1n-$name" 80 env RECIPE=qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config STEPS=50 \
    LOGS="$GL" GRAD_REDUCE_FP32=false CUSTOM_NAME="bridge-1n-$name" CUSTOM_TP=$tp CUSTOM_PP=$pp \
    MEGATRON_EXTRA="$MEG_COMMON $MEG_TP1 $FAST $*" bash "$RUNS" custom-megatron
}
case "${1:-}" in
  rdsp-titan)
    run rdsp-35b-1n-pp8 40 bash "$G/scripts/rdsp_local.sh" "$GL/rdsp-1n-pp8.log" --stages 8 --cuts measured $(vision_first 1) \
      $(stage_flags 1 7 1) --rows 1 --microbatches 64
    LOG=$GL/rdsp-1n-pp8.log
    if nofit "$LOG"; then
      mark "NOFIT rdsp-35b-1n-pp8: again with every layer recomputed"
      export RECOMPUTE=1
      run rdsp-35b-1n-pp8-rc 40 env RECOMPUTE=1 bash "$G/scripts/rdsp_local.sh" "$GL/rdsp-1n-pp8-rc.log" --stages 8 --cuts measured $(vision_first 1) \
        $(stage_flags 1 7 1) --rows 1 --microbatches 64
      LOG=$GL/rdsp-1n-pp8-rc.log; export RECOMPUTE=0
    fi
    K=$(python3 "$B/two_node_35b/scripts/layout_from_log.py" vision-width "$LOG" 8)
    mark "VISION 1N K=$K"
    if [ -n "$K" ] && [ "$K" -lt 8 ]; then
      P=$(( (8 - K) / 2 ))
      export RECOMPUTE=1   # the two-node vision stage needed it; one node has half the memory
      run rdsp-35b-1n-vision 40 env RECOMPUTE=1 bash "$G/scripts/rdsp_local.sh" "$GL/rdsp-1n-k$K-rc.log" --stages $((P + 1)) --cuts 0,measured \
        $(vision_first "$K") $(stage_flags 1 $P 2) --rows "$K" --microbatches $((64 / K))
      export RECOMPUTE=0
    fi
    run titan-35b-1n-fit 25 env NAME=-gap MODEL=$MODEL EXPORT_NAME=$EXPORT_NAME ROWS=64 \
      bash "$B/torchtitan/scripts/run.sh" fit qwen35_35b_a3b_cord_fsdp8_ep8_rc_bf16
    if grep -qaE "step: +3 " "$B/torchtitan/logs/fit-qwen35_35b_a3b_cord_fsdp8_ep8_rc_bf16-gap.log"; then
      run titan-35b-1n 40 env NAME=-gap MODEL=$MODEL EXPORT_NAME=$EXPORT_NAME ROWS=64 \
        bash "$B/torchtitan/scripts/run.sh" run qwen35_35b_a3b_cord_fsdp8_ep8_rc_bf16
    else
      mark "NOFIT titan-35b-1n"
    fi ;;
  bridge)
    bridge1 tp1ep8 1 1 model.expert_model_parallel_size=8
    if nofit "$GL/bridge-1n-tp1ep8.log"; then
      mark "NOFIT bridge-35b-1n-tp1ep8: again with every layer recomputed"
      bridge1 tp1ep8-rc 1 1 model.expert_model_parallel_size=8 $MEG_RC
    fi
    bridge1 pp2ep4 1 2 model.expert_model_parallel_size=4 model.batch_p2p_sync=false
    if nofit "$GL/bridge-1n-pp2ep4.log"; then
      mark "NOFIT bridge-35b-1n-pp2ep4: again with every layer recomputed"
      bridge1 pp2ep4-rc 1 2 model.expert_model_parallel_size=4 model.batch_p2p_sync=false $MEG_RC
    fi ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
mark "ONE NODE 35B DONE $1"
