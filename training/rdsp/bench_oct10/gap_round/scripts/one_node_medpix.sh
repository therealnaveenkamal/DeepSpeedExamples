#!/bin/bash
# one_node_medpix.sh <bridge|others>: phase F of ../plan.md, the dense 27B on one node (8 GPUs) at
# NVIDIA's published fine-tuning setup (MedPix, 32 rows of 4096 a step, 50 steps), each system at
# its fastest phase-A setting (slide 14's layouts). queue3.sh starts "bridge" on node 1 and
# "others" on node 2 at the same time; cleanup touches this node only.
#   bridge  Megatron-Bridge TP1 x PP8 at its fastest phase-A cut (cut1, cut0 or vp2; 3 | 9x6 | 7
#           if none of them ran), every layer recomputed
#   others  TorchTitan at its fastest phase-A setting (FSDP8 variants or PP8), then rdsp TP1 x PP8
#           with its measured cut
set -uo pipefail
cd ~/rdsp
B=~/rdsp/bench_oct10
G=$B/gap_round
P8=$B/bridge_vs_rdsp/27b_tp1pp8dp1
TL=$B/torchtitan/logs
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
export MODEL=Qwen3.5-27B GBS=32 SEQ=4096 DATASET=medpix EXPORT_NAME=medpix_steps-Qwen3.5-27B
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]orchrun"; pkill -f "[t]rain_vl.py"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1
  sleep 10
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  $SMI > "$csv" &
  local smi=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi
  cleanup
  mark "END $name exit $rc"
}
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
layout() { echo "\"model.pipeline_model_parallel_layout='$1'\""; }
case "${1:-}" in
  bridge)
    best=$(python3 "$G/scripts/medians.py" --best bridge "$P8"/logs/bridge-gap-cut1.log \
      "$P8"/logs/bridge-gap-cut0.log "$P8"/logs/bridge-gap-vp2.log | awk '{print $2}')
    case "$best" in
      *cut1.log) cut="$(layout 'Et|t*9|t*10|t*9|t*9|t*9|t*10|t*7L') model.batch_p2p_sync=false" ;;
      *cut0.log) cut="$(layout 'E|t*9|t*10|t*10|t*10|t*9|t*9|t*7L') model.batch_p2p_sync=false" ;;
      *vp2.log)  cut="$(layout 'E|t*5|t*5|t*5|t*5|t*5|t*5|t*4|t|t*4|t*5|t*4|t*4|t*4|t*5|t*3L') model.overlap_p2p_comm=true model.batch_p2p_comm=false" ;;
      *)         cut="model.num_layers_in_first_pipeline_stage=3 model.num_layers_in_last_pipeline_stage=7" ;;
    esac
    mark "BRIDGE 27B MEDPIX CUT from ${best:-none}"
    run bridge-27b-medpix 60 "$P8/logs/nvsmi-bridge-medpix.csv" env NAME=-medpix \
      MEG_RC="$RC_ALL $cut $FAST" bash "$P8/scripts/run.sh" bridge ;;
  others)
    best=$(python3 "$G/scripts/medians.py" --best titan "$TL"/run-qwen35_27b_cord_fsdp8_rc-gap.log \
      "$TL"/run-qwen35_27b_cord_fsdp8_rc_bf16-gap.log "$TL"/run-qwen35_27b_cord_fsdp8_ffnrc_bf16-gap.log \
      "$TL"/run-qwen35_27b_cord_pp8_rc-gap.log | awk '{print $2}')
    cfg=$(basename "${best:-run-qwen35_27b_cord_fsdp8_rc_bf16-gap.log}" -gap.log); cfg=${cfg#run-}
    mark "TITAN 27B MEDPIX CONFIG $cfg"
    run "titan-27b-medpix-$cfg" 40 "$TL/nvsmi-medpix-$cfg.csv" env NAME=-medpix ROWS=32 \
      TT_DCP="$HOME/tt_dcp/Qwen3.5-27B" LOG_RANK=0,7 bash "$B/torchtitan/scripts/run.sh" run "$cfg"
    run rdsp-27b-medpix 40 "$P8/logs/nvsmi-rdsp-medpix.csv" env NAME=-medpix \
      DATA="exported:$HOME/mimo/$EXPORT_NAME" \
      RDSP_ARGS="--master-dtype remainder --cuts measured --microbatches 32 --seq 4096" bash "$P8/scripts/run.sh" rdsp ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
mark "ONE NODE MEDPIX DONE $1"
