#!/bin/bash
# one_node_c4.sh <bridge|others>, on one node after queue5.sh exported the 27B's receipt rows in
# NVIDIA's MedPix shape (cord4096_steps-Qwen3.5-27B: 32 rows of 4096 a step, 10 steps): phase F's
# three 27B runs again with only the images changed (receipts, 39% image tokens at 2048, against
# MedPix's 6%). Same settings as one_node_medpix.sh:
#   bridge   Bridge PP8 at its best phase-A cut (cut1), every layer recomputed
#   others   TorchTitan FSDP8 (bf16 reduce, every layer recomputed), then rdsp PP8 (measured cut)
# Logs: bridge_vs_rdsp/27b_tp1pp8dp1/logs/{bridge,rdsp}-cord4096.log,
# torchtitan/logs/run-qwen35_27b_cord_fsdp8_rc_bf16-cord4096.log.
set -uo pipefail
cd ~/rdsp
B=~/rdsp/bench_oct10
P8=$B/bridge_vs_rdsp/27b_tp1pp8dp1
TL=$B/torchtitan/logs
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
export MODEL=Qwen3.5-27B GBS=32 SEQ=4096 DATASET=cord_v2 STEPS=10 EXPORT_NAME=cord4096_steps-Qwen3.5-27B
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]orchrun"; pkill -f "[t]rain_vl.py"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1
  sleep 10
}
run() {  # run <name> <timeout minutes> <log> <command...>: stopped early when the log shows an
  # out-of-memory error (rdsp hangs after one)
  local name=$1 minutes=$2 log=$3; shift 3
  mark "START $name"
  $SMI > "${log%.log}-nvsmi.csv" &
  local smi=$!
  "$@" > /dev/null 2>&1 &
  local pid=$! t=0
  while kill -0 $pid 2>/dev/null; do
    if grep -qaE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null; then
      sleep 20; mark "OOM $name"; break
    fi
    [ $t -ge $((minutes * 60)) ] && { mark "TIMEOUT $name"; break; }
    sleep 15; t=$((t + 15))
  done
  kill -0 $pid 2>/dev/null && cleanup
  wait $pid; local rc=$?
  kill $smi
  cleanup
  mark "END $name exit $rc"
}
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
case "${1:-}" in
  bridge)
    cut="\"model.pipeline_model_parallel_layout='Et|t*9|t*10|t*9|t*9|t*9|t*10|t*7L'\" model.batch_p2p_sync=false"
    rm -f "$P8/logs/bridge-cord4096.log"
    run bridge-27b-cord4096 30 "$P8/logs/bridge-cord4096.log" env NAME=-cord4096 \
      MEG_RC="$RC_ALL $cut $FAST" bash "$P8/scripts/run.sh" bridge ;;
  others)
    rm -f "$TL/run-qwen35_27b_cord_fsdp8_rc_bf16-cord4096.log" "$P8/logs/rdsp-cord4096.log"
    run titan-27b-cord4096 25 "$TL/run-qwen35_27b_cord_fsdp8_rc_bf16-cord4096.log" env NAME=-cord4096 \
      ROWS=32 TT_DCP="$HOME/tt_dcp/Qwen3.5-27B" LOG_RANK=0,7 \
      bash "$B/torchtitan/scripts/run.sh" run qwen35_27b_cord_fsdp8_rc_bf16
    run rdsp-27b-cord4096 25 "$P8/logs/rdsp-cord4096.log" env NAME=-cord4096 \
      DATA="exported:$HOME/mimo/$EXPORT_NAME" \
      RDSP_ARGS="--master-dtype remainder --cuts measured --microbatches 32 --seq 4096" \
      bash "$P8/scripts/run.sh" rdsp ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
mark "ONE NODE CORD4096 DONE $1"
