#!/bin/bash
# queue4.sh: takes over from queue3.sh during fix-c1 (stopped there; fix-c1 finishes on its own),
# replacing queue3's last rdsp-only extra (C recomputing 48 of 64) with a Bridge D fit check
# recomputing 20 of 32 (user, 18:45). Bridge's checkpoint converts on the CPU (16 cores, nice 19)
# while fix-c1 runs. queue3.sh (from queue2.sh): the remainder master is used as checked by hand at 18:05 (gpu_checks2's
# 'fail' was its own measurement order). queue2.sh replaced queue.sh after its GPU checks failed (17:56; see scripts/gpu_checks2.py).
# queue.sh was stopped during its first run, fix-d1 (no fixes: a same-box baseline), which
# finishes on its own. Then, unattended, rdsp only:
#   1. wait for fix-d1, then the corrected GPU checks (scripts/gpu_checks2.py); the remainder
#      master is used only if its check passes; vision varlen is reported, not used
#   2. D TP2xPP2xDP2: remainder + measured cut; + partial recompute (fit 20, then 24 of 32)
#   3. C TP4xDP2 (no pipeline): remainder, full recompute
#   4. E TP2xPP4: remainder + measured cut; then partial recompute (fit 12 of 16)
#   5. A TP4xPP2 (no recompute): remainder + measured cut
#   6. C partial recompute (fit 48 of 64): rdsp only, an extra, not a comparison
# Each run: timeout, nvidia-smi samples (every GPU's used memory every 0.5 s), "=== END <run> exit <code>".
set -x
cd ~/rdsp
B=~/rdsp/bench_oct10
A=$B/bridge_vs_rdsp/27b_tp4pp2dp1
C=$B/bridge_vs_rdsp/27b_tp4dp2
D=$B/bridge_vs_rdsp/27b_tp2pp2dp2
E=$B/bridge_vs_rdsp/27b_tp2pp4dp1
F=$B/rdsp_fixes_27b
G="--optimizer-group-numel 250000000"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  (. ~/venv/bin/activate && ray stop --force >/dev/null 2>&1)
  # every container but the CPU conversion, which runs across fix-c1, E and A
  sudo docker ps --format '{{.ID}} {{.Names}}' | awk '$2 != "bridge-convert" {print $1}' \
    | xargs -r sudo docker kill >/dev/null 2>&1
  sleep 10
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > "$csv" &
  local smi=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi; cleanup
  mark "END $name exit $rc"
}
rd() {  # rd <name> <layout dir> <timeout minutes> [VAR=value...]: rdsp via the layout's run.sh
  local name=$1 dir=$2 minutes=$3; shift 3
  run "$name" "$minutes" "$dir/logs/nvsmi-$name.csv" env NAME="-$name" "$@" bash "$dir/scripts/run.sh" rdsp
}
fits() { grep -qE "^step 2 " "$1" 2>/dev/null; }  # fits <log>

FIX="--master-dtype remainder"
bridge_fits() { grep -qE "iteration +3/" "$1" 2>/dev/null; }

# ---- Bridge's checkpoint, on the CPU while fix-c1 uses the GPUs
(sudo docker run --rm --name bridge-convert --ipc=host --shm-size=64g --cpuset-cpus 176-191 \
   -v "$HOME/hf:/root/.cache/huggingface" -v "$HOME/mimo:/workspace" \
   -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge \
   nvcr.io/nvidia/nemo:26.08 nice -n 19 python scripts/conversion/setup_conversion.py import \
   --hf-model Qwen/Qwen3.5-27B --megatron-path /workspace/std/Qwen3.5-27B \
   > $A/logs/convert-fixes.log 2>&1; echo "convert exit $?" >> $A/logs/convert-fixes.log;
 sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo") &
CONV=$!

# ---- fix-c1 (started by queue3.sh) to its end
for i in $(seq 1 300); do pgrep -f "[t]rain_vl.py" >/dev/null || break; sleep 10; done
pkill -f "[n]vidia-smi --query-gpu=timestamp"; cleanup
mark "END fix-c1 (queue3.sh) $(grep -c '^step ' $C/logs/rdsp-fix-c1.log) steps"

# ---- E TP2xPP4 (Bridge 24.70, rdsp 22.65; full recompute)
rd fix-e1 $E 45 RDSP_ARGS="$FIX --cuts measured"
rd fix-e2-rc12-fit $E 15 STEPS=3 RC_BLOCKS=12 RDSP_ARGS="$FIX --cuts measured"
if fits $E/logs/rdsp-fix-e2-rc12-fit.log; then
  rd fix-e2-rc12 $E 45 RC_BLOCKS=12 RDSP_ARGS="$FIX --cuts measured"
else
  mark "NOFIT E recomputing 12 of 16"
fi

# ---- A TP4xPP2, no recompute (Bridge 25.31, rdsp 22.78); its run.sh names logs rdsp-equal<NAME>
run fix-a1 45 $A/logs/nvsmi-fix-a1.csv env RECOMPUTE=0 NAME=-fix-a1 \
  RDSP_ARGS="--grad-dtype bf16 --fused-adam $G $FIX --cuts measured" bash $A/scripts/run.sh rdsp

# ---- Bridge D recomputing 20 of 32 (block method, as its 24 of 32 run): 3-step fit check,
#      50 steps if it fits
wait $CONV
if ! grep -q "convert exit 0" $A/logs/convert-fixes.log; then
  mark "CPU CONVERT FAILED: converting with the GPUs"
  NAME=-fixes2 bash $A/scripts/run.sh convert
fi
mark "BRIDGE CHECKPOINT $(ls ~/mimo/std/Qwen3.5-27B | head -3 | tr '\n' ' ')"
RC20="model.recompute_granularity=full model.recompute_method=block model.recompute_num_layers=20"
run bridge-d-rc20-fit 25 $D/logs/nvsmi-bridge-rc20-fit.csv env STEPS=3 NAME=-rc20-fit MEG_RC="$RC20" \
  bash $D/scripts/run.sh bridge
if bridge_fits $D/logs/bridge-rc20-fit.log; then
  run bridge-d-rc20 45 $D/logs/nvsmi-bridge-rc20-50.csv env NAME=-rc20-50 MEG_RC="$RC20" \
    bash $D/scripts/run.sh bridge
else
  mark "NOFIT Bridge D recomputing 20 of 32"
fi

# ---- the corrected GPU checks, for the record (not gating anything)
run checks2b 15 $F/logs/nvsmi-checks2b.csv bash -c "unset LD_LIBRARY_PATH; . ~/venv/bin/activate; \
  . ~/cuda_env.sh; DS_BUILD_OPS=0 HF_HOME=~/hf python $F/scripts/gpu_checks2.py \
  ~/mimo/cord_steps-Qwen3.5-27B Qwen/Qwen3.5-27B $F/results > $F/logs/gpu_checks2b.log 2>&1"
mark "ALL DONE"
