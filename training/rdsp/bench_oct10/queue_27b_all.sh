#!/bin/bash
# On the H100 box: every 27B run of plan_27b_50step.md, in order, unattended.
#   prep: environment, hardware record, 27B weights + Bridge's exported rows, Bridge checkpoint;
#         TorchTitan's environment builds in the background meanwhile (CPU and network only)
#   A1 Bridge TP4xPP2 · A2 rdsp TP4xPP2 · B1 rdsp vision 1 | TP4 | TP2 | TP1 · C2 rdsp TP4xDP2
#   C1 Bridge TP4xDP2 · B2 MIMO vision-1 (MIMO checkpoint, then the run: expected out of memory)
#   A3 TorchTitan TP4xPP2 · C3 TorchTitan TP4xDP2: a 3-step fit check first, then 50 steps;
#      if the fit check fails, the same with full recompute.
# Each run has a timeout (a stage that runs out of memory can leave the rest waiting), its own
# nvidia-smi samples (every GPU's used memory every 0.5 s) and a marker line "=== END <run> exit <code>".
set -x
cd ~/rdsp
B=~/rdsp/bench_oct10
A=$B/bridge_vs_rdsp/27b_tp4pp2dp1
C=$B/bridge_vs_rdsp/27b_tp4dp2
V=$B/mimo_vs_rdsp/27b_v1
T=$B/torchtitan
G="--optimizer-group-numel 250000000"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  (. ~/venv/bin/activate && ray stop --force >/dev/null 2>&1)
  sudo docker ps -q | xargs -r sudo docker kill >/dev/null 2>&1
  pkill -f "[t]orchrun" ; pkill -f "[t]orchtitan.train"
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
  return $rc
}
titan_steps() {  # titan_steps <log>: the last step TorchTitan logged
  sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -oE "step: +[0-9]+" | awk '{print $2}' | sort -n | tail -1
}
titan() {  # titan <name> <config>: fit check, then 50 steps; with recompute if the fit check fails
  local name=$1 config=$2
  run "$name-fit" 40 $T/logs/nvsmi-fit-$config.csv bash $T/scripts/run.sh fit $config
  if [ "$(titan_steps $T/logs/fit-$config.log)" != 3 ]; then
    config=${config}_rc
    run "$name-fit-rc" 40 $T/logs/nvsmi-fit-$config.csv bash $T/scripts/run.sh fit $config
    [ "$(titan_steps $T/logs/fit-$config.log)" = 3 ] || { mark "SKIP $name: no fit"; return; }
  fi
  run "$name" 75 $T/logs/nvsmi-run-$config.csv bash $T/scripts/run.sh run $config
}

# ---- prep
bash bench/setup_node.sh > ~/setup.log 2>&1 || { mark "SETUP FAILED"; exit 1; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi topo -m; lscpu | head -20; } \
  > $A/logs/hardware-50step.txt 2>&1
(bash $T/scripts/setup_box.sh > ~/tt_setup.log 2>&1) &
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > ~/download.log 2>&1 &
DL=$!
bash $A/scripts/run.sh export
wait $DL
NAME=-50step bash $A/scripts/run.sh convert
mark "PREP DONE"

# ---- A: TP4 x PP2 (vision shares stage 0)
run A1 60 $A/logs/nvsmi-A1-bridge-50.csv env RECOMPUTE=0 GRAD_REDUCE_FP32=false NAME=-bf16grads-50 \
  EXTRA="$(cat $A/scripts/meg_extra_bf16.txt)" bash $A/scripts/run.sh bridge
run A2 45 $A/logs/nvsmi-A2-rdsp-50.csv env RECOMPUTE=0 NAME=-bf16-groups-50 \
  RDSP_ARGS="--grad-dtype bf16 --fused-adam $G" bash $A/scripts/run.sh rdsp

# ---- B1: vision on its own GPU (rdsp)
run B1 50 $V/logs/nvsmi-B1-rdsp-50.csv env STEPS=50 LOSS=token-mean PROFILE=0 RECOMPUTE=1 \
  NAME=-rc-groups-50 EXTRA="$G" bash $V/scripts/rdsp_fit.sh

# ---- C: TP4 x DP2, no pipeline
run C2 60 $C/logs/nvsmi-C2-rdsp-50.csv env NAME=-50 bash $C/scripts/run.sh rdsp
run C1 70 $C/logs/nvsmi-C1-bridge-50.csv env NAME=-50 bash $C/scripts/run.sh bridge

# ---- B2: MIMO vision-1 (fixed fp32 settings, no recompute option: expected out of memory)
run B2-convert 30 $V/logs/nvsmi-B2-convert.csv bash $V/scripts/mimo_fit.sh convert
run B2 20 $V/logs/nvsmi-B2-mimo-50.csv env STEPS=50 NAME=-50 bash $V/scripts/mimo_fit.sh fit

# ---- TorchTitan: A3 (TP4 x PP2), C3 (TP4 x FSDP 2)
for i in $(seq 1 90); do grep -q "TT SETUP DONE" ~/tt_setup.log && break; sleep 20; done
if grep -q "TT SETUP DONE" ~/tt_setup.log; then
  titan A3 qwen35_27b_cord_tp4pp2
  titan C3 qwen35_27b_cord_tp4dp2
else
  mark "SKIP A3 C3: TorchTitan setup failed (~/tt_setup.log)"
fi
mark "ALL DONE"
