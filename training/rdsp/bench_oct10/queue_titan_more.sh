#!/bin/bash
# On the H100 box: TorchTitan on the layouts the 2026-10-17 session ran only for Bridge and rdsp
# (results_27b_20261017.md), each recomputed as those runs were:
#   D3     TP2 x PP2 x FSDP 2 + recompute   (against D1 Bridge 17.47 s, D2 rdsp 17.70 s)
#   E3     TP2 x PP4 + recompute            (against E1 24.70 s, E2 22.65 s)
#   C3-rc  TP4 x FSDP 2 + recompute         (against C1 26.95 s, C2 29.26 s; C3 without recompute: 38.16 s)
# Prep as queue_27b_all.sh, without the Bridge checkpoint (no Bridge run here). Each run: a 3-step
# fit check, then 50 steps; timeouts; nvidia-smi samples; "=== END <run> exit <code>" markers.
set -x
cd ~/rdsp
B=~/rdsp/bench_oct10
A=$B/bridge_vs_rdsp/27b_tp4pp2dp1
T=$B/torchtitan
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
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
}
titan_steps() {  # titan_steps <log>: the last step TorchTitan logged
  sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -oE "step: +[0-9]+" | awk '{print $2}' | sort -n | tail -1
}
titan() {  # titan <name> <config>: fit check, then 50 steps if it fits
  local name=$1 config=$2
  run "$name-fit" 40 $T/logs/nvsmi-fit-$config.csv bash $T/scripts/run.sh fit $config
  [ "$(titan_steps $T/logs/fit-$config.log)" = 3 ] || { mark "SKIP $name: no fit"; return; }
  run "$name" 75 $T/logs/nvsmi-run-$config.csv bash $T/scripts/run.sh run $config
}

# ---- prep
bash bench/setup_node.sh > ~/setup.log 2>&1 || { mark "SETUP FAILED"; exit 1; }
(bash $T/scripts/setup_box.sh > ~/tt_setup.log 2>&1) &
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > ~/download.log 2>&1 &
DL=$!
bash $A/scripts/run.sh export
wait $DL
for i in $(seq 1 90); do grep -q "TT SETUP DONE" ~/tt_setup.log && break; sleep 20; done
grep -q "TT SETUP DONE" ~/tt_setup.log || { mark "TT SETUP FAILED"; exit 1; }
mark "PREP DONE"

titan D3 qwen35_27b_cord_tp2pp2dp2_rc
titan E3 qwen35_27b_cord_tp2pp4_rc
titan C3-rc qwen35_27b_cord_tp4dp2_rc
mark "ALL DONE"
