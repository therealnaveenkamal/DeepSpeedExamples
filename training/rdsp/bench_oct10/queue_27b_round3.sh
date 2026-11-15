#!/bin/bash
# On the H100 box: the third 27B round (plan_27b_50step.md, "Round 3"), unattended.
#   1. prep: environment, weights, Bridge's exported rows, Bridge checkpoint; TorchTitan builds alongside
#   2. no-recompute fit checks (3 steps), Bridge and rdsp on C TP4xDP2, D TP2xPP2xDP2, E TP2xPP4;
#      any that fits gets 50 steps
#   3. rdsp D: ZeRO-2 (50 steps), then a profile: 10 steps with rdsp's timers, 12 under Nsight Systems
#   4. Bridge D with partial recompute (block method, 24 then 28 of 32 layers per stage)
#   5. TorchTitan: no-recompute fit checks on D and E, then D, E and C with recompute (fit, 50 steps)
# Each run: timeout, nvidia-smi samples (every GPU's used memory every 0.5 s), "=== END <run> exit <code>".
set -x
cd ~/rdsp
B=~/rdsp/bench_oct10
A=$B/bridge_vs_rdsp/27b_tp4pp2dp1
C=$B/bridge_vs_rdsp/27b_tp4dp2
D=$B/bridge_vs_rdsp/27b_tp2pp2dp2
E=$B/bridge_vs_rdsp/27b_tp2pp4dp1
T=$B/torchtitan
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
}
rdsp_fits() { grep -qE "^step 2 " "$1" 2>/dev/null; }
bridge_fits() { grep -qE "iteration +3/" "$1" 2>/dev/null; }
titan_steps() { sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "step: +[0-9]+" | awk '{print $2}' | sort -n | tail -1; }

norc() {  # norc <name> <layout dir> <rdsp|bridge>: 3 steps without recompute; 50 steps if it fits
  local name=$1 dir=$2 sys=$3 log minutes=20
  # rdsp can hang (not exit) when a stage runs out of memory in a forward pass: a short timeout
  [ $sys = rdsp ] && minutes=12
  run "$name-norc-fit" $minutes $dir/logs/nvsmi-$name-norc-fit.csv \
    env RECOMPUTE=0 STEPS=3 NAME=-norc-fit bash $dir/scripts/run.sh $sys
  log=$dir/logs/$sys-norc-fit.log
  if { [ $sys = rdsp ] && rdsp_fits $log; } || { [ $sys = bridge ] && bridge_fits $log; }; then
    run "$name-norc" 50 $dir/logs/nvsmi-$name-norc-50.csv env RECOMPUTE=0 NAME=-norc-50 bash $dir/scripts/run.sh $sys
  else
    mark "NOFIT $name without recompute"
  fi
}
titan() {  # titan <name> <config>: 3-step fit check, then 50 steps if it fits
  local name=$1 config=$2
  run "$name-fit" 40 $T/logs/nvsmi-fit-$config.csv bash $T/scripts/run.sh fit $config
  [ "$(titan_steps $T/logs/fit-$config.log)" = 3 ] || { mark "NOFIT $name ($config)"; return; }
  run "$name" 80 $T/logs/nvsmi-run-$config.csv bash $T/scripts/run.sh run $config
}

# ---- 1. prep
bash bench/setup_node.sh > ~/setup.log 2>&1 || { mark "SETUP FAILED"; exit 1; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi topo -m; lscpu | head -20; } \
  > $A/logs/hardware-round3.txt 2>&1
(bash $T/scripts/setup_box.sh > ~/tt_setup.log 2>&1) &
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > ~/download.log 2>&1 &
DL=$!
bash $A/scripts/run.sh export
wait $DL
NAME=-round3 bash $A/scripts/run.sh convert
mark "PREP DONE"

# ---- 2. no-recompute fit checks
norc C2 $C rdsp
norc D2 $D rdsp
norc E2 $E rdsp
norc C1 $C bridge
norc D1 $D bridge
norc E1 $E bridge

# ---- 3. rdsp D: ZeRO-2, then the profile
run D2-zero2 45 $D/logs/nvsmi-D2-zero2-50.csv env ZERO=2 NAME=-zero2-50 bash $D/scripts/run.sh rdsp
mkdir -p $D/profiles
t0=$(date +%s)
run D2-profile10 25 $D/logs/nvsmi-D2-profile10.csv \
  env STEPS=10 NAME=-profile10 RDSP_ARGS=--profile bash $D/scripts/run.sh rdsp &
P=$!
until grep -qE "^step 3 " $D/logs/rdsp-profile10.log 2>/dev/null; do kill -0 $P 2>/dev/null || break; sleep 2; done
STEADY=$(( $(date +%s) - t0 )); echo $STEADY > $D/profiles/steady-rdsp.s
wait $P
if command -v nsys >/dev/null; then
  run D2-nsys 30 $D/logs/nvsmi-D2-nsys.csv env STEPS=12 NAME=-nsys \
    PROFILER="nsys profile -t cuda,nvtx -s none --cpuctxsw=none --delay $STEADY --duration 40 --force-overwrite true -o $D/profiles/rdsp-d2" \
    bash $D/scripts/run.sh rdsp
  # the container's nsys is newer than the host's: export there
  sudo docker run --gpus all --rm --ipc=host -v "$D/profiles:/p" nvcr.io/nvidia/nemo:26.08 bash -c \
    "nsys export --type sqlite --force-overwrite true -o /p/rdsp-d2.sqlite /p/rdsp-d2.nsys-rep"
  sudo chown -R "$(id -u):$(id -g)" $D/profiles
  mark "D2 PROFILE DONE"
else
  mark "SKIP D2-nsys: no nsys on the host"
fi

# ---- 4. Bridge D with partial recompute: the first N layers of each stage (block method)
for n in 24 28; do
  RC_N="model.recompute_granularity=full model.recompute_method=block model.recompute_num_layers=$n"
  run D1-rc$n-fit 25 $D/logs/nvsmi-D1-rc$n-fit.csv env STEPS=3 NAME=-rc$n-fit MEG_RC="$RC_N" bash $D/scripts/run.sh bridge
  if bridge_fits $D/logs/bridge-rc$n-fit.log; then
    run D1-rc$n 50 $D/logs/nvsmi-D1-rc$n-50.csv env NAME=-rc$n-50 MEG_RC="$RC_N" bash $D/scripts/run.sh bridge
    break
  fi
  mark "NOFIT D1 recomputing $n of 32 layers"
done

# ---- 5. TorchTitan
for i in $(seq 1 90); do grep -q "TT SETUP DONE" ~/tt_setup.log && break; sleep 20; done
if grep -q "TT SETUP DONE" ~/tt_setup.log; then
  titan D3-norc qwen35_27b_cord_tp2pp2dp2
  titan E3-norc qwen35_27b_cord_tp2pp4
  titan D3 qwen35_27b_cord_tp2pp2dp2_rc
  titan E3 qwen35_27b_cord_tp2pp4_rc
  titan C3-rc qwen35_27b_cord_tp4dp2_rc
else
  mark "SKIP TorchTitan: setup failed (~/tt_setup.log)"
fi
mark "ALL DONE"
