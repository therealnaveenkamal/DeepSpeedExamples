#!/bin/bash
# On the H100 box: rdsp's four 27B fixes (plan.md), unattended. rdsp only: Bridge's numbers are
# the 2026-10-17 runs (results_27b_20261017.md), same box type, data and settings.
#   1. prep: environment, weights, Bridge's exported rows (no Bridge checkpoint)
#   2. GPU checks (scripts/gpu_checks.py): a fix whose check fails is left out of every run
#   3. D TP2xPP2xDP2: fixes 1+4; + measured cut; + partial recompute (fit 20, then 24 of 32)
#   4. C TP4xDP2 (no pipeline): fixes 1+4, full recompute
#   5. E TP2xPP4: fixes + measured cut; then partial recompute (fit 12 of 16)
#   6. A TP4xPP2 (no recompute): fixes + measured cut
#   7. C partial recompute (fit 48 of 64): rdsp only, so an extra, not a comparison
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
  sudo docker ps -q | xargs -r sudo docker kill >/dev/null 2>&1
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

# ---- 1. prep
bash bench/setup_node.sh > ~/setup.log 2>&1 || { mark "SETUP FAILED"; exit 1; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi topo -m; lscpu | head -20; } \
  > $F/logs/hardware.txt 2>&1
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > ~/download.log 2>&1 &
DL=$!
bash $A/scripts/run.sh export
wait $DL
mark "PREP DONE"

# ---- 2. GPU checks
run checks 25 $F/logs/nvsmi-checks.csv bash -c "unset LD_LIBRARY_PATH; . ~/venv/bin/activate; \
  . ~/cuda_env.sh; DS_BUILD_OPS=0 HF_HOME=~/hf python $F/scripts/gpu_checks.py \
  ~/mimo/cord_steps-Qwen3.5-27B Qwen/Qwen3.5-27B $F/results > $F/logs/gpu_checks.log 2>&1"
FIX=""
[ -f $F/results/varlen.ok ] && FIX="$FIX --vision-attn varlen"
[ -f $F/results/remainder.ok ] && FIX="$FIX --master-dtype remainder"
mark "FIXES:$FIX"

# ---- 3. D TP2xPP2xDP2 (Bridge 17.47 full recompute, 17.05 recomputing 24 of 32; rdsp 17.70)
rd fix-d1 $D 35 RDSP_ARGS="$FIX"
rd fix-d2 $D 40 RDSP_ARGS="$FIX --cuts measured"
for n in 20 24; do
  rd fix-d3-rc$n-fit $D 15 STEPS=3 RC_BLOCKS=$n RDSP_ARGS="$FIX --cuts measured"
  if fits $D/logs/rdsp-fix-d3-rc$n-fit.log; then
    rd fix-d3-rc$n $D 40 RC_BLOCKS=$n RDSP_ARGS="$FIX --cuts measured"
    break
  fi
  mark "NOFIT D recomputing $n of 32"
done

# ---- 4. C TP4xDP2, no pipeline (Bridge 26.95, rdsp 29.26; both full recompute)
rd fix-c1 $C 45 RDSP_ARGS="$FIX"

# ---- 5. E TP2xPP4 (Bridge 24.70, rdsp 22.65; full recompute)
rd fix-e1 $E 45 RDSP_ARGS="$FIX --cuts measured"
rd fix-e2-rc12-fit $E 15 STEPS=3 RC_BLOCKS=12 RDSP_ARGS="$FIX --cuts measured"
if fits $E/logs/rdsp-fix-e2-rc12-fit.log; then
  rd fix-e2-rc12 $E 45 RC_BLOCKS=12 RDSP_ARGS="$FIX --cuts measured"
else
  mark "NOFIT E recomputing 12 of 16"
fi

# ---- 6. A TP4xPP2, no recompute (Bridge 25.31, rdsp 22.78); its run.sh names logs rdsp-equal<NAME>
run fix-a1 45 $A/logs/nvsmi-fix-a1.csv env RECOMPUTE=0 NAME=-fix-a1 \
  RDSP_ARGS="--grad-dtype bf16 --fused-adam $G $FIX --cuts measured" bash $A/scripts/run.sh rdsp

# ---- 7. C, recomputing 48 of 64 (an extra: Bridge has no partial-recompute C run)
rd fix-c2-rc48-fit $C 15 STEPS=3 RC_BLOCKS=48 RDSP_ARGS="$FIX"
if fits $C/logs/rdsp-fix-c2-rc48-fit.log; then
  rd fix-c2-rc48 $C 45 RC_BLOCKS=48 RDSP_ARGS="$FIX"
else
  mark "NOFIT C recomputing 48 of 64"
fi
mark "ALL DONE"
