#!/bin/bash
# queue2.sh: replaces queue.sh after its GPU checks failed (17:56; see scripts/gpu_checks2.py).
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

# ---- 1. fix-d1 (started by queue.sh) to its end, then the checks
for i in $(seq 1 240); do pgrep -f "[t]rain_vl.py" >/dev/null || break; sleep 10; done
pkill -f "[n]vidia-smi --query-gpu=timestamp"; cleanup
mark "END fix-d1 (queue.sh) $(grep -c '^step ' $D/logs/rdsp-fix-d1.log) steps"
run checks2 15 $F/logs/nvsmi-checks2.csv bash -c "unset LD_LIBRARY_PATH; . ~/venv/bin/activate; \
  . ~/cuda_env.sh; DS_BUILD_OPS=0 HF_HOME=~/hf python $F/scripts/gpu_checks2.py \
  ~/mimo/cord_steps-Qwen3.5-27B Qwen/Qwen3.5-27B $F/results > $F/logs/gpu_checks2.log 2>&1"
FIX=""
[ -f $F/results/remainder.ok ] && FIX="--master-dtype remainder"
mark "FIXES: $FIX"

# ---- 2. D TP2xPP2xDP2 (Bridge 17.47 full recompute, 17.05 recomputing 24 of 32; rdsp 17.70)
rd fix-d2 $D 40 RDSP_ARGS="$FIX --cuts measured"
for n in 20 24; do
  rd fix-d3-rc$n-fit $D 15 STEPS=3 RC_BLOCKS=$n RDSP_ARGS="$FIX --cuts measured"
  if fits $D/logs/rdsp-fix-d3-rc$n-fit.log; then
    rd fix-d3-rc$n $D 40 RC_BLOCKS=$n RDSP_ARGS="$FIX --cuts measured"
    break
  fi
  mark "NOFIT D recomputing $n of 32"
done

# ---- 3. C TP4xDP2, no pipeline (Bridge 26.95, rdsp 29.26; both full recompute)
rd fix-c1 $C 45 RDSP_ARGS="$FIX"

# ---- 4. E TP2xPP4 (Bridge 24.70, rdsp 22.65; full recompute)
rd fix-e1 $E 45 RDSP_ARGS="$FIX --cuts measured"
rd fix-e2-rc12-fit $E 15 STEPS=3 RC_BLOCKS=12 RDSP_ARGS="$FIX --cuts measured"
if fits $E/logs/rdsp-fix-e2-rc12-fit.log; then
  rd fix-e2-rc12 $E 45 RC_BLOCKS=12 RDSP_ARGS="$FIX --cuts measured"
else
  mark "NOFIT E recomputing 12 of 16"
fi

# ---- 5. A TP4xPP2, no recompute (Bridge 25.31, rdsp 22.78); its run.sh names logs rdsp-equal<NAME>
run fix-a1 45 $A/logs/nvsmi-fix-a1.csv env RECOMPUTE=0 NAME=-fix-a1 \
  RDSP_ARGS="--grad-dtype bf16 --fused-adam $G $FIX --cuts measured" bash $A/scripts/run.sh rdsp

# ---- 6. C, recomputing 48 of 64 (an extra: Bridge has no partial-recompute C run)
rd fix-c2-rc48-fit $C 15 STEPS=3 RC_BLOCKS=48 RDSP_ARGS="$FIX"
if fits $C/logs/rdsp-fix-c2-rc48-fit.log; then
  rd fix-c2-rc48 $C 45 RC_BLOCKS=48 RDSP_ARGS="$FIX"
else
  mark "NOFIT C recomputing 48 of 64"
fi
mark "ALL DONE"
