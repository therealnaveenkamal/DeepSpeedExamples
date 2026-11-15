#!/bin/bash
# queue7c.sh: replaces the rest of queue7.sh and queue7b.sh (stopped 21:30). Bridge's PP8 run
# timed out: its first iteration compiles for ~10 min and NCCL's default timeout is 10 min, so
# every Bridge run here gets dist.distributed_timeout_minutes=60 (a launch setting, not a speed one).
#   Bridge PP8 even; Bridge PP8 on rdsp's cut; Bridge E sweep (first stage 1, 3, 7; 20 steps);
#   Bridge D sweep (first stage 30, 29; 20 steps); Bridge D recomputing 20 of 32 (fit);
#   the corrected GPU checks
# queue7.sh: "find where rdsp is stronger, and why" (user, 19:35). Takes over from queue5.sh during
# fix-e2-rc12 (which finishes on its own), then:
#   A (rdsp, measured cut) with Bridge's checkpoint converting alongside on GPU 7
#   Bridge on rdsp's measured cuts, full recompute: E 5|20|20|19, D 27|37 -> does the cut alone
#     explain rdsp's lead?
#   C control: rdsp with the fp32 master, 12 steps (box vs remainder master)
#   PP8 (TP1 x PP8, full recompute): rdsp measured cut, Bridge even, Bridge on rdsp's cut ->
#     does the lead keep growing with pipeline depth?
#   Bridge D recomputing 20 of 32 (fit check), the corrected GPU checks
# queue5.sh (from queue4.sh, whose CPU-only conversion failed: Megatron-Bridge sets a CUDA device
# even with --device cpu): Bridge's checkpoint converts during fix-a1 on GPU 7 (context only;
# A peaks ~65 GiB of 79) and 16 CPU cores at nice 19.
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

P8=$B/bridge_vs_rdsp/27b_tp1pp8dp1
TO="dist.distributed_timeout_minutes=60"
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1 $TO"
bridge_fits() { grep -qE "iteration +3/" "$1" 2>/dev/null; }
cut_overrides() {
  python3 - "$1" <<'PY'
import ast, sys
stages = ast.literal_eval(sys.argv[1].split(":", 1)[1].strip())
counts = [b - a for a, b in stages]
first, last, n = counts[0], counts[-1], len(counts)
middle = sum(counts[1:-1])
r = middle % (n - 2) if n > 2 else 0
if r:  # Megatron's middle stages are even: move the last stage's count by the smaller amount
    last = last + r if r <= (n - 2) - r else last - ((n - 2) - r)
print(f"model.num_layers_in_first_pipeline_stage={first}"
      + (f" model.num_layers_in_last_pipeline_stage={last}" if n > 2 else ""))
PY
}
last_cut() { grep -a "stages (blocks)" "$1" | tail -1; }
for i in $(seq 1 60); do pgrep -f "[r]un_recipe.py" >/dev/null || break; sleep 5; done
cleanup

# ---- PP8: Bridge even, Bridge on rdsp's cut
run bridge-p8 50 $P8/logs/nvsmi-bridge-50.csv env NAME=-50 MEG_RC="$RC_ALL" bash $P8/scripts/run.sh bridge
C8=$(cut_overrides "$(last_cut $P8/logs/rdsp-fix-p8.log)"); mark "BRIDGE PP8 CUT $C8"
run bridge-p8-cut 50 $P8/logs/nvsmi-bridge-cut-50.csv env NAME=-cut-50 MEG_RC="$RC_ALL $C8" \
  bash $P8/scripts/run.sh bridge

# ---- Bridge's own best cut: E (last stage 19, middle even), D
for f in 1 3 7; do
  run bridge-e-first$f 35 $E/logs/nvsmi-bridge-first$f.csv env STEPS=20 NAME=-first$f-20 \
    MEG_RC="$RC_ALL model.num_layers_in_first_pipeline_stage=$f model.num_layers_in_last_pipeline_stage=19" \
    bash $E/scripts/run.sh bridge
done
for f in 30 29; do
  run bridge-d-first$f 35 $D/logs/nvsmi-bridge-first$f.csv env STEPS=20 NAME=-first$f-20 \
    MEG_RC="$RC_ALL model.num_layers_in_first_pipeline_stage=$f" bash $D/scripts/run.sh bridge
done

# ---- Bridge D recomputing 20 of 32 (block method), fit check; the corrected GPU checks
RC20="model.recompute_granularity=full model.recompute_method=block model.recompute_num_layers=20 $TO"
run bridge-d-rc20-fit 30 $D/logs/nvsmi-bridge-rc20-fit.csv env STEPS=3 NAME=-rc20-fit MEG_RC="$RC20" \
  bash $D/scripts/run.sh bridge
bridge_fits $D/logs/bridge-rc20-fit.log && mark "FITS Bridge D recomputing 20 of 32" \
  || mark "NOFIT Bridge D recomputing 20 of 32"
run checks2b 15 $F/logs/nvsmi-checks2b.csv bash -c "unset LD_LIBRARY_PATH; . ~/venv/bin/activate; \
  . ~/cuda_env.sh; DS_BUILD_OPS=0 HF_HOME=~/hf python $F/scripts/gpu_checks2.py \
  ~/mimo/cord_steps-Qwen3.5-27B Qwen/Qwen3.5-27B $F/results > $F/logs/gpu_checks2b.log 2>&1"
mark "ALL DONE"
