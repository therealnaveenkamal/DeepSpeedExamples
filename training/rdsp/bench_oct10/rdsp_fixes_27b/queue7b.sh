#!/bin/bash
# queue7b.sh: after queue7.sh, Bridge's own best cut on E (TP2 x PP4, full recompute): rdsp's
# measured cut 5|20|20|19 gave Bridge 19.39 s (even split 24.70), but it was measured on rdsp's
# stage costs. Sweep the first stage's layers (1, 3, 7; the last keeps 19, the middle stays even),
# 20 steps each (median of steps 6-20; Bridge's first iteration compiles for ~10 min). Then D.
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

RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
for i in $(seq 1 1080); do grep -q "^=== ALL DONE" ~/queue7.out && break; sleep 10; done
for f in 1 3 7; do
  run bridge-e-first$f 30 $E/logs/nvsmi-bridge-first$f.csv env STEPS=20 NAME=-first$f-20 \
    MEG_RC="$RC_ALL model.num_layers_in_first_pipeline_stage=$f model.num_layers_in_last_pipeline_stage=19" \
    bash $E/scripts/run.sh bridge
done
# D (TP2 x PP2 x DP2): rdsp's cut 27|37 ran Bridge out of memory on stage 1 (68.6 GiB by
# PyTorch, 79.2 GiB in use); milder cuts
for f in 30 29; do
  run bridge-d-first$f 30 $D/logs/nvsmi-bridge-first$f.csv env STEPS=20 NAME=-first$f-20 \
    MEG_RC="$RC_ALL model.num_layers_in_first_pipeline_stage=$f" bash $D/scripts/run.sh bridge
done
mark "ALL DONE"
