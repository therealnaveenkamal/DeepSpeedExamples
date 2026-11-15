#!/bin/bash
# queue8.sh: after queue7d.sh's ALL DONE, the other systems' home regimes on one node (user 20:00:
# "single node complete first"):
#   TorchTitan's: FSDP over 8 GPUs (27B), and rdsp in the same regime (one stage, ZeRO-3)
#   MegatronMIMO's: vision encoder on its own GPU (9B, where MIMO fits one node): both language
#     layouts of 2026-10-11, MIMO and rdsp, rerun on this box
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

T=$B/torchtitan
Z=$B/bridge_vs_rdsp/27b_dp8
M=$B/mimo_vs_rdsp
titan_steps() { sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "step: +[0-9]+" | awk '{print $2}' | sort -n | tail -1; }
titan() {  # titan <name> <config>: 3-step fit check, then 50 steps if it fits
  local name=$1 config=$2
  run "$name-fit" 40 $T/logs/nvsmi-fit-$config.csv bash $T/scripts/run.sh fit $config
  [ "$(titan_steps $T/logs/fit-$config.log)" = 3 ] || { mark "NOFIT $name ($config)"; return; }
  run "$name" 80 $T/logs/nvsmi-run-$config.csv bash $T/scripts/run.sh run $config
}
for i in $(seq 1 1440); do grep -q "^=== ALL DONE" ~/queue7d.out && break; sleep 10; done
grep -q "TT SETUP DONE" ~/tt_setup.log || bash $T/scripts/setup_box.sh > ~/tt_setup2.log 2>&1

# ---- why the remainder master is 5% slower on C (31.29 vs 29.74 s, same box): per-phase timers
for m in fp32 remainder; do
  rd fix-c-prof-$m $C 20 STEPS=6 RDSP_ARGS="--master-dtype $m --profile"
done

# ---- TorchTitan's regime, 27B: FSDP over 8 GPUs; rdsp one stage with ZeRO-3
rd fix-z3-fit $Z 25 STEPS=3
if fits $Z/logs/rdsp-fix-z3-fit.log; then rd fix-z3 $Z 60; else mark "NOFIT rdsp ZeRO-3 DP8"; fi
titan T8-norc qwen35_27b_cord_fsdp8
titan T8 qwen35_27b_cord_fsdp8_rc

# ---- MegatronMIMO's regime, 9B: vision on its own GPU, language on 4 (2026-10-11 layouts)
grep -q "DL9B DONE 0" ~/download9b.log || bash ~/dl9b.sh > ~/download9b-2.log 2>&1
run export-9b 30 $M/logs-nvsmi-export-9b.csv env MODEL=Qwen3.5-9B bash bench/mimo/runs.sh export
run convert-9b 30 $M/logs-nvsmi-convert-9b.csv bash $M/scripts/run.sh convert
for L in 9b_v1_tp2pp2dp1 9b_v1_tp2pp1dp2; do
  for s in mimo rdsp; do
    run $s-$L 35 $M/$L/logs/nvsmi-$s-oct17.csv bash $M/scripts/run.sh $s $L
    mv $M/$L/logs/$s.log $M/$L/logs/$s-oct17.log
  done
done
mark "ALL DONE"
