#!/bin/bash
# queue11.sh: why rdsp is 24% faster than Bridge on TP4 x PP2 (no recompute) even at Bridge's best
# tried cut (user, 03:00): per-stage timers on both at the same cut 28 | 36, and rdsp at its own
# measured cut. Diagnosis only: profiled step times are not results.
# queue9.sh: gaps in the single-node map, while the table is written (2026-10-18 02:05):
#   rdsp PP8 recomputing 8 of each stage's layers (one keeps its activations): fit, then 50 steps
#   Bridge A (TP4 x PP2, no recompute) first-stage sweep 30, 28 (rdsp's measured cut: 25 | 39)
#   Bridge with Megatron FSDP over 8 GPUs (TorchTitan's regime), 3-step fit check
# queue8b.sh: the 9B part of queue8.sh again. queue8.sh's 9B export used a recipe name that does
# not exist for 9B, and the rerun (qwen35_vl_9b_sft_4gpu_h100_bf16_config, as on 2026-10-11) was
# killed at 33 of 50 steps by a queue cleanup. Export alone, then the four 9B runs.
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

FIX="--master-dtype remainder"
TO="dist.distributed_timeout_minutes=60"
run prof-a-rdsp-cut28 25 $A/logs/nvsmi-prof-rdsp-cut28.csv env RECOMPUTE=0 STEPS=8 NAME=-prof-cut28 \
  RDSP_ARGS="--grad-dtype bf16 --fused-adam $G $FIX --cuts 28 --profile" bash $A/scripts/run.sh rdsp
run prof-a-bridge-cut28 40 $A/logs/nvsmi-prof-bridge-cut28.csv env RECOMPUTE=0 GRAD_REDUCE_FP32=false STEPS=12 \
  NAME=-prof-cut28 EXTRA="$(cat $A/scripts/meg_extra_bf16.txt) model.num_layers_in_first_pipeline_stage=28 $TO logger.timing_log_level=2 logger.timing_log_option=all" \
  bash $A/scripts/run.sh bridge
run prof-a-rdsp-measured 25 $A/logs/nvsmi-prof-rdsp-measured.csv env RECOMPUTE=0 STEPS=8 NAME=-prof-measured \
  RDSP_ARGS="--grad-dtype bf16 --fused-adam $G $FIX --cuts measured --profile" bash $A/scripts/run.sh rdsp
mark "ALL DONE"
