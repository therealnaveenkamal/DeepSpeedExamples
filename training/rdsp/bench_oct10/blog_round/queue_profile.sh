#!/bin/bash
# queue_profile.sh, on the node after prep.sh (run it with ops/h100/remote.sh): the diagnosis
# session of plan.md. One node, 8 x H100.
#   E2  27B, eight stages of 1 GPU, 50 steps unprofiled: rdsp (measured cut), then Megatron-Bridge
#       at the cut rdsp just measured (October's cut if that one does not fit)
#   E1  Megatron-Bridge at that cut: its own per-stage timers (15 steps), then py-spy (15 steps)
#   E3  9B, 1 vision GPU + language TP2 x DP2: MIMO and rdsp, 50 steps unprofiled, then py-spy of both
#   T1  rdsp on that layout again with RDSP_DIRECT_GRAD_ADD=1 (the gradient-accumulation trial)
#   O1  rdsp on the same 9B layout under Nsight Systems with call stacks (RUN_BT=0 skips);
#       only with RUN_O1=1: plain Nsight traces of both systems
# Profiled runs are diagnosis: only the 50-step runs give step times. TAG (default: today, e.g.
# nov12) suffixes every log, so October's files stay. Markers: "=== START/END <run>", medians as
# "=== MEDIAN ...", and "=== ALL DONE BLOG" last.
set -uo pipefail
cd ~/rdsp
B=~/rdsp/bench_oct10
P8=$B/bridge_vs_rdsp/27b_tp1pp8dp1
M=$B/mimo_vs_rdsp
L9=9b_v1_tp2pp1dp2
TAG=${TAG:-$(date -u +%b%d | tr 'A-Z' 'a-z')}
MED="python3 $B/gap_round/scripts/medians.py"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
  pkill -f "[t]orch.distributed.run"; pkill -f "[t]orchrun"; pkill -f "[t]rain_vl.py"; sudo pkill -f "[p]y-spy record"
  bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1
  sleep 10
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  $SMI > "$csv" &
  local smi=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi
  cleanup
  mark "END $name exit $rc"
}
median() { mark "MEDIAN $1 $($MED "$2" "$3" | awk '{print $1}') s ($3)"; }

# ---- 27B, eight stages ----
# Bridge: every layer recomputed, NVIDIA's perf recipes' check flags off, 8 data workers, no sync
# after each batched send/receive: the settings of October's 14.34 s run (gap_round/scripts/one_node.sh)
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
OCT_CUT="Et|t*9|t*10|t*9|t*9|t*9|t*10|t*7L"
layout() { echo "\"model.pipeline_model_parallel_layout='$1'\""; }   # reaches Hydra through runs.sh's bash -c
bridge() {  # bridge <mode: bridge|timers|pyspy> <log suffix> <cut> <timeout minutes>
  run "bridge-27b-$1$2" "$4" "$P8/logs/nvsmi-$1-bridge$2.csv" env NAME="$2" \
    MEG_RC="$RC_ALL $(layout "$3") $FAST model.batch_p2p_sync=false" bash "$P8/scripts/run.sh" "$1"
}
cut_of() {  # cut_of <rdsp log>: rdsp's final cut as a Megatron layout string
  python3 - "$1" <<'PY'
import ast, re, sys
found = re.findall(r"^stages \(blocks\): (\[.*\])\s*$", open(sys.argv[1], errors="replace").read(), re.M)
sizes = [stop - start for start, stop in ast.literal_eval(found[-1])]
part = lambda n: "" if n == 0 else "t" if n == 1 else f"t*{n}"
assert len(sizes) == 8 and sum(sizes) == 64 and all(n > 0 for n in sizes[1:]), sizes
print("|".join(["E" + part(sizes[0])] + [part(n) for n in sizes[1:-1]] + [part(sizes[-1]) + "L"]))
PY
}
mkdir -p "$P8/logs" "$P8/profiles"
mark "QUEUE START tag $TAG"

run rdsp-27b 40 "$P8/logs/nvsmi-rdsp-$TAG.csv" env NAME="-$TAG" \
  RDSP_ARGS="--master-dtype remainder --cuts measured" bash "$P8/scripts/run.sh" rdsp
median "rdsp 27B" rdsp "$P8/logs/rdsp-$TAG.log"
CUT=$(cut_of "$P8/logs/rdsp-$TAG.log") || CUT=$OCT_CUT
mark "CUT rdsp measured: $CUT (October: $OCT_CUT)"

bridge bridge "-$TAG" "$CUT" 60
if ! grep -qE "iteration +50/" "$P8/logs/bridge-$TAG.log" && [ "$CUT" != "$OCT_CUT" ]; then
  mark "NOFIT Bridge at $CUT: again at October's cut"
  CUT=$OCT_CUT
  mv "$P8/logs/bridge-$TAG.log" "$P8/logs/bridge-$TAG-newcut-failed.log"
  bridge bridge "-$TAG" "$CUT" 60
fi
median "bridge 27B at $CUT" bridge "$P8/logs/bridge-$TAG.log"

bridge timers "-$TAG" "$CUT" 30
bridge pyspy "-$TAG" "$CUT" 30

# ---- 9B, 1 vision GPU + language TP2 x DP2 ----
mkdir -p "$M/$L9/logs"
run mimo-9b 30 "$M/$L9/logs/nvsmi-mimo-$TAG.csv" bash "$M/scripts/run.sh" mimo $L9
mv "$M/$L9/logs/mimo.log" "$M/$L9/logs/mimo-$TAG.log"
median "mimo 9B" bridge "$M/$L9/logs/mimo-$TAG.log"
run rdsp-9b 30 "$M/$L9/logs/nvsmi-rdsp-$TAG.csv" bash "$M/scripts/run.sh" rdsp $L9
mv "$M/$L9/logs/rdsp.log" "$M/$L9/logs/rdsp-$TAG.log"
median "rdsp 9B" rdsp "$M/$L9/logs/rdsp-$TAG.log"
# Trial (approved 2026-11-11): the same run with the gradient added straight into the fp32 buffer
# (one GPU kernel per parameter per microbatch instead of two; same bits). The log must show
# "rdsp: direct gradient add on" from the stage workers, else the switch did not reach them.
run rdsp-9b-directadd 30 "$M/$L9/logs/nvsmi-rdsp-$TAG-directadd.csv" env RDSP_DIRECT_GRAD_ADD=1 \
  bash "$M/scripts/run.sh" rdsp $L9
mv "$M/$L9/logs/rdsp.log" "$M/$L9/logs/rdsp-$TAG-directadd.log"
median "rdsp 9B direct add" rdsp "$M/$L9/logs/rdsp-$TAG-directadd.log"
mark "DIRECT ADD lines in log: $(grep -c "direct gradient add on" "$M/$L9/logs/rdsp-$TAG-directadd.log")"
run pyspy-9b 40 "$M/$L9/logs/nvsmi-pyspy-$TAG.csv" env TAG="-$TAG" bash "$M/scripts/run.sh" pyspy $L9
# O1, narrowed after the October traces were re-read (2026-11-11): they name the gradient
# accumulation but not the source of rdsp's ~47,500 extra small GPU operations a step, so this
# run records the call stack of every launch. RUN_BT=0 skips it.
if [ "${RUN_BT:-1}" = 1 ]; then
  run bt-9b 25 "$M/$L9/logs/nvsmi-bt-$TAG.csv" env TAG="-$TAG" bash "$M/scripts/run.sh" backtrace $L9
fi
if [ "${RUN_O1:-0}" = 1 ]; then
  run nsys-9b 45 "$M/$L9/logs/nvsmi-nsys-$TAG.csv" env TAG="-$TAG" bash "$M/scripts/run.sh" nsys $L9
fi
mark "ALL DONE BLOG"
