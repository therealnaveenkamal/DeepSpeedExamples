#!/bin/bash
# queue_ep.sh, on node 1 (ops/h100/remote2.sh, after push2.sh): the 35B on two nodes with rdsp's
# language stages splitting their experts (DeepSpeed AutoEP, inside each stage's node) and no
# recompute, against the slide's vision layout, which recomputes every block (6.38 s, 2026-10-18)
# because each GPU there holds all 256 experts of its layers. ../plan.md.
#   P  prep: environments, NeMo container (row export only), 35B weights, the 35B's rows (128 x
#      2048 a step, 50 steps), NCCL check; no Bridge checkpoint
#   C  AutoEP's GPU tests on node 1 (stage-local EP, with and without folding)
#   R0 PP8 x DP2, EP2, nothing recomputed: the measured cut, and the log the vision width is read
#      from (layout_from_log.py vision-width, as on 2026-10-18; not by hand)
#   R1 vision K + language stages of 2, EP2, nothing recomputed (K for stages of 2)
#   R2 vision K + language stages of 4, EP4, nothing recomputed (K for stages of 4)
#      R1/R2 that run out of memory: again with every block recomputed
#   R3 the slide's run again on these boxes: vision 4 + 6 stages of 2, no EP, every block recomputed
# rdsp hangs after a stage runs out of memory: run() stops it when the log says so.
# Markers ("=== ") for ops/h100/guard.sh.
set -x
cd ~/rdsp
. ~/env2.sh
B=~/rdsp/bench_oct10
T=$B/two_node_35b
E=$B/ep_round
EL=$E/logs; mkdir -p "$EL"
BOTH=~/rdsp/bench/two_node/both.sh
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
up() { bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2" > /dev/null 2>&1 || mark "FAILED ray cluster up"; }
down() { bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2" > /dev/null 2>&1; }
cleanup() { pkill -f "[t]rain_vl.py"; $SSH2 "pkill -f '[t]rain_vl.py'" < /dev/null; sleep 10; }
reached() { grep -qaE "^step 3 " "$1" 2>/dev/null; }
oom() { grep -qaE "OutOfMemoryError|CUDA out of memory" "$1" 2>/dev/null; }
run() {  # run <name> <timeout minutes> <log> <command...>: an rdsp run over both nodes, stopped
  # early when its log shows an out-of-memory error
  local name=$1 minutes=$2 log=$3; shift 3
  mark "START $name"
  rm -f "$log"
  $SMI > "$EL/nvsmi-$name-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$EL/nvsmi-$name-node2.csv" &
  local smi2=$!
  "$@" > /dev/null 2>&1 &
  local pid=$! t=0
  while kill -0 $pid 2>/dev/null; do
    if oom "$log"; then sleep 20; mark "OOM $name"; break; fi
    [ $t -ge $((minutes * 60)) ] && { mark "TIMEOUT $name"; break; }
    sleep 15; t=$((t + 15))
  done
  kill -0 $pid 2>/dev/null && cleanup
  wait $pid; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  cleanup; down
  mark "END $name exit $rc median $(python3 "$B/gap_round/scripts/medians.py" rdsp "$log" | awk '{print $1}')"
}

mark "QUEUE START"
# ---- P
SKIP_CONVERT=1 bash "$T/scripts/prep.sh" > "$EL/prep.log" 2>&1
grep -q "^=== PREP DONE" "$EL/prep.log" || { mark "PREP FAILED (ep_round/logs/prep.log)"; tail -20 "$EL/prep.log"; exit 1; }
cat ~/mimo/cord_steps-Qwen3.5-35B-A3B/stats.json; echo   # 129,997 real tokens a step on 2026-10-18
mark "PREP DONE"
down

# ---- C
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; . ~/cuda_env.sh
 python -c "import pytest" 2>/dev/null || pip install -q pytest
 cd ~/rdsp && timeout 900 python -m pytest -q -rs tests/integration/test_heterogeneous_pipeline.py -k autoep) \
  > "$EL/autoep_tests.log" 2>&1
mark "AUTOEP GPU TESTS exit $? $(tail -1 "$EL/autoep_tests.log")"

# ---- R0
export MODEL=Qwen3.5-35B-A3B   # common.sh: GBS 128, STEPS 50, the rows above
R0=$T/pp8dp2/logs/rdsp-ep2-nvrc.log
up; run rdsp-35b-pp8dp2-ep2 25 "$R0" env LANG_EP=2 VISION_RC=0 NAME=-nvrc bash "$T/pp8dp2/run.sh" rdsp
if ! reached "$R0"; then
  mark "R0 again with the vision encoder recomputed (the slide's PP8 x DP2 setting)"
  R0=$T/pp8dp2/logs/rdsp-ep2.log
  up; run rdsp-35b-pp8dp2-ep2-vrc 25 "$R0" env LANG_EP=2 bash "$T/pp8dp2/run.sh" rdsp
fi
K2=$(python3 "$T/scripts/layout_from_log.py" vision-width "$R0" 16 2 2>/dev/null); K2=${K2:-4}
K4=$(python3 "$T/scripts/layout_from_log.py" vision-width "$R0" 16 4 2>/dev/null); K4=${K4:-4}
mark "VISION WIDTH from R0: stages of 2 -> K=$K2, stages of 4 -> K=$K4"

# ---- R1, R2
vision() {  # vision <K> <language stage GPUs> <EP>: the vision layout, nothing recomputed; if it
  # runs out of memory, again with every block recomputed
  local k=$1 sg=$2 ep=$3 s=""; [ "$sg" != 2 ] && s=-s$sg
  local name=rdsp-35b-k$k$s-ep$ep log=$T/vision_stage/logs/rdsp-k$k$s-ep$ep-nvrc.log
  up; run "$name" 25 "$log" env K=$k STAGE_GPUS=$sg LANG_EP=$ep VISION_RC=0 NAME=-nvrc bash "$T/vision_stage/run.sh" rdsp
  if ! reached "$log" && oom "$log"; then
    mark "NOFIT $name: again with every block recomputed"
    up; run "$name-rc" 25 "$T/vision_stage/logs/rdsp-k$k$s-ep$ep-rc-nvrc.log" \
      env K=$k STAGE_GPUS=$sg LANG_EP=$ep RECOMPUTE=1 NAME=-nvrc bash "$T/vision_stage/run.sh" rdsp
  fi
}
vision "$K2" 2 2
[ "$K2" != 4 ] && vision 4 2 2   # the slide's shape too
vision "$K4" 4 4
[ "$K4" != 4 ] && vision 4 4 4

# ---- R3
up; run rdsp-35b-k4-rc-base 25 "$T/vision_stage/logs/rdsp-k4-rc-base.log" \
  env K=4 RECOMPUTE=1 NAME=-base bash "$T/vision_stage/run.sh" rdsp

for f in "$T"/pp8dp2/logs/rdsp-ep2*.log "$T"/vision_stage/logs/rdsp-k*-ep*.log "$T"/vision_stage/logs/rdsp-k4-rc-base.log; do
  mark "MEDIAN $(python3 "$B/gap_round/scripts/medians.py" rdsp "$f" || echo "none $f")"
done
mark "ALL DONE EP"
