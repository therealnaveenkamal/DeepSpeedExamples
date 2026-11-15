#!/bin/bash
# On node 1 after queue35c.sh: the two runs that test "rdsp matters most for sparse models", and
# a profile of TorchTitan.
#   0. disk: Bridge's converted 35B checkpoint removed on both nodes (Bridge's runs are done);
#      the dense Qwen3.5-27B weights downloaded on both nodes meanwhile
#   1. TorchTitan FSDP16 (35B-A3B), 15 steps, Nsight on node 1's 8 ranks over steps ~10-13:
#      where its 11.84 s step goes (communication or compute)
#   2. Bridge's rows for the 27B exported at 128 a step (node 1; rdsp's driver reads them there)
#   3. rdsp 27B TP1 x PP8 x DP2 (--cuts measured), as run 1 on the 35B
#   4. rdsp 27B vision stage, K from run 3's measured times (as on the 35B)
# 50 steps each (rdsp); one that runs out of memory before step 3 runs again with every block
# recomputed. Markers for ops/h100/guard.sh.
set -x
cd ~/rdsp
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  bash bench/two_node/both.sh --cleanup
  pkill -f "[t]rain_vl.py"
  sleep 10
  bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv prefix> <command...>
  local name=$1 minutes=$2 csv=$3; shift 3
  mark "START $name"
  $SMI > "$csv-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$csv-node2.csv" &
  local smi2=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  cleanup
  mark "END $name exit $rc"
  return $rc
}
newest() { ls -t "$1"/$2*.log 2>/dev/null | head -1; }  # newest <logs dir> <log prefix>
go27() {  # go27 <name> <layout> <log prefix> <minutes> [VAR=value...]: rdsp on the 27B
  local name=$1 dir=$T/$2 prefix=$3 minutes=$4; shift 4
  run "$name" "$minutes" "$dir/logs-27b/nvsmi-$name" env MODEL=Qwen3.5-27B "$@" bash "$dir/run.sh" rdsp
  local log; log=$(newest "$dir/logs-27b" "$prefix")
  grep -qE "^step 2 " "$log" 2>/dev/null && return 0
  grep -qaE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null \
    || { mark "FAILED $name: not out of memory, not rerun"; return 1; }
  mark "NOFIT $name: again with every block recomputed"
  run "$name-rc" "$minutes" "$dir/logs-27b/nvsmi-$name-rc" \
    env MODEL=Qwen3.5-27B RECOMPUTE=1 "$@" bash "$dir/run.sh" rdsp
}
mkdir -p "$T"/{pp8dp2,vision_stage}/logs-27b "$T/fsdp16/logs" "$T/prep/logs"

# ---- 0. disk, and the 27B weights on both nodes
rm -rf ~/mimo/std/Qwen3.5-35B-A3B ~/mimo/std/Qwen3.5-35B-A3B.done
$SSH2 "rm -rf ~/mimo/std/Qwen3.5-35B-A3B" < /dev/null
DL="unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c \"import huggingface_hub as h; \
h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])\""
bash -c "$DL" > "$T/prep/logs/download27b-node1.log" 2>&1 &
D1=$!
$SSH2 "$DL" < /dev/null > "$T/prep/logs/download27b-node2.log" 2>&1 &
D2=$!
bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"

# ---- 1. TorchTitan FSDP16, profiled
run fsdp16-titan-nsys 30 "$T/fsdp16/logs/nvsmi-fsdp16-titan-nsys" \
  env STEPS=15 NAME=-nsys NSYS_DELAY=450 NSYS_DURATION=36 bash "$T/fsdp16/run.sh" titan
R=$T/fsdp16/logs/nsys-qwen35_35b_a3b_cord_fsdp16-node0
if [ -f "$R.nsys-rep" ]; then
  nsys export --type sqlite --force-overwrite true -o "$R.sqlite" "$R.nsys-rep" > /dev/null 2>&1
  python3 "$T/../bridge_vs_rdsp/27b_tp4pp2dp1/scripts/nsys_stage_busy.py" "$R.sqlite" titan \
    > "$T/fsdp16/logs/nsys-titan-busy.txt" 2>&1
  cat "$T/fsdp16/logs/nsys-titan-busy.txt"
fi
mark "PROFILE DONE"

# ---- 2. the 27B's rows, 128 a step
wait $D1 $D2
env MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config GBS=128 STEPS=50 \
  LOGS="$T/prep/logs" bash bench/mimo/runs.sh export > /dev/null 2>&1
mv "$T/prep/logs/export.log" "$T/prep/logs/export27b.log" 2>/dev/null
[ -f ~/mimo/cord_steps-Qwen3.5-27B/stats.json ] || { mark "FAILED export 27B"; exit 1; }
mark "EXPORT 27B DONE"

# ---- 3-4. rdsp on the 27B: PP8xDP2, then the vision stage
go27 pp8dp2-rdsp-27b pp8dp2 rdsp 60
RUN27=$T/pp8dp2/logs-27b/rdsp.log; [ -f "$T/pp8dp2/logs-27b/rdsp-rc.log" ] && RUN27=$T/pp8dp2/logs-27b/rdsp-rc.log
K=$(python3 "$T/scripts/layout_from_log.py" vision-width "$RUN27" 16)
mark "VISION 27B K=$K"
[ -n "$K" ] && go27 vision-stage-rdsp-27b vision_stage "rdsp-k$K" 60 K="$K"

bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
mark "ALL DONE"
