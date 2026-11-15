#!/bin/bash
# Phase 2 on node 1 of the two-node pair, after queue35.sh (ops/h100/remote2.sh with this file).
# The layouts that check the 27B claims, without expert parallelism (the 27B is dense):
#   1. rdsp ZeRO-3 over 16, against TorchTitan FSDP16 (queue35.sh).
#   2. Bridge PP8xDP2 again, the container's own library path restored (runs.sh: the multi-node
#      flags had replaced it, dropping its CUDA forward-compatibility library; the first attempt
#      deadlocked across the node boundary), NCCL INFO logs; then at the cut nearest rdsp's.
#   3. Bridge TP4xDP4, experts split over TP (no EP): Bridge's ground, as on the 27B.
# (TCP runs between the nodes, NET=socket: dropped 2026-10-18.)
# Each run: 50 steps; one that runs out of memory before step 3 runs again with
# every block recomputed. SKIP / ~/queue35b.skip: run names to leave out. Markers for
# ops/h100/guard.sh.
set -x
cd ~/rdsp
. ~/env2.sh
SKIP=${SKIP:-$(cat ~/queue35b.skip 2>/dev/null)}
T=~/rdsp/bench_oct10/two_node_35b
P=$T/pp8dp2
V=$T/vision_stage
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {  # every container, torchrun and rdsp driver on both nodes; a fresh Ray cluster
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
newest() { ls -t "$1"/logs/$2*.log 2>/dev/null | head -1; }  # newest <layout dir> <log prefix>
reached() {  # reached <log> <rdsp|bridge|titan>: the run got past step 2
  [ -f "$1" ] || return 1
  case $2 in
    rdsp)   grep -qE "^step 2 " "$1" ;;
    bridge) grep -qE "iteration +3/" "$1" || [ "$(grep -caE 'Step Time : ' "$1")" -ge 3 ] ;;
    titan)  [ "$(sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -oE "step: +[0-9]+" | awk '{print $2}' \
              | sort -n | tail -1)" -ge 3 ] 2>/dev/null ;;
  esac
}
go() {  # go <name> <rdsp|bridge|titan> <layout dir> <log prefix> <minutes> <run.sh case> [VAR=value...]
  local name=$1 sys=$2 dir=$3 prefix=$4 minutes=$5 case=$6; shift 6
  case " $SKIP " in *" $name "*) mark "SKIP $name: in SKIP"; return 0;; esac
  run "$name" "$minutes" "$dir/logs/nvsmi-$name" env "$@" bash "$dir/run.sh" "$case"
  local log; log=$(newest "$dir" "$prefix")
  reached "$log" "$sys" && return 0
  # recompute helps only a run out of memory; a slow (TCP) or stuck one would just repeat
  grep -qaE "OutOfMemoryError|CUDA out of memory" "$log" 2>/dev/null \
    || { mark "FAILED $name: not out of memory, not rerun"; return 1; }
  mark "NOFIT $name: again with every block recomputed"
  run "$name-rc" "$minutes" "$dir/logs/nvsmi-$name-rc" env RECOMPUTE=1 "$@" bash "$dir/run.sh" "$case"
  reached "$(newest "$dir" "$prefix")" "$sys" || { mark "SKIP $name: no fit"; return 1; }
}
mkdir -p "$T"/{pp8dp2,tp2ep16,tp4dp4,fsdp16,vision_stage,pp16}/logs "$T/prep/logs"
bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2"

# ---- 1. TorchTitan's ground: rdsp ZeRO-3 over 16 (TorchTitan FSDP16 ran in queue35.sh)
go fsdp16-rdsp-zero3 rdsp "$T/fsdp16" rdsp-zero3 60 rdsp

# ---- 2. Bridge PP8xDP2, library path fixed, NCCL INFO logs; then the cut nearest rdsp's
BRIDGE_PP=0
go pp8dp2-bridge-even bridge "$P" bridge-even 80 bridge NAME=-even NCCL_DEBUG=INFO && BRIDGE_PP=1
CUT=$(python3 "$T/scripts/layout_from_log.py" megatron-cut "$P/logs/rdsp.log" 40)
[ $BRIDGE_PP = 1 ] && go pp8dp2-bridge-cut bridge "$P" bridge-cut 80 bridge \
  NAME="-cut${CUT/ /-}" CUT="$CUT"

# ---- 3. Bridge's ground without expert parallelism: TP4xDP4 (rdsp cannot hold this layout)
BRIDGE_TP=0
go tp4dp4-bridge bridge "$T/tp4dp4" bridge 80 bridge && BRIDGE_TP=1

bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
mark "ALL DONE"
