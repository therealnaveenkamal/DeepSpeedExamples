#!/bin/bash
# On node 1 of the two-node pair (ops/h100: launch2.sh, push2.sh, then remote2.sh with this file):
# every run of two_node_35b/plan.md, in order, unattended.
#   prep (scripts/prep.sh) · 1 rdsp PP8xDP2 · 2 Bridge PP8xDP2 even, then at the cut
#   nearest rdsp's · 8 rdsp vision stage (K from run 1) · 9 rdsp PP16 · 4-5 Bridge and rdsp
#   TP2xEP16 · 3 and 6 TorchTitan FSDP16, + EP8 · 7 rdsp ZeRO-3
# Each run: 50 steps; one that stops before step 3 (out of memory) runs again with every block
# recomputed. Each has a timeout, nvidia-smi samples from both nodes (every GPU's used memory
# every 0.5 s) and a marker line "=== END <run> exit <code>" for ops/h100/guard.sh.
# SKIP: run names (go's first argument) to leave out, e.g. ones already done; ~/queue35.skip, if
# present, holds them (remote2.sh passes no environment).
set -x
cd ~/rdsp
SKIP=${SKIP:-$(cat ~/queue35.skip 2>/dev/null)}
. ~/env2.sh
T=~/rdsp/bench_oct10/two_node_35b
P=$T/pp8dp2
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {  # every container, torchrun and rdsp driver on both nodes; a fresh Ray cluster
  bash bench/two_node/both.sh --cleanup
  pkill -f "[t]rain_vl.py"
  sleep 10
  bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
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
    bridge) grep -qE "iteration +3/" "$1" ;;
    titan)  [ "$(sed 's/\x1b\[[0-9;]*m//g' "$1" | grep -oE "step: +[0-9]+" | awk '{print $2}' \
              | sort -n | tail -1)" -ge 3 ] 2>/dev/null ;;
  esac
}
go() {  # go <name> <rdsp|bridge|titan> <layout dir> <log prefix> <minutes> <run.sh case> [VAR=value...]
  local name=$1 sys=$2 dir=$3 prefix=$4 minutes=$5 case=$6; shift 6
  case " $SKIP " in *" $name "*) mark "SKIP $name: in SKIP"; return 0;; esac
  run "$name" "$minutes" "$dir/logs/nvsmi-$name" env "$@" bash "$dir/run.sh" "$case"
  reached "$(newest "$dir" "$prefix")" "$sys" && return 0
  mark "NOFIT $name: again with every block recomputed"
  run "$name-rc" "$minutes" "$dir/logs/nvsmi-$name-rc" env RECOMPUTE=1 "$@" bash "$dir/run.sh" "$case"
  reached "$(newest "$dir" "$prefix")" "$sys" || { mark "SKIP $name: no fit"; return 1; }
}
mkdir -p "$T"/{pp8dp2,tp2ep16,fsdp16,vision_stage,pp16}/logs

# ---- prep
timeout $((120 * 60)) bash "$T/scripts/prep.sh" || { mark "PREP FAILED"; exit 1; }

# ---- 1-2: PP8xDP2, rdsp then Bridge
go pp8dp2-rdsp rdsp "$P" rdsp 60 rdsp
RUN1=$P/logs/rdsp.log; [ -f "$P/logs/rdsp-rc.log" ] && RUN1=$P/logs/rdsp-rc.log
go pp8dp2-bridge-even bridge "$P" bridge 80 bridge NAME=-even
CUT=$(python3 "$T/scripts/layout_from_log.py" megatron-cut "$RUN1" 40)
mark "CUT nearest rdsp's: first last = $CUT"
[ -n "$CUT" ] && [ "$CUT" != "5 5" ] && go pp8dp2-bridge-cut bridge "$P" bridge 80 bridge \
  NAME="-cut${CUT/ /-}" CUT="$CUT"

# ---- 8-9: rdsp's own layouts
K=$(python3 "$T/scripts/layout_from_log.py" vision-width "$RUN1" 16)
mark "VISION K=$K"
[ -n "$K" ] && go vision-stage-rdsp rdsp "$T/vision_stage" rdsp 60 rdsp K="$K"
go pp16-rdsp rdsp "$T/pp16" rdsp 75 rdsp

# ---- 4-5: Bridge's own layout for this model
go tp2ep16-bridge bridge "$T/tp2ep16" bridge 80 bridge
go tp2ep16-rdsp rdsp "$T/tp2ep16" rdsp 60 rdsp

# ---- 3, 6, 7: fully sharded
go fsdp16-titan titan "$T/fsdp16" run-qwen35_35b_a3b_cord_fsdp16 60 titan
go fsdp16-titan-ep8 titan "$T/fsdp16" run-qwen35_35b_a3b_cord_fsdp16_ep8 60 titan-ep8
go fsdp16-rdsp-zero3 rdsp "$T/fsdp16" rdsp-zero3 60 rdsp

bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
mark "ALL DONE"
