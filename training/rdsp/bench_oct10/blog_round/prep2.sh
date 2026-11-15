#!/bin/bash
# prep2.sh, on node 1 of the two-node pair after ops/h100/push2.sh (which leaves ~/env2.sh and
# ~/key2.pem here): both nodes ready for the 27B MIMO-layout runs (1 vision GPU + language
# TP4 x PP2, 9 GPUs). Run it with ops/h100/remote2.sh.
#   1. node_prep2.sh on both nodes at once: environment, NeMo container, py-spy, 27B weights
#   2. node 1: Megatron-Bridge's CORD-v2 rows at 128 a step, and MegatronMIMO's 27B checkpoint
#      (language TP4); both copied to node 2
#   3. NCCL all-reduce over both nodes, and the Ray cluster for rdsp
# Safe to rerun: finished parts are skipped. Markers: "--- START/END <part>", then
# "=== PREP DONE" or "=== PREP FAILED <what>".
set -uo pipefail
RDSP=~/rdsp
B=$RDSP/bench_oct10
. ~/env2.sh
LOGS=$B/blog_round/prep2_logs; mkdir -p "$LOGS"
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
step() { echo "--- $* $(date -u +%H:%M:%S)"; }
fail() { mark "PREP FAILED $*"; exit 1; }

step "START node prep"
NODE_PREP="[ -f ~/.blog_node_prep_done ] || bash ~/rdsp/bench_oct10/blog_round/node_prep2.sh"
$SSH2 "$NODE_PREP" < /dev/null >> "$LOGS/node2_prep.log" 2>&1 &
N2=$!
bash -c "$NODE_PREP" >> "$LOGS/node1_prep.log" 2>&1; rc1=$?
wait $N2; rc2=$?
cp ~/hardware.txt "$LOGS/hardware-node1.txt"; $SSH2 "cat ~/hardware.txt" < /dev/null > "$LOGS/hardware-node2.txt"
[ $rc1 = 0 ] && [ $rc2 = 0 ] || fail "node prep: node 1 exit $rc1, node 2 exit $rc2"
grep -h "H100\|EFA devices" "$LOGS"/hardware-node*.txt | sort | uniq -c
step "END node prep"

step "START 27B rows and MIMO checkpoint"
export MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config
NAME128=cord_steps-Qwen3.5-27B-128
if [ ! -f "$HOME/mimo/$NAME128/stats.json" ]; then
  env GBS=128 STEPS=50 EXPORT_NAME=$NAME128 LOGS="$LOGS" bash "$RDSP/bench/mimo/runs.sh" export > "$LOGS/export128.out" 2>&1; rc=$?
  [ $rc = 0 ] && [ -f "$HOME/mimo/$NAME128/stats.json" ] || fail "27B export exit $rc"
fi
cat "$HOME/mimo/$NAME128/stats.json"; echo
if [ ! -f "$HOME/mimo/Qwen3.5-27B-mimo.done" ]; then
  env CONVERT_TP=4 LOGS="$LOGS" bash "$RDSP/bench/mimo/runs.sh" convert-mimo > "$LOGS/convert-mimo.out" 2>&1; rc=$?
  [ $rc = 0 ] && [ -d "$HOME/mimo/Qwen3.5-27B-mimo" ] || fail "27B MIMO checkpoint exit $rc"
  touch "$HOME/mimo/Qwen3.5-27B-mimo.done"
fi
$SSH2 "mkdir -p ~/mimo" < /dev/null
for d in $NAME128 Qwen3.5-27B-mimo; do
  rsync -a -e "ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no" "$HOME/mimo/$d/" "ubuntu@$PRIV2:mimo/$d/" \
    || fail "copy $d to node 2"
done
step "END 27B rows and MIMO checkpoint"

step "START nccl check"
bash "$RDSP/bench/two_node/both.sh" bash "$RDSP/bench/two_node/nccl_check.sh" > "$LOGS/nccl_check.log" 2>&1
grep -E "bus bandwidth" "$LOGS/nccl_check.log"
grep -m2 -oE "NET/OFI Selected [^ ]+ [^ ]+|Using network [A-Za-z]+|NET/Socket" "$LOGS/nccl_check.log" | sort -u
step "END nccl check"
bash "$RDSP/bench/two_node/ray_cluster.sh" up "$PRIV1" "$PRIV2" > "$LOGS/ray_up.log" 2>&1 \
  || { cat "$LOGS/ray_up.log"; fail "ray cluster"; }
df -h ~ | tail -1; $SSH2 "df -h ~ | tail -1" < /dev/null
mark "PREP DONE"
