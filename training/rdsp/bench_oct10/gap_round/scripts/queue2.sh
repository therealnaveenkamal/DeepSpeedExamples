#!/bin/bash
# queue2.sh, on node 1 (ops/h100/remote2.sh), after queue.sh: the grid's other two cells
# (../plan.md, phases C and D), so every system gets its best configuration on both models at
# both sizes.
#   phase C   Qwen3.5-35B-A3B on one node, both nodes at once: node 1 rdsp + TorchTitan, node 2
#             Bridge (scripts/one_node35.sh); node 2's markers copied here with "node2" in front
#   phase D   dense Qwen3.5-27B on 16 GPUs: Bridge (NVIDIA's 16-GPU recipe TP4 x PP4, and
#             PP8 x DP2 at its best phase-A cut), TorchTitan FSDP16, rdsp PP8 x DP2
# Markers ("=== ") for ops/h100/guard.sh.
set -x
cd ~/rdsp
. ~/env2.sh
B=~/rdsp/bench_oct10
G=$B/gap_round
T=$B/two_node_35b
P8=$B/bridge_vs_rdsp/27b_tp1pp8dp1
GL=$G/logs; mkdir -p "$GL"
RUNS=~/rdsp/bench/mimo/runs.sh
BOTH=~/rdsp/bench/two_node/both.sh
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
RSYNC2="rsync -a -e 'ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no'"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() { bash "$BOTH" --cleanup; pkill -f "[t]rain_vl.py"; sleep 10; }
run() {  # run <name> <timeout minutes> <command...>: a two-node run
  local name=$1 minutes=$2; shift 2
  mark "START $name"
  $SMI > "$GL/nvsmi-$name-node1.csv" &
  local smi1=$!
  $SSH2 "$SMI" < /dev/null > "$GL/nvsmi-$name-node2.csv" &
  local smi2=$!
  timeout $((minutes * 60)) "$@"; local rc=$?
  kill $smi1 $smi2; $SSH2 "pkill -f '[n]vidia-smi --query-gpu=timestamp'" < /dev/null
  cleanup
  mark "END $name exit $rc"
}
nofit() {  # nofit <log>: ran out of memory before step 3
  ! grep -qaE "^step 3 |iteration +3/|step: +3 " "$1" 2>/dev/null &&
    grep -qaE "OutOfMemoryError|CUDA out of memory|out of memory" "$1" 2>/dev/null
}
bridge_median() {  # bridge_median <log>: median seconds per step over steps 6-50, or nothing
  python3 - "$1" <<'EOF'
import re, statistics, sys
try:
    t = {int(m.group(1)): float(m.group(2)) / 1000 for m in re.finditer(
        r"iteration +(\d+)/ *\d+ \|.*?elapsed time per iteration \(ms\): ([\d.]+)", open(sys.argv[1], errors="ignore").read())}
except OSError:
    sys.exit()
s = [t[i] for i in range(6, 51) if i in t]
if len(s) >= 40:
    print(f"{statistics.median(s):.2f}")
EOF
}
export_rows() {  # export_rows <model> <recipe> <rows a step> <folder>: Bridge's rows, on node 1
  [ -f "$HOME/mimo/$4/stats.json" ] && return 0
  env MODEL=$1 RECIPE=$2 GBS=$3 STEPS=50 EXPORT_NAME=$4 LOGS="$GL" bash "$RUNS" export > /dev/null 2>&1
  mv "$GL/export.log" "$GL/export-$4.log" 2>/dev/null
  [ -f "$HOME/mimo/$4/stats.json" ] && cat "$HOME/mimo/$4/stats.json"
}
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
MEG27="$(cat "$B/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/meg_extra_bf16.txt") dataset.pad_to_max_length=true \
dist.distributed_timeout_minutes=60"

while pgrep -f "[b]ash queue.sh" > /dev/null; do sleep 30; done
mark "QUEUE2 START"
reached() {  # reached <log>: 3 steps done (rdsp "step 3", Bridge "iteration 3/" or a third
  # "Step Time" line on its two-node logs, TorchTitan "step: 3")
  grep -qaE "^step 3 |iteration +3/|step: +3 " "$1" 2>/dev/null || [ "$(grep -ac '^Step Time :' "$1" 2>/dev/null)" -ge 3 ]
}
nofit() { ! reached "$1" && grep -qaE "OutOfMemoryError|CUDA out of memory|out of memory" "$1" 2>/dev/null; }
# 2026-10-19 03:45 UTC, the user: phases C (35B on one node) and D (27B on two nodes) dropped. Left
# here: A8 (TorchTitan 8 stages on the 27B, one node; phase A lost it) and Megatron-MIMO on the 27B
# over two nodes with our memory settings (gap_round/mimo/patch_mimo.py).
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"

# A8 on node 2: TorchTitan's DCP copy (on NVMe), then TorchTitan 8 stages at rdsp's cut
$SSH2 "[ -f ~/tt_dcp/Qwen3.5-27B/.metadata ] || (unset LD_LIBRARY_PATH; . ~/tt_venv/bin/activate; cd ~/torchtitan && \
  python scripts/checkpoint_conversion/convert_from_hf.py \$(ls -d ~/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/* | head -1) \
  ~/tt_dcp/Qwen3.5-27B --model_name qwen3_5 --model_flavor 27B) > ~/convert-dcp-27b-node2.log 2>&1; \
  cd ~/rdsp && env NAME=-gap TT_DCP=/home/ubuntu/tt_dcp/Qwen3.5-27B LOG_RANK=0,7 timeout 3000 \
  bash bench_oct10/torchtitan/scripts/run.sh run qwen35_27b_cord_pp8_rc > /dev/null 2>&1" < /dev/null &
A8=$!
mark "START titan-27b-pp8 on node 2"

# meanwhile on node 1: Megatron-MIMO's 27B checkpoint (language over 4 GPUs), copied to node 2
mark "START convert-mimo-27b"
if [ ! -d ~/mimo/Qwen3.5-27B-mimo ]; then
  env MODEL=Qwen3.5-27B CONVERT_TP=4 LOGS="$GL" bash "$RUNS" convert-mimo > /dev/null 2>&1
fi
if [ -d ~/mimo/Qwen3.5-27B-mimo ]; then
  $SSH2 "mkdir -p ~/mimo" < /dev/null
  eval "$RSYNC2 $HOME/mimo/Qwen3.5-27B-mimo/ ubuntu@$PRIV2:mimo/Qwen3.5-27B-mimo/" || mark "FAILED copy MIMO 27B to node 2"
else
  mark "FAILED MIMO 27B conversion"
fi
mark "END convert-mimo-27b"
wait $A8
TT8=$($SSH2 "python3 ~/rdsp/bench_oct10/gap_round/scripts/medians.py titan \
  ~/rdsp/bench_oct10/torchtitan/logs/run-qwen35_27b_cord_pp8_rc-gap.log" < /dev/null)
mark "END titan-27b-pp8 on node 2: ${TT8:-no 50-step median}"

# Megatron-MIMO, 27B, two nodes: the vision encoder alone on 1 GPU of node 2 (MIMO's fastest
# vision width in its 9B sweep, and NVIDIA's tutorial layout), language TP4 x PP2 on node 1's 8 GPUs:
# 9 GPUs, node 2's other 7 idle, 1 row a microbatch. As shipped first; if it runs out of memory,
# with our memory settings (gap_round/mimo/patch_mimo.py); then also every language layer
# recomputed. Receipt rows, 128 x 2048 a step, as rdsp's two-node 27B run (10.19 s, 2026-10-18).
mimo9() {  # mimo9 <name> [VAR=value...]
  local name=$1; shift
  run "mimo-27b-$name" 70 bash "$BOTH" env MODEL=Qwen3.5-27B GBS=128 STEPS=50 LOGS="$GL" "$@" \
    MIMO_NAME="mimo-$name" MIMO_GPUS=9 MIMO_MBS=1 MIMO_NPROC0=8 MIMO_NPROC1=1 \
    MIMO_LANGUAGE=tp=4,pp=2,dp=1,rank_offset=0 MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=8 bash "$RUNS" mimo-layout
}
if [ -d ~/mimo/Qwen3.5-27B-mimo ]; then
  mimo9 v1-shipped
  if nofit "$GL/mimo-v1-shipped.log"; then
    mark "NOFIT mimo-27b-v1-shipped: again with our memory settings"
    mimo9 v1-patched MIMO_PATCH=1
    if nofit "$GL/mimo-v1-patched.log"; then
      mark "NOFIT mimo-27b-v1-patched: again with every language layer recomputed"
      mimo9 v1-patched-rc MIMO_PATCH=1 MIMO_RECOMPUTE=1
    fi
  fi
fi
# Megatron-MIMO as shipped, NVIDIA's tutorial layout over 17 GPUs: language TP4 x PP2 x DP2 on the
# two 8-GPU nodes, the vision encoder alone on the third box (p5.4xlarge, ops/h100/launch3.sh);
# 2 rows a microbatch (1 per language copy). Only if the third box is up and ready.
if [ -f ~/env3.sh ] && [ -d ~/mimo/Qwen3.5-27B-mimo ]; then
  . ~/env3.sh
  if ssh -i "$HOME/key2.pem" -o StrictHostKeyChecking=no -o BatchMode=yes "ubuntu@$PRIV3" "[ -f ~/.box3_ready ]" < /dev/null; then
    mark "START copy MIMO 27B to box 3"
    eval "rsync -a -e 'ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no' $HOME/mimo/Qwen3.5-27B-mimo/ ubuntu@$PRIV3:mimo/Qwen3.5-27B-mimo/" \
      || mark "FAILED copy MIMO 27B to box 3"
    run mimo-27b-17gpu 70 bash ~/rdsp/bench/two_node/three.sh env MODEL=Qwen3.5-27B GBS=128 STEPS=50 LOGS="$GL" \
      MIMO_NAME=mimo-17gpu MIMO_GPUS=17 MIMO_MBS=2 MIMO_NPROC0=8 MIMO_NPROC1=8 MIMO_NPROC2=1 \
      MIMO_LANGUAGE=tp=4,pp=2,dp=2,rank_offset=0 MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=16 bash "$RUNS" mimo-layout
    bash ~/rdsp/bench/two_node/three.sh --cleanup
  else
    mark "SKIP mimo-27b-17gpu: box 3 not ready"
  fi
fi
# MIMO 17 GPUs again when the first try failed in NCCL ("Internal check failed" at its first
# broadcast, 2026-10-19 05:31): NCCL INFO logs, PXN off (nodes with 8, 8 and 1 GPUs)
if grep -qa "ncclInternalError\|ncclRemoteError" "$GL/mimo-17gpu.log" 2>/dev/null && [ -f ~/env3.sh ]; then
  mark "RETRY mimo-27b-17gpu: NCCL INFO, PXN off"
  run mimo-27b-17gpu-retry 40 bash ~/rdsp/bench/two_node/three.sh env MODEL=Qwen3.5-27B GBS=128 STEPS=50 LOGS="$GL" \
    NCCL_DEBUG=INFO NCCL_P2P_PXN_LEVEL=0 MIMO_NAME=mimo-17gpu-retry MIMO_GPUS=17 MIMO_MBS=2 \
    MIMO_NPROC0=8 MIMO_NPROC1=8 MIMO_NPROC2=1 MIMO_LANGUAGE=tp=4,pp=2,dp=2,rank_offset=0 \
    MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=16 bash "$RUNS" mimo-layout
  bash ~/rdsp/bench/two_node/three.sh --cleanup
fi
if grep -qa "ncclInternalError\|ncclRemoteError" "$GL/mimo-17gpu-retry.log" 2>/dev/null && [ -f ~/env3.sh ]; then
  mark "RETRY mimo-27b-17gpu over TCP: the container's aws-ofi-nccl failed (Failed to find callback)"
  run mimo-27b-17gpu-tcp 50 bash ~/rdsp/bench/two_node/three.sh env MODEL=Qwen3.5-27B GBS=128 STEPS=50 LOGS="$GL" \
    NET=socket MIMO_NAME=mimo-17gpu-tcp MIMO_GPUS=17 MIMO_MBS=2 \
    MIMO_NPROC0=8 MIMO_NPROC1=8 MIMO_NPROC2=1 MIMO_LANGUAGE=tp=4,pp=2,dp=2,rank_offset=0 \
    MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=16 bash "$RUNS" mimo-layout
  bash ~/rdsp/bench/two_node/three.sh --cleanup
fi
# rdsp over the same 17 GPUs, beside MIMO's: the vision encoder alone on the third box's GPU, the
# language model as 8 stages of 2 GPUs on the two 8-GPU nodes (scripts/rdsp17.sh, driver on the
# third box). The 27B's receipt rows at 128 a step first, on all three boxes.
if [ -f ~/env3.sh ]; then
  . ~/env3.sh
  SSH3="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV3"
  if $SSH3 "[ -f ~/.box3_rdsp_ready ]" < /dev/null; then
    mark "START export 27B 128 rows"
    export_rows Qwen3.5-27B qwen35_vl_27b_sft_16gpu_h100_bf16_config 128 cord_steps-Qwen3.5-27B-128 \
      || mark "FAILED export 27B 128"
    for h in "$PRIV2" "$PRIV3"; do
      eval "$RSYNC2 $HOME/mimo/cord_steps-Qwen3.5-27B-128/ ubuntu@$h:mimo/cord_steps-Qwen3.5-27B-128/" \
        || mark "FAILED copy 27B 128 rows to $h"
    done
    mark "END export 27B 128 rows"
    bash ~/rdsp/bench/two_node/ray_cluster3.sh up || mark "FAILED ray cluster over 3 nodes"
    run rdsp-27b-17gpu 50 bash -c "$SSH3 'bash ~/rdsp/bench_oct10/gap_round/scripts/rdsp17.sh' < /dev/null > $GL/rdsp-17gpu.log 2>&1"
    if nofit "$GL/rdsp-17gpu.log"; then
      mark "NOFIT rdsp-27b-17gpu: again with every block recomputed"
      bash ~/rdsp/bench/two_node/ray_cluster3.sh up
      run rdsp-27b-17gpu-rc 50 bash -c "$SSH3 'RECOMPUTE=1 bash ~/rdsp/bench_oct10/gap_round/scripts/rdsp17.sh' < /dev/null > $GL/rdsp-17gpu-rc.log 2>&1"
    fi
    $SSH3 "pkill -f '[t]rain_vl.py'" < /dev/null
    bash ~/rdsp/bench/two_node/ray_cluster3.sh down
  else
    mark "SKIP rdsp-27b-17gpu: box 3 not ready for rdsp"
  fi
  mark "BOX 3 SHUTTING DOWN (terminates)"
  $SSH3 "sudo shutdown -h now" < /dev/null
fi
mark "ALL DONE 2"
