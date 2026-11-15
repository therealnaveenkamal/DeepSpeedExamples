#!/bin/bash
# queue.sh, on node 1 (ops/h100/remote2.sh): the gap round (../plan.md) end to end.
#   prep      scripts/prep.sh
#   phase A   one-node runs on both nodes at once: node 1 Bridge (A1-A3), node 2 TorchTitan and
#             rdsp (A4-A7); node 2's markers are copied here with "node2" in front
#   phase B   two-node runs B1-B6
# Markers ("=== ") for ops/h100/guard.sh. Logs: each layout's own logs directory, and ../logs.
set -x
cd ~/rdsp
. ~/env2.sh
B=~/rdsp/bench_oct10
G=$B/gap_round
T=$B/two_node_35b
GL=$G/logs; mkdir -p "$GL"
SSH2="ssh -i $HOME/key2.pem -o StrictHostKeyChecking=no -o BatchMode=yes ubuntu@$PRIV2"
SMI="nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500"
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
cleanup() {
  bash bench/two_node/both.sh --cleanup
  pkill -f "[t]rain_vl.py"
  sleep 10
}
run() {  # run <name> <timeout minutes> <nvidia-smi csv prefix> <command...>: a two-node run
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
oom() { grep -qaE "OutOfMemoryError|CUDA out of memory|out of memory" "$1" 2>/dev/null; }
reached() { grep -qaE "^step 3 |iteration +3/|step: +3 " "$1" 2>/dev/null; }   # rdsp, Bridge, TorchTitan

# ---- prep
bash "$G/scripts/prep.sh" || { mark "FAILED prep"; exit 1; }

# ---- phase A: one node each, at the same time (no Ray cluster: rdsp starts its own on node 2)
bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
$SSH2 "bash ~/rdsp/bench_oct10/gap_round/scripts/one_node.sh others 2>&1 | tee ~/one_node-others.out \
  | grep --line-buffered '^=== '" < /dev/null | sed -u 's/^=== /=== node2 /' &
N2=$!
bash "$G/scripts/one_node.sh" bridge
wait $N2
$SSH2 "cat ~/one_node-others.out" < /dev/null > "$GL/one_node-others-node2.out"
mark "PHASE A DONE"

# ---- phase B: both nodes
. "$T/scripts/common.sh"   # MEG_COMMON, MEG_TP1, MEG_RC, RUNS, BOTH, rdsp settings
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
bridge35() {  # bridge35 <name> <tp> <pp> [overrides...]: Bridge on the 35B over both nodes
  local name=$1 tp=$2 pp=$3; shift 3
  run "bridge-35b-$name" 80 "$GL/nvsmi-bridge-$name" bash "$BOTH" env MODEL=Qwen3.5-35B-A3B \
    RECIPE=qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config GBS=128 STEPS=50 LOGS="$GL" NET=efa \
    GRAD_REDUCE_FP32=false CUSTOM_NAME="bridge-$name" CUSTOM_TP=$tp CUSTOM_PP=$pp \
    MEGATRON_EXTRA="$MEG_COMMON $MEG_TP1 $FAST $*" bash "$RUNS" custom-megatron
}
bash bench/two_node/ray_cluster.sh up "$PRIV1" "$PRIV2" || mark "FAILED ray cluster (rdsp runs will fail)"

# B1: Bridge TP1 x EP8 x DP16; recompute only if it does not fit without
bridge35 tp1ep8 1 1 model.expert_model_parallel_size=8
if ! reached "$GL/bridge-tp1ep8.log" && oom "$GL/bridge-tp1ep8.log"; then
  mark "NOFIT bridge-35b-tp1ep8: again with every layer recomputed"
  bridge35 tp1ep8-rc 1 1 model.expert_model_parallel_size=8 $MEG_RC
fi

# B2, B3: TorchTitan FSDP16 + EP8, fp32 then bf16 gradient reduce
run titan-35b-ep8 40 "$T/fsdp16/logs/nvsmi-gap-titan-ep8" env NAME=-gap bash "$T/fsdp16/run.sh" titan-ep8
run titan-35b-ep8-bf16 40 "$T/fsdp16/logs/nvsmi-gap-titan-ep8-bf16" env NAME=-gap bash "$T/fsdp16/run.sh" titan-ep8-bf16

# (two-node rdsp rerun dropped 2026-10-19: the user's call, numbers already measured)
# B1 again: node 2's disk was full when B1 started (its side never launched; 03:09 UTC)
bridge35 tp1ep8-again 1 1 model.expert_model_parallel_size=8
if ! reached "$GL/bridge-tp1ep8-again.log" && oom "$GL/bridge-tp1ep8-again.log"; then
  mark "NOFIT bridge-35b-tp1ep8-again: again with every layer recomputed"
  bridge35 tp1ep8-rc 1 1 model.expert_model_parallel_size=8 $MEG_RC
fi

# B5: NCCL all-reduce and all-to-all, host stack and inside Bridge's container
run nccl-host 10 "$GL/nvsmi-nccl-host" bash -c "bash $BOTH bash bench/two_node/nccl_check.sh > $GL/nccl-check-host.log 2>&1"
run nccl-container 15 "$GL/nvsmi-nccl-container" bash "$BOTH" env LOGS="$GL" bash "$RUNS" nccl-check
grep -hE "bus bandwidth" "$GL/nccl-check-host.log" "$GL/nccl-check-container.log"

# B6: Bridge TP1 x PP2 x EP8 (pipeline across the nodes, experts inside each), unbatched sends
bridge35 pp2ep8 1 2 model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false
if ! reached "$GL/bridge-pp2ep8.log" && oom "$GL/bridge-pp2ep8.log"; then
  mark "NOFIT bridge-35b-pp2ep8: again with every layer recomputed"
  bridge35 pp2ep8-rc 1 2 model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false $MEG_RC
fi
# B6 with NVIDIA's measured settings (Megatron-Bridge main, Qwen3.6-35B-A3B "SFT · H100"): layers
# 17 | 23 (the vision encoder on stage 0), TE's fused cross-entropy; every layer recomputed only if
# it does not fit without (NVIDIA recomputes)
NV="\"model.pipeline_model_parallel_layout='Et*17|t*23L'\" model.cross_entropy_fusion_impl=te"
bridge35 pp2ep8-nv 1 2 model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false $NV
if ! reached "$GL/bridge-pp2ep8-nv.log" && oom "$GL/bridge-pp2ep8-nv.log"; then
  mark "NOFIT bridge-35b-pp2ep8-nv: again with every layer recomputed"
  bridge35 pp2ep8-nv-rc 1 2 model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false $NV $MEG_RC
fi

bash bench/two_node/ray_cluster.sh down "$PRIV1" "$PRIV2"
mark "ALL DONE"
