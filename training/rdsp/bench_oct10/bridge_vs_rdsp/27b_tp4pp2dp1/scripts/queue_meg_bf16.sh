#!/bin/bash
# On the H100 box (2026-10-12): Megatron-Bridge only, 27B TP4xPP2xDP1, no recompute, 3 steps,
# with bf16 gradients, an exact fp32 master (16-bit remainders) and fp32 Adam: the
# precision-aware optimizer. The overrides are read from ~/meg_extra.txt when the run starts,
# so they can be corrected while the checkpoint converts. nvidia-smi samples every GPU's
# used memory (all of it, PyTorch or not) every 0.5 s during the run.
set -x
cd ~/rdsp
S=~/rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts
L=~/rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/logs
[ -d ~/mimo/std/Qwen3.5-27B ] || NAME=-bf16 bash $S/run.sh convert
echo "CONVERT DONE"
while [ ! -f ~/meg_extra.ready ]; do sleep 10; done   # overrides checked against the container's code
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > $L/nvsmi-bridge-bf16.csv &
SMI=$!
RECOMPUTE=0 GRAD_REDUCE_FP32=false NAME=-bf16grads EXTRA="$(cat ~/meg_extra.txt)" bash $S/run.sh preflight-bridge
kill $SMI
grep -E "iteration +[0-9]+/|OutOfMemory|memory \(GB\)|Error|error:" $L/preflight-bridge-bf16grads.log | cut -c1-400 | tail -30
awk -F', ' '{if ($3 > m[$2]) m[$2] = $3} END {for (g in m) print "GPU", g, "max used MiB", m[g]}' $L/nvsmi-bridge-bf16.csv | sort -k2n
echo "QUEUE DONE"
