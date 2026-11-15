#!/bin/bash
# rdsp_local.sh <log path> [train_vl.py flags]: rdsp on this node only (its own local Ray), with
# two_node_35b's shared settings (scripts/common.sh RDSP_COMMON). MODEL and EXPORT_NAME (the
# rows in ~/mimo) from the environment; 50 steps.
set -uo pipefail
LOG=$1; shift
B=~/rdsp/bench_oct10
. "$B/two_node_35b/scripts/common.sh"
unset RAY_ADDRESS
(unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
 env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python "$RDSP/train_vl.py" --model "Qwen/$MODEL" --dataset "exported:$HOME/mimo/${EXPORT_NAME:?}" \
     --steps "${STEPS:-50}" $RDSP_COMMON "$@") 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
