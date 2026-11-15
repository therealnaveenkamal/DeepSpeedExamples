#!/bin/bash
# rdsp, Qwen3.5-27B, vision encoder alone on 1 GPU + language on the other 7 GPUs as
# stages of TP4, TP2 and TP1 (--cuts 0,balanced; layers split by parameters per GPU).
# Every row padded to 2048 (equal work). bf16 gradient accumulation + fused AdamW, fp32
# master and Adam.
#   STEPS (default 3: a memory check), LOSS (default microbatch-mean, MIMO's objective;
#   token-mean for the 50-step run, like the TP4xPP2 runs), PROFILE (default 1: --profile),
#   NAME (log suffix), EXTRA (more train_vl.py flags), RECOMPUTE=1 (recompute every block of
#   the three language stages). On the H100 box.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RDSP=$(cd "$HERE/../../../.." && pwd)
LOGS=$HERE/../logs; mkdir -p "$LOGS"
RC=""; [ "${RECOMPUTE:-0}" = 1 ] && RC=",recompute=1"
PROF=""; [ "${PROFILE:-1}" = 1 ] && PROF="--profile"
(unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
 env DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 python "$RDSP/train_vl.py" \
    --model Qwen/Qwen3.5-27B --dataset "exported:$HOME/mimo/cord_steps-Qwen3.5-27B" \
    --stages 4 --cuts 0,balanced \
    --stage "0:gpus=1,zero=1" \
    --stage "1:gpus=4,tp=4,zero=1$RC" \
    --stage "2:gpus=2,tp=2,zero=1$RC" \
    --stage "3:gpus=1,zero=1$RC" \
    --rows 1 --microbatches 64 --seq 2048 --pad-multiple 0 \
    --steps "${STEPS:-3}" --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
    --grad-dtype bf16 --fused-adam --loss "${LOSS:-microbatch-mean}" --liger \
    --prefetch --sharded-loss --drop-padding-mask $PROF ${EXTRA:-}) 2>&1 | tee "$LOGS/fit-rdsp${NAME:-}.log"
