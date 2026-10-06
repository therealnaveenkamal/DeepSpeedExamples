#!/bin/bash
# rdsp vs Megatron (MegatronMIMO and Megatron-Bridge) on one 8-GPU node,
# Qwen3.5 on CORD-v2. Run on the node after bench/setup_node.sh:
#
#   MODEL=Qwen3.5-2B runs.sh convert             # HF -> Megatron checkpoints
#   MODEL=Qwen3.5-2B runs.sh export              # Bridge's samples, replayed by rdsp
#   MODEL=Qwen3.5-2B runs.sh noncoloc-mimo  | noncoloc-rdsp   # vision 1 GPU + language TP2xDP2
#   MODEL=Qwen3.5-2B runs.sh shared-megatron | shared-rdsp    # TP2xPP2xDP2
#   MODEL=Qwen3.5-4B runs.sh tp4pp2-megatron | tp4pp2-rdsp    # TP4xPP2xDP1
#   MODEL=Qwen3.5-4B runs.sh coloc-rdsp      # rdsp only: colocated vision
#   python summarize.py ~/runs                   # the results table
#
# Case names are the same for both models: set LOGS=~/runs-4b for the 4B runs.
#
# Settings held equal: seq 2048, 64 rows per step, one row per GPU per
# microbatch, AdamW lr 1e-5 constant, betas 0.9/0.95, eps 1e-8, no weight
# decay, no clipping, no warm-up, bf16 with fp32 gradients and optimizer
# state split over data parallel, no recompute, 50 steps. Megatron's
# defaults that differ are overridden here (clipping, warm-up, cosine decay,
# the MTP head, the shuffling sampler). Loss: the Bridge runs train on the
# token mean (calculate_per_token_loss); MegatronMIMO trains on per-
# microbatch means and the rdsp run against it does the same. rdsp runs the
# recipes in ../../recipes with DATA=exported:<Bridge's samples>.
set -euo pipefail
MODEL=${MODEL:-Qwen3.5-2B}
export STEPS=${STEPS:-50}
WORK=$HOME/mimo                       # /workspace inside the container
IMAGE=nvcr.io/nvidia/nemo:26.08
LOGS=${LOGS:-$HOME/runs}; mkdir -p "$LOGS"
RDSP=$(cd "$(dirname "$0")/../.." && pwd)
SIZE=$(echo "${MODEL#Qwen3.5-}" | tr '[:upper:]' '[:lower:]')    # 2b, 4b
case $SIZE in
  2b) RECIPE=${RECIPE:-qwen35_vl_2b_sft_1gpu_h100_bf16_config} ;;
  *)  RECIPE=${RECIPE:-qwen35_vl_${SIZE}_sft_2gpu_h100_bf16_config} ;;
esac

container() {  # container <command...>
  sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
    -v "$HOME/hf:/root/.cache/huggingface" -v "$WORK:/workspace" -v "$RDSP:/rdsp" \
    -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge \
    ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG} \
    "$IMAGE" bash -c "$*"
  sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$WORK"   # the container writes as root
}

STD_DATA="model.seq_length=2048 dataset.seq_length=2048 dataset.source.dataset_name=cord_v2 \
  dataset.dataloader_type=single train.global_batch_size=64"
STD_TRAIN="$STD_DATA train.micro_batch_size=1 train.train_iters=$STEPS \
  checkpoint.pretrained_checkpoint=/workspace/std/$MODEL checkpoint.save=null \
  model.mtp_num_layers=null model.calculate_per_token_loss=true \
  model.recompute_granularity=null \
  optimizer.lr=1e-5 optimizer.min_lr=1e-5 optimizer.adam_beta1=0.9 optimizer.adam_beta2=0.95 \
  optimizer.adam_eps=1e-8 optimizer.weight_decay=0.0 optimizer.clip_grad=0.0 \
  scheduler.lr_warmup_iters=0 scheduler.lr_decay_style=constant \
  scheduler.start_weight_decay=0.0 scheduler.end_weight_decay=0.0 \
  ddp.grad_reduce_in_fp32=true ddp.use_distributed_optimizer=true ddp.average_in_collective=false \
  validation.eval_iters=0 logger.log_interval=1"
# Diagnosis only, never for published runs: MIMO_EXTRA (e.g. "--pad-to-seq-length
# false"), MEGATRON_EXTRA (Bridge overrides, e.g. "logger.timing_log_level=2"),
# RDSP_EXTRA (train_vl.py flags, e.g. "--profile"), NCCL_DEBUG=INFO.
MIMO_TRAIN="--hf-model Qwen/$MODEL --dataset-name cord_v2 --seq-length 2048 \
  --global-batch-size 64 --train-iters $STEPS --dataloader-type single \
  --pretrained-checkpoint /workspace/$MODEL-mimo \
  --lr 1e-5 --min-lr 1e-5 --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.0 \
  --start-weight-decay 0.0 --end-weight-decay 0.0 --clip-grad 0.0 --lr-warmup-iters 0 \
  --log-interval 1 --experiment-root /workspace/runs ${MIMO_EXTRA:-}"

megatron() {  # megatron <log name> <tp> <pp>: the standard Bridge recipe
  container "python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe $RECIPE --step_func qwen3_vl_step $STD_TRAIN \
    model.tensor_model_parallel_size=$2 model.pipeline_model_parallel_size=$3 ${MEGATRON_EXTRA:-}" \
    2>&1 | tee "$LOGS/$1.log"
}

rdsp() {  # rdsp <log name> <recipe> [train_vl flags]
  local name=$1 recipe=$RDSP/recipes/$2; shift 2
  [ -f "$recipe" ] || { echo "no recipe $recipe for $MODEL" >&2; exit 1; }
  (unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && . "$HOME/cuda_env.sh" && \
   DS_BUILD_OPS=0 HF_HOME=$HOME/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   DATA=exported:$WORK/cord_steps-$MODEL bash "$recipe" --steps "$STEPS" "$@" ${RDSP_EXTRA:-}) \
    2>&1 | tee "$LOGS/$name.log"
}

case "${1:-}" in
  convert)
    container "python scripts/conversion/setup_conversion.py import --hf-model Qwen/$MODEL \
        --megatron-path /workspace/std/$MODEL && \
      python -m torch.distributed.run --nproc_per_node=2 examples/conversion/convert_megatron_mimo.py \
        import --hf-model Qwen/$MODEL --megatron-path /workspace/$MODEL-mimo \
        --component language=tp=1,dp=1,rank_offset=0 --component images=tp=1,dp=1,rank_offset=1 \
        --torch-dtype bfloat16" 2>&1 | tee "$LOGS/convert.log" ;;
  export)
    container "python /rdsp/bench/mimo/export_bridge_batches.py --out /workspace/cord_steps-$MODEL \
      --steps $STEPS --recipe $RECIPE $STD_DATA dataset.pad_to_max_length=true" \
      2>&1 | tee "$LOGS/export.log" ;;
  noncoloc-mimo)     # language TP2 x DP2 on ranks 0-3, images on rank 4
    container "python -m torch.distributed.run --nproc_per_node=5 \
      examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py $MIMO_TRAIN --micro-batch-size 2 \
      --run-name noncoloc --component language=tp=2,pp=1,dp=2,rank_offset=0 \
      --component images=tp=1,pp=1,dp=1,rank_offset=4" 2>&1 | tee "$LOGS/noncoloc-mimo.log" ;;
  noncoloc-rdsp)
    rdsp noncoloc-rdsp "qwen35-$SIZE-vision1-tp2dp2.sh" ;;
  shared-megatron)
    megatron shared-megatron 2 2 ;;
  shared-rdsp)
    rdsp shared-rdsp "qwen35-$SIZE-tp2pp2dp2.sh" ;;
  tp4pp2-megatron)
    megatron tp4pp2-megatron 4 2 ;;
  tp4pp2-rdsp)
    rdsp tp4pp2-rdsp "qwen35-$SIZE-tp4pp2dp1.sh" ;;
  coloc-rdsp)        # rdsp only: vision encoder on every GPU, language TP2 x PP2 x DP2
    rdsp coloc-rdsp "qwen35-$SIZE-coloc-tp2pp2dp2.sh" ;;
  *) sed -n 2,13p "$0"; exit 1 ;;
esac
