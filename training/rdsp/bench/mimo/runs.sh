#!/bin/bash
# Placement-matched runs: Megatron-Bridge (standard and MegatronMIMO) and rdsp
# on one 8-GPU node, Qwen3.5 (4B by default; MODEL=Qwen3.5-9B RECIPE=...) on CORD-v2. Runs on the instance:
#
#   runs.sh convert                      # HF -> Megatron checkpoints (standard, MIMO)
#   runs.sh export                       # Bridge's own samples for rdsp (50 steps)
#   runs.sh gate                         # same first step: Megatron vs rdsp, rdsp vs HF
#   runs.sh shared-megatron | shared-rdsp
#   runs.sh noncoloc-mimo   | noncoloc-rdsp
#   runs.sh coloc-rdsp
#
# Every run uses the settings below. Megatron's defaults that differ are
# overridden: clipping (1.0), LR warmup (200), cosine decay, the MTP head
# (standard recipe), and the shuffling sampler (standard recipe).
#   seq 2048, global batch 64, 1 row per language data-parallel rank per
#   microbatch, padding to a multiple of 128 (rdsp: per step), bf16 with
#   fp32 master weights and fp32 gradient reduction, AdamW lr 1e-5
#   constant, betas 0.9/0.95, eps 1e-8, no weight decay, optimizer state
#   sharded over data parallel (rdsp: ZeRO-2, which also splits gradients;
#   same reduce-scatter traffic as Megatron's distributed optimizer, less
#   memory: rdsp does not split the 248k-word embedding and head over TP),
#   loss = mean over all supervised tokens of the step (what Megatron
#   reports; the standard run trains on it too; MIMO, which cannot change
#   it, trains on per-microbatch means), no
#   recompute, no offload, 50 steps.
set -euo pipefail
MODEL=${MODEL:-Qwen3.5-4B}
export STEPS=${STEPS:-50}
WORK=$HOME/mimo                       # /workspace inside the container
IMAGE=nvcr.io/nvidia/nemo:26.08
LOGS=$HOME/runs; mkdir -p "$LOGS"

container() {  # container <command...>
  sudo docker run --gpus all --rm --ipc=host --shm-size=64g --ulimit memlock=-1 \
    -v "$HOME/hf:/root/.cache/huggingface" -v "$WORK:/workspace" -v "$HOME/rdsp:/rdsp" \
    -e HF_HOME=/root/.cache/huggingface -e PYTHONUNBUFFERED=1 -w /opt/Megatron-Bridge \
    "$IMAGE" bash -c "$*"
}

# shared by the standard recipe and the export
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
RECIPE=${RECIPE:-qwen35_vl_4b_sft_2gpu_h100_bf16_config}

# MIMO: --micro-batch-size is the global microbatch over the language DP group
MIMO_TRAIN="--hf-model Qwen/$MODEL --dataset-name cord_v2 --seq-length 2048 \
  --global-batch-size 64 --train-iters $STEPS --dataloader-type single \
  --pretrained-checkpoint /workspace/$MODEL-mimo \
  --lr 1e-5 --min-lr 1e-5 --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.0 \
  --start-weight-decay 0.0 --end-weight-decay 0.0 --clip-grad 0.0 --lr-warmup-iters 0 \
  --log-interval 1 --experiment-root /workspace/runs"

DATA=cord_steps-$MODEL                # Bridge's samples for this model (export)
# VIS_OPTS / LANG_OPTS: extra stage keys for noncoloc-rdsp, e.g. VIS_OPTS=,compile=1
# RDSP_EXTRA: more train_vl flags for one run, e.g. RDSP_EXTRA="--prefetch --profile"
RDSP_TRAIN="--model Qwen/$MODEL --dataset exported:$WORK/$DATA --pad-multiple 128 \
  --seq 2048 --steps $STEPS --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 \
  --grad-dtype fp32 --loss token-mean --liger"

rdsp() {  # rdsp <log name> <train_vl args...>
  local name=$1; shift
  (cd "$HOME/rdsp" && unset LD_LIBRARY_PATH && . "$HOME/venv/bin/activate" && \
   . "$HOME/cuda_env.sh" && DS_BUILD_OPS=0 HF_HOME=$HOME/hf \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   python train_vl.py $RDSP_TRAIN $RDSP_EXTRA "$@") 2>&1 | tee "$LOGS/$name.log"
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
    container "python /rdsp/bench/mimo/export_bridge_batches.py --out /workspace/$DATA \
      --steps $STEPS --recipe $RECIPE $STD_DATA dataset.pad_to_max_length=true" \
      2>&1 | tee "$LOGS/export.log" ;;
  megatron-layout)   # TP=.. PP=.. runs.sh megatron-layout: any standard-recipe layout
    container "python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
      --recipe $RECIPE --step_func qwen3_vl_step $STD_TRAIN \
      model.tensor_model_parallel_size=${TP:-1} model.pipeline_model_parallel_size=${PP:-4}" \
      2>&1 | tee "$LOGS/megatron-tp${TP:-1}pp${PP:-4}.log" ;;
  shared-megatron)   # TP2 x PP2 x DP2, encoder on PP0
    container "python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
      --recipe $RECIPE --step_func qwen3_vl_step $STD_TRAIN \
      model.tensor_model_parallel_size=2 model.pipeline_model_parallel_size=2" \
      2>&1 | tee "$LOGS/shared-megatron.log" ;;
  shared-rdsp)
    # ZeRO-1 reduces gradients once per step, as Megatron does (ZeRO-2 does
    # it after every microbatch: ~2x slower over PCIe). Hugging Face layers
    # keep more activations than Transformer Engine's, so 32 GB needs recompute.
    # Liger kernels; cut 14 with recompute only on the vision stage balances
    # the two stages best within 32 GB
    rdsp shared-rdsp --rows 2 --microbatches 32 --stages 2 --cuts 14 \
      --stage 0:gpus=4,tp=2,zero=1,recompute=1 --stage 1:gpus=4,tp=2,zero=1 ;;
  noncoloc-mimo)     # language TP4 on ranks 0-3, images on rank 4. One language stage:
                     # MIMO's checkpoint load fails for a tied embedding (Qwen3.5-4B)
                     # split over pipeline stages. One image rank: with two, MIMO's
                     # fan-in fails when images differ in size (CORD-v2 does)
    container "python -m torch.distributed.run --nproc_per_node=5 \
      examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py $MIMO_TRAIN --micro-batch-size $((4 / ${LANG_TP:-4} > 1 ? 4 / ${LANG_TP:-4} : 2)) \
      --run-name noncoloc --component language=tp=${LANG_TP:-4},pp=1,dp=$((4 / ${LANG_TP:-4})),rank_offset=0 \
      --component images=tp=1,pp=1,dp=1,rank_offset=4" 2>&1 | tee "$LOGS/noncoloc-mimo.log" ;;
  noncoloc-rdsp)
    rdsp noncoloc-rdsp --rows 2 --microbatches 32 --stages 2 --cuts 0 \
      --stage 0:gpus=1,zero=1${VIS_OPTS:-} --stage 1:gpus=4,tp=${LANG_TP:-4},zero=1${LANG_OPTS:-} ;;
  pp4-megatron)      # TP1 x PP4 x DP2; without recompute it runs out of 32 GB
    container "python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
      --recipe $RECIPE --step_func qwen3_vl_step $STD_TRAIN \
      model.tensor_model_parallel_size=1 model.pipeline_model_parallel_size=4 \
      model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1" \
      2>&1 | tee "$LOGS/pp4-megatron.log" ;;
  pp4-rdsp)          # the same layout, vision on stage 0; recompute where it must
    rdsp pp4-rdsp --rows 2 --microbatches 32 --stages 4 --cuts 8,18,28 \
      --stage 0:gpus=2,zero=1,recompute=1 --stage 1:gpus=2,zero=1,recompute=1 \
      --stage 2:gpus=2,zero=1,recompute=1 --stage 3:gpus=2,zero=1 ;;
  best-rdsp)         # rdsp's fastest layout on 8x 32 GB PCIe GPUs: 4 stages x DP2, no TP,
                     # encoder on all 8 GPUs, recompute only on the first stage (4
                     # microbatches in flight), cuts balanced by cost (head ~6.6 layers)
    rdsp best-rdsp --rows 2 --microbatches 32 --stages 4 --cuts 7,17,27 \
      --colocated-vision --vision-recompute --stage 0:gpus=2,zero=1,recompute=1 \
      --stage 1:gpus=2,zero=1 --stage 2:gpus=2,zero=1 --stage 3:gpus=2,zero=1 ;;
  coloc-rdsp)        # language as shared-rdsp, encoder on all 8 GPUs
    rdsp coloc-rdsp --rows 2 --microbatches 32 --stages 2 --cuts 14 --colocated-vision \
      --stage 0:gpus=4,tp=2,zero=1,recompute=1 --stage 1:gpus=4,tp=2,zero=1 ;;
  gate)
    # 1) same layout, same first step: the reported losses must agree
    STEPS=1 "$0" shared-megatron; STEPS=1 "$0" shared-rdsp
    grep -a "iteration *1/" "$LOGS/shared-megatron.log" | grep -o "lm loss: [0-9.E+-]*"
    grep -a "^step 0" "$LOGS/shared-rdsp.log"
    # 2) rdsp against the unsplit HF model (a one-rank last stage, where the
    #    per-rank mean is the microbatch's mean)
    rdsp gate-rdsp --rows 2 --microbatches 32 --stages 2 --cuts 16 --check --steps 1 \
      --stage 0:gpus=4,tp=2,zero=2 --stage 1:gpus=2,tp=2,zero=2 ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
