#!/bin/bash
# rdsp17.sh, on the third box (ops/h100/launch3.sh) with the 3-node Ray cluster up
# (bench/two_node/ray_cluster3.sh): the dense 27B over 17 GPUs, beside Megatron-MIMO's 17-GPU
# tutorial layout. The driver runs here, and rdsp's stages fill the driver's node first, so the
# vision encoder gets this box's GPU alone (stage 0, 1 GPU) and the language model runs as 8 stages
# of 2 GPUs on the two 8-GPU nodes; --cuts 0,measured. Receipt rows, 128 x 2048 a step, 2 rows a
# microbatch (1 per language GPU). RECOMPUTE=1: every block recomputed. Log: ../logs/rdsp-17gpu.log.
set -uo pipefail
export MODEL=Qwen3.5-27B EXPORT_NAME=cord_steps-Qwen3.5-27B-128
. ~/rdsp/bench_oct10/two_node_35b/scripts/common.sh   # RDSP_COMMON, vision_first, stage_flags, rdsp
export LOGS=~/rdsp/bench_oct10/gap_round/logs
mkdir -p "$LOGS"
TAG=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc
rdsp "rdsp-17gpu$TAG" --stages 9 --cuts 0,measured $(vision_first 1) $(stage_flags 1 8 2) \
  --rows 2 --microbatches 64
