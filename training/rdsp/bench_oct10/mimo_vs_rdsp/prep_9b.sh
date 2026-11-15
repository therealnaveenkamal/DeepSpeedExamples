#!/bin/bash
# On the H100 box: Qwen3.5-9B in MegatronMIMO's checkpoint format (the standard Bridge
# checkpoint and Bridge's exported samples already exist from the Bridge runs).
set -x
sudo docker run --gpus all --ipc=host --shm-size=64g --ulimit memlock=-1 \
  -v "$HOME/hf:/root/.cache/huggingface" -v "$HOME/mimo:/workspace" \
  -e HF_HOME=/root/.cache/huggingface -w /opt/Megatron-Bridge nvcr.io/nvidia/nemo:26.08 bash -c \
  "python -m torch.distributed.run --nproc_per_node=2 examples/conversion/convert_megatron_mimo.py \
     import --hf-model Qwen/Qwen3.5-9B --megatron-path /workspace/Qwen3.5-9B-mimo \
     --component language=tp=1,dp=1,rank_offset=0 --component images=tp=1,dp=1,rank_offset=1 \
     --torch-dtype bfloat16"
sudo chown -R "$(id -u):$(id -g)" "$HOME/hf" "$HOME/mimo"
ls -la ~/mimo/Qwen3.5-9B-mimo && echo "PREP DONE"
