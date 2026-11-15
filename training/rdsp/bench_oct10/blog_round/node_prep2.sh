#!/bin/bash
# node_prep2.sh: one node of the two-node pair ready for the 27B MIMO-layout runs (prep2.sh runs it
# on both nodes at once): rdsp's environment and the NeMo container (bench/setup_node.sh), py-spy,
# the Qwen3.5-27B weights, and a record of the node's GPUs and EFA devices in ~/hardware.txt.
set -uo pipefail
RDSP=~/rdsp
bash "$RDSP/bench/setup_node.sh" > ~/setup.log 2>&1 || { echo "SETUP FAILED"; exit 1; }
(export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
 [ -x ~/venv/bin/py-spy ] || uv pip install --python ~/venv/bin/python py-spy) > ~/pyspy_install.log 2>&1
[ -x ~/venv/bin/py-spy ] || { echo "PYSPY INSTALL FAILED"; exit 1; }
(unset LD_LIBRARY_PATH; . ~/venv/bin/activate; HF_HOME=~/hf python -c "import huggingface_hub as h; \
 h.snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])") \
  > ~/download.log 2>&1 || { echo "DOWNLOAD FAILED"; exit 1; }
{ hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "EFA devices: $(ls /sys/class/infiniband 2>/dev/null | wc -l)"
  /opt/amazon/efa/bin/fi_info -p efa 2>/dev/null | grep -m3 -E "provider|fabric|domain"; } > ~/hardware.txt 2>&1
touch ~/.blog_node_prep_done
echo "NODE PREP DONE"
