# Reproducing the benchmark

Two sets of runs:
- rdsp against MegatronMIMO and Megatron-Bridge on Qwen3.5-2B and 4B, in three layouts, on one 8-GPU node, in about 3 hours. This comes first.
- The numbers in the blog post: Qwen3.5-9B, 27B and 35B-A3B on one or two 8× H100 nodes, against Megatron-Bridge, TorchTitan and MegatronMIMO. See [H100 runs](#h100-runs).

## Node

- 8 NVIDIA GPUs with at least 32 GB each. The published numbers come from an AWS g7.48xlarge: 8× RTX PRO 4500 Blackwell, 32 GB, PCIe, no NVLink.
- NVIDIA driver and CUDA, Docker with the NVIDIA container runtime.
- About 300 GB of free disk: the NeMo container, two models in Hugging Face and Megatron formats, and the exported samples.

## Setup

```bash
git clone <this repository> && cd DeepSpeedExamples/training/rdsp
bash bench/setup_node.sh
```

This builds `~/venv` (torch 2.13, DeepSpeed at commit `53a2ac44`, transformers 5.17.0, rdsp) and pulls `nvcr.io/nvidia/nemo:26.08`, which holds Megatron-Bridge and MegatronMIMO.

## Runs

`bench/mimo/bench.sh` runs any set of layouts, each pair back to back with Megatron first, and prints the table:

| Layout | Megatron side | rdsp recipe | GPUs |
|---|---|---|---|
| `noncoloc` | MegatronMIMO: vision on 1 GPU, language TP2×DP2 | `qwen35-<size>-vision1-tp2dp2.sh` | 5 |
| `tp2` | Megatron-Bridge TP2×PP2×DP2 | `qwen35-<size>-tp2pp2dp2.sh` | 8 |
| `tp4` | Megatron-Bridge TP4×PP2×DP1 | `qwen35-<size>-tp4pp2dp1.sh` | 8 |
| `coloc` | none (rdsp only): vision on every GPU, language TP2×PP2×DP2 | `qwen35-<size>-coloc-tp2pp2dp2.sh` | 8 |

```bash
MODEL=Qwen3.5-2B bash bench/mimo/bench.sh noncoloc tp2
MODEL=Qwen3.5-4B PRICE=5.32 bash bench/mimo/bench.sh all
MODEL=Qwen3.5-4B ROUNDS=2 bash bench/mimo/bench.sh tp2 tp4      # repeat for noise
DRY_RUN=1 MODEL=Qwen3.5-4B bash bench/mimo/bench.sh all         # print the plan only
```

The first run per model converts the checkpoint (the Megatron side needs its own format) and exports the training samples; rdsp replays the exact samples Megatron-Bridge's loader produces, in the same order. Both are reused afterwards. A layout without a recipe for the model is skipped (Qwen3.5-2B has no TP4: 2 key/value heads).

Each run is 50 steps (`STEPS`). Logs go to `~/bench/<model>/r<round>/<case>.log`, with `summary.txt` next to them: median step time over steps 6–50, real tokens/s, and rdsp's cost per million tokens at `PRICE` $/h for the node. Single cases still run through `bench/mimo/runs.sh <case>`.

## Expected results

| Model | Layout | Megatron | rdsp |
|---|---|---|---|
| 2B | Vision on 1 GPU + language TP2×DP2 (vs MIMO) | 9.64 s | 8.30 s |
| 2B | TP2×PP2×DP2 (vs Bridge) | 7.39 s | 5.30 s |
| 4B | TP2×PP2×DP2 (vs Bridge) | 10.04 s | 8.69 s |
| 4B | TP4×PP2×DP1 (vs Bridge) | 15.93 s | 13.29 s |

In the vs-Bridge rows the two systems don't do equal work: Megatron-Bridge pads every row to 2048 and rdsp doesn't (see Padding below).

Noise:
- Repeated runs land within about 2% (rdsp 4B TP2×PP2×DP2: 8.71 s and 8.69 s).
- The 4B Megatron numbers come from a second machine of the same type.
- Around steps 4–5, losses vary by up to 0.08 between repeated runs on both systems, because some GPU kernels aren't bit-for-bit deterministic.

## Settings held equal

- Sequence length 2048, 64 rows per step, one row per GPU per microbatch.
- AdamW, learning rate 1e-5 constant, betas 0.9 / 0.95, eps 1e-8; no weight decay, clipping or warm-up.
- bf16 compute; fp32 gradients and optimizer state, split over data parallel.
- No activation recompute, 50 steps.
- Megatron defaults that differ are overridden in `runs.sh`: clipping, warm-up, cosine decay, the multi-token-prediction head, the shuffling sampler.

Differences that remain:
- **Loss averaging.** Megatron-Bridge trains on the per-token mean. MegatronMIMO trains on per-microbatch means and can't be changed, so the rdsp run against it does the same.
- **Padding.** Megatron-Bridge pads every row to 2048 whenever it pipelines or splits experts: its training config turns on `dataset.pad_to_max_length` when PP > 1 or EP > 1 (`requires_fixed_seq_len`), whatever the run asks for. rdsp pads each microbatch to its longest row. So in `tp2` and `tp4` Megatron-Bridge computes about 2 positions per real token against rdsp's 1.1-1.2, and the vs-Bridge rows above are not equal work. MegatronMIMO pads every row to 2048 by default; `MIMO_EXTRA="--pad-to-seq-length false"` turns on its dynamic padding (9.60 s vs 9.64 s), so its row is unaffected. The H100 runs below pad every row to 2048 on every system.
- **Pipeline split.** Megatron splits the decoder layers evenly. rdsp computes its split from the model and the first batch (`--cuts balanced`).
- **Embeddings.** In the pipeline layouts, Megatron ties the input embedding and the output layer across stages; rdsp trains two copies.

## Cost

About 3 hours of node time for both models, roughly $16 at the $5.32/h spot price of a g7.48xlarge in us-east-1 (October 2026).

## H100 runs

The blog post's numbers: Qwen3.5-35B-A3B on two nodes, Qwen3.5-27B on one node, and the MegatronMIMO comparison on 9B (one node) and 27B (two nodes). The scripts are in `bench_oct10/`. Each run is 50 steps, except the MIMO 27B pair, which is 10.

### Nodes

- AWS p5.48xlarge: 8× H100 80GB with NVLink, 32 EFA network cards. Use the AWS Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 22.04) and a 500 GB disk.
- Two-node runs: both nodes in one cluster placement group, EFA enabled on all 32 cards.
- The scripts expect this folder at `~/rdsp` on every node (`ln -s "$PWD" ~/rdsp`). Run everything from `~/rdsp`, in bash.
- Two-node runs start from node 1. It needs:
  - `~/env2.sh`, which exports `PRIV1` and `PRIV2`, the two nodes' private IP addresses;
  - `~/key2.pem`, an SSH key that logs in to node 2 as `ubuntu`.

### Setup

For the two-node runs, run this on node 1. It takes about 1 h 45 min, and steps that already finished are skipped if you run it again:

```bash
bash bench_oct10/gap_round/scripts/prep.sh
```

It sets up both nodes:
- the environments: the setup above (`bench/setup_node.sh`, with the NeMo container), plus TorchTitan at commit 3c36a77 in `~/tt_venv` on a torch nightly, 2.16.0.dev20261012 (`bench_oct10/torchtitan/scripts/setup_box.sh`);
- the 35B and 27B weights;
- Megatron-Bridge's checkpoints on node 1;
- Megatron-Bridge's exported CORD-v2 rows, on both nodes: 128 a step for the 35B, 64 a step for the 27B;
- an NCCL check and the two-node Ray cluster (`bench/two_node/ray_cluster.sh up $PRIV1 $PRIV2`).

The MIMO 27B pair also needs the 27B rows at 128 a step and MIMO's own 27B checkpoint, both copied to node 2:

```bash
MODEL=Qwen3.5-27B RECIPE=qwen35_vl_27b_sft_16gpu_h100_bf16_config GBS=128 STEPS=50 \
  EXPORT_NAME=cord_steps-Qwen3.5-27B-128 bash bench/mimo/runs.sh export
MODEL=Qwen3.5-27B CONVERT_TP=4 bash bench/mimo/runs.sh convert-mimo
rsync -a -e "ssh -i ~/key2.pem" ~/mimo/cord_steps-Qwen3.5-27B-128 ~/mimo/Qwen3.5-27B-mimo ubuntu@$PRIV2:mimo/
```

The MIMO 9B runs use one node with the setup above (`bash bench/setup_node.sh`). They need the 9B weights, Megatron-Bridge's checkpoint and its exported rows, then MIMO's checkpoint:

```bash
bash bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/scripts/run.sh prep
bash bench_oct10/mimo_vs_rdsp/scripts/run.sh convert
```

### 35B-A3B, two nodes

Run these on node 1. rdsp needs the Ray cluster up.

```bash
T=bench_oct10/two_node_35b
RECOMPUTE=1 K=4 bash $T/vision_stage/run.sh rdsp    # vision encoder on its own 4-GPU stage, 6 language stages of 2 GPUs
bash $T/pp8dp2/run.sh rdsp                          # TP1 × PP8 × DP2, measured cut
bash $T/fsdp16/run.sh titan-ep8-bf16                # TorchTitan FSDP over 16 GPUs + EP8
```

`K=4` is the vision stage's width. We chose it with `$T/scripts/layout_from_log.py vision-width $T/pp8dp2/logs/rdsp.log 16`, which reads the stage times from the PP8 × DP2 log. The vision-stage layout runs out of memory unless every block is recomputed (`RECOMPUTE=1`): each language GPU holds all 256 experts of its layers.

Megatron-Bridge TP1 × PP2 × EP8 uses the layout of NVIDIA's H100 recipe for this model. It keeps NVIDIA's measured settings (layers split 17 | 23, Transformer Engine's fused cross-entropy) but drops the recipe's 12 virtual stages, because 40 layers don't split into 2 × 12 equal chunks. This is case B6-nv of `bench_oct10/gap_round/scripts/queue.sh`. On its own:

```bash
. bench_oct10/two_node_35b/scripts/common.sh       # MEG_COMMON, MEG_TP1
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
NV="\"model.pipeline_model_parallel_layout='Et*17|t*23L'\" model.cross_entropy_fusion_impl=te"
bash bench/two_node/both.sh env MODEL=Qwen3.5-35B-A3B RECIPE=qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config \
  GBS=128 STEPS=50 LOGS=$HOME/rdsp/bench_oct10/gap_round/logs NET=efa GRAD_REDUCE_FP32=false \
  CUSTOM_NAME=bridge-pp2ep8-nv CUSTOM_TP=1 CUSTOM_PP=2 \
  MEGATRON_EXTRA="$MEG_COMMON $MEG_TP1 $FAST model.expert_model_parallel_size=8 model.batch_p2p_comm=false model.batch_p2p_sync=false $NV" \
  bash bench/mimo/runs.sh custom-megatron
```

`bench/two_node/both.sh` starts the same command on both nodes. Node 2 gets only the command, so every setting has to be passed inside it. Node 2's output goes to `~/both-<time>.log` on node 2.

### 27B, one node

Use one node of the prepared pair, with the two-node Ray cluster down (`bash bench/two_node/ray_cluster.sh down $PRIV1 $PRIV2`). Megatron-Bridge's 27B checkpoint is on node 1 only. Phase A of the gap round ran Megatron-Bridge on node 1 while the other two systems ran on node 2 (`bench_oct10/gap_round/scripts/one_node.sh bridge|others`).

```bash
P8=bench_oct10/bridge_vs_rdsp/27b_tp1pp8dp1/scripts/run.sh
NAME=-fix-p8 RDSP_ARGS="--master-dtype remainder --cuts measured" bash $P8 rdsp    # rdsp PP8, every layer recomputed
bash bench_oct10/torchtitan/scripts/run.sh run qwen35_27b_cord_fsdp8_rc_bf16           # TorchTitan FSDP over 8 GPUs

# Megatron-Bridge PP8 at rdsp's measured cut, every layer recomputed, NVIDIA's speed flags
RC_ALL="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
FAST="rerun_state_machine.check_for_nan_in_loss=false ddp.check_for_nan_in_grad=false dataset.num_workers=8"
LAYOUT="\"model.pipeline_model_parallel_layout='Et|t*9|t*10|t*9|t*9|t*9|t*10|t*7L'\""
NAME=-gap-cut1 MEG_RC="$RC_ALL $LAYOUT $FAST model.batch_p2p_sync=false" bash $P8 bridge
```

The layout string is rdsp's measured cut, 1 | 9 | 10 | 9 | 9 | 9 | 10 | 7 layers: `E` is the embedding, `t*9` nine layers, `L` the loss. Megatron-Bridge's even split runs out of memory. rdsp prints its final cut near the top of its log; if a rerun measures a different cut, give Megatron-Bridge that one.

### MegatronMIMO comparison

9B, one node, 5 GPUs: one vision GPU plus a language model on 4 GPUs.

```bash
for L in 9b_v1_tp2pp2dp1 9b_v1_tp2pp1dp2; do                 # language TP2 × PP2, then TP2 × DP2
  bash bench_oct10/mimo_vs_rdsp/scripts/run.sh mimo $L
  bash bench_oct10/mimo_vs_rdsp/scripts/run.sh rdsp $L
done
```

27B, two nodes, 9 GPUs: one vision GPU plus language TP4 × PP2. MegatronMIMO's example launcher runs out of memory on the 27B as shipped. `MIMO_PATCH=1` makes `runs.sh` apply `bench_oct10/gap_round/mimo/patch_mimo.py` to a copy of the launcher in `/tmp`; MIMO's library is unchanged, and `gap_round/mimo/patch.diff` shows the result. The patch switches MIMO to BF16 gradients and an exact FP32 master copy stored as 2-byte remainders, the same settings rdsp uses here.

```bash
bash bench/two_node/both.sh env MODEL=Qwen3.5-27B GBS=128 STEPS=10 LOGS=$HOME/rdsp/bench_oct10/gap_round/logs \
  MIMO_PATCH=1 MIMO_NAME=mimo-9gpu-10 MIMO_GPUS=9 MIMO_MBS=1 MIMO_NPROC0=8 MIMO_NPROC1=1 \
  MIMO_LANGUAGE=tp=4,pp=2,dp=1,rank_offset=0 MIMO_IMAGES=tp=1,pp=1,dp=1,rank_offset=8 \
  bash bench/mimo/runs.sh mimo-layout
bash bench/two_node/ray_cluster.sh up $PRIV1 $PRIV2
bash bench_oct10/gap_round/scripts/rdsp9.sh
```

The two systems place the 9 GPUs differently:
- MIMO puts its language model on node 1's 8 GPUs and its vision GPU on node 2. MIMO prints its step times only on node 2, in `~/both-<time>.log`.
- rdsp puts the vision GPU and its first language stage on node 1 and its second stage on node 2.

### Step times

Each number is the median over steps 6-50 (MIMO 27B: steps 4-10). Compute it with `bench_oct10/gap_round/scripts/medians.py`:

```bash
python3 bench_oct10/gap_round/scripts/medians.py rdsp <log>      # or bridge, titan; MIMO logs use bridge
MEDIAN_FIRST=4 MEDIAN_MIN=5 python3 bench_oct10/gap_round/scripts/medians.py bridge <10-step log>
```

| Model | Run | Step time | Log |
|---|---|---|---|
| 35B-A3B | rdsp, vision stage of 4 GPUs + 6 language stages of 2 | 6.38 s | `two_node_35b/vision_stage/logs/rdsp-k4-rc.log` |
| 35B-A3B | Megatron-Bridge TP1 × PP2 × EP8 | 8.67 s | `gap_round/logs/bridge-pp2ep8-nv.log` |
| 35B-A3B | TorchTitan FSDP16 + EP8 | 9.11 s | `two_node_35b/fsdp16/logs/run-qwen35_35b_a3b_cord_fsdp16_ep8_bf16-node0.log` |
| 35B-A3B | rdsp TP1 × PP8 × DP2 | 9.60 s | `two_node_35b/pp8dp2/logs/rdsp.log` |
| 27B | rdsp PP8 | 11.19 s | `bridge_vs_rdsp/27b_tp1pp8dp1/logs/rdsp-fix-p8.log` |
| 27B | TorchTitan FSDP8 | 13.16 s | `torchtitan/logs/run-qwen35_27b_cord_fsdp8_rc_bf16.log` |
| 27B | Megatron-Bridge PP8 at rdsp's cut | 14.34 s | `bridge_vs_rdsp/27b_tp1pp8dp1/logs/bridge-gap-cut1.log` |
| 9B | 1 vision GPU + TP2 × PP2: rdsp / MIMO | 8.63 s / 9.32 s | `mimo_vs_rdsp/9b_v1_tp2pp2dp1/logs/rdsp-nov11-fix-t.log`, `mimo-nov11-r1.log` |
| 9B | 1 vision GPU + TP2 × DP2: rdsp / MIMO | 7.73 s / 8.41 s | `mimo_vs_rdsp/9b_v1_tp2pp1dp2/logs/rdsp-nov11-directadd.log`, `mimo-nov11.log` |
| 27B | 1 vision GPU + TP4 × PP2: rdsp / MIMO | 31.23 s / 30.82 s | `blog_round/logs27b/rdsp-27b-1rows-10steps-profile-run1.log`, `mimo-27b-1rows-10steps-run1-node2.log` |

Log paths are relative to `bench_oct10/`. rdsp numbers its steps from 0. Counting from rdsp's step 5 instead of 6 moves its medians by at most 0.01 s; the 9B rdsp numbers above count from step 5.

### Settings held equal

- Sequence length 2048, every row padded to 2048 on every system. rdsp and TorchTitan replay Megatron-Bridge's exported CORD-v2 rows in the same order.
- 128 rows a step on two nodes, 64 on one node.
- Loss: the mean over every supervised token in the step. In the MIMO comparisons it is averaged per microbatch, because MIMO does that and can't be changed.
- AdamW, learning rate 1e-5 constant, betas 0.9 / 0.95, eps 1e-8; no weight decay, clipping or warm-up.
- Gradients and master copy:
  - rdsp and Megatron-Bridge use BF16 gradients and an exact FP32 master copy stored as the BF16 weight plus a 2-byte remainder (rdsp `--master-dtype remainder`; Megatron-Bridge `store_param_remainders` in `bridge_vs_rdsp/27b_tp4pp2dp1/scripts/meg_extra_bf16.txt`).
  - TorchTitan reduces gradients in BF16 and keeps its own FP32 master copy.
  - MIMO 9B uses MIMO's fixed FP32 gradients and master copy, and rdsp matches it (`--grad-dtype fp32`).
- 35B-A3B: the router's load-balancing loss is off on every system (each defines it differently, so only "off" is equal). The multi-token-prediction layer is off, and no tokens are dropped.

Differences that remain:
- **Recompute.**
  - 35B-A3B: rdsp's vision-stage layout recomputes every block; rdsp PP8 × DP2 recomputes only the vision encoder; Megatron-Bridge and TorchTitan recompute nothing.
  - 27B: every system recomputes every layer.
  - MIMO comparisons: no recompute.
- **One run each.** rdsp's 27B run came from a different p5.48xlarge (2026-10-17) than Megatron-Bridge's and TorchTitan's (2026-10-19), because a same-node rerun failed on a full disk.

### Cost

Each 50-step run takes 8-18 minutes, including startup; Megatron-Bridge's first step alone takes 1-4 minutes. With the setup, all the runs above come to about 4-5 hours of the two-node pair. Rough estimate: $250-300 at $56-64/h spot for two p5.48xlarge (us-west-2 and us-east-2, October 2026).
