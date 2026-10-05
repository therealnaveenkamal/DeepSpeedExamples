# rdsp release polish

## Goal

Make rdsp ready for PyTorch-blog readers. A reader can:

1. reproduce the benchmark on one 8-GPU node,
2. train their own model from a recipe,
3. understand the design,
4. trust the code.

No new features. No GPU spend. CPU suite stays green.

## Writing rules (all docs)

- Short sentences, active voice, numbers over adjectives.
- Every claim either measured (with the setup) or marked as an estimate.
- No filler, no rhetorical framing, no em-dash asides, no "not X but Y".
- Define TP, PP, DP, cut, stage once, in the README; use them consistently after.

## 1. Code (trust)

- Review `src/` and the entry scripts with `mattpocock-skills:code-review`
  against `docs/CODING_STANDARDS.md` plus the smell baseline. Fixed point: the
  first rdsp commit on `feature/rdsp`.
- Fix confirmed findings, test-first. Record skipped ones with a reason.
- Public API contract: export what every layout needs (`StageOverride`;
  `ConnectionOverride` if used). Each public name gets a docstring stating
  inputs, guarantees, and the errors it raises.
- `docs/CONTRACTS.md`: the public API in one page: what each call guarantees,
  supported layouts (from `support_matrix.py`), failure semantics, what is
  not supported.

## 2. Recipes (train your own model)

- `recipes/`: one runnable script per validated (model, layout):
  - `qwen35-2b-vision1-tp2dp2.sh` (MIMO layout)
  - `qwen35-2b-tp2pp2dp2.sh`
  - `qwen35-4b-tp2pp2dp2.sh`
  - `qwen35-4b-tp4pp2dp1.sh`
  - `qwen3-text-pp.sh` (text-only, `train.py`)
- Each recipe: header comment with GPUs, memory, measured step time; flags only
  through `train_vl.py` / `train.py`. Defaults: `--cuts balanced` and the fast
  flags.
- `docs/GETTING_STARTED.md`: install, run a recipe, change model / data /
  layout, read the logs, common failures (OOM, KV heads vs TP).
- Test: every recipe's flags parse with the script's argparse (CPU, no run).

## 3. Benchmark reproduction

- Move node setup into the repo: `bench/setup_node.sh` (from the scratch
  setup: venv, torch, DeepSpeed pin, Liger, NeMo container).
- `bench/mimo/runs.sh` stays the single entry point; add the final cases
  (balanced cuts, current flags) and drop stale ones.
- `bench/mimo/summarize.py` reads the run logs and prints the results table.
- `docs/REPRODUCE.md`: node requirements, setup, convert / export, run each
  pair, summarize, expected numbers and noise.

## 4. Docs (understand)

- README: what rdsp is, install, quickstart (one recipe), **Results** section,
  links to the other docs. Drop stale Qwen3-0.6B / H100 results.
- Results section: the table below, plus hardware, software versions,
  settings held equal, and remaining differences.

  | Model | Layout | Megatron | rdsp | Step time |
  |---|---|---|---|---|
  | 2B | Vision on 1 GPU + language TP2×DP2 (vs MIMO) | 9.64 s | 8.30 s | −14% |
  | 2B | TP2×PP2×DP2 (vs Bridge) | 7.39 s | 5.30 s | −28% |
  | 4B | TP2×PP2×DP2 (vs Bridge) | 10.04 s | 8.69 s | −13% |
  | 4B | TP4×PP2×DP1 (vs Bridge) | 15.93 s | 13.29 s | −17% |

  Hardware: AWS g7.48xlarge, 8× NVIDIA RTX PRO 4500 Blackwell (32 GB),
  PCIe, no NVLink.
- `docs/ENGINEERING.md`: add vision placements (stage, vision-only stage,
  colocated), balanced cuts and the vision cost estimate, memory (12 bytes per
  parameter, per-microbatch padding), prefetch, split-vocabulary loss.
- `docs/BENCHMARK_RESULTS.md`: replace with the current results and the
  history of what moved the numbers.

## Order

1 → 2 → 3 → 4. Docs last so they describe the final code.

## Done when

- Review findings fixed or recorded; CPU suite green.
- Every recipe parses; every doc command matches a real flag.
- README results table matches the logs in `bench/results/`.
