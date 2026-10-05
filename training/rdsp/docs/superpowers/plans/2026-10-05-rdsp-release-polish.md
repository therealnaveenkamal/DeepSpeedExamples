# rdsp release polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rdsp reproducible, usable, understandable and reviewable for PyTorch-blog readers.

**Architecture:** Review and fix the library first, then add recipes and the benchmark reproduction kit on top of the existing entry points (`train_vl.py`, `train.py`, `bench/mimo/runs.sh`), then rewrite the docs against the final code. Docs-as-tests keep commands and flags honest.

**Tech Stack:** Python 3.12, PyTorch, DeepSpeed, Ray, pytest, ruff, bash.

**Spec:** `docs/superpowers/specs/2026-10-05-rdsp-release-polish-design.md`

## Global Constraints

- No new features. No GPU spend. CPU suite (`pytest tests --ignore=tests/integration/test_vl_layouts_gpu.py --ignore=tests/integration/test_p7_checkpoint_gpu.py`) green after every task.
- Writing rules (all docs): short sentences, active voice, numbers over adjectives; every claim measured (with setup) or marked as an estimate; no filler, no rhetorical framing, no em-dash asides, no "not X but Y"; define TP, PP, DP, cut, stage once in the README.
- No MegatronMIMO / Megatron mentions in rdsp code comments (docs and bench may name them).
- Results are the four rows in the spec, on AWS g7.48xlarge, 8× RTX PRO 4500 Blackwell (32 GB), PCIe, no NVLink.
- Commit after each task: edit in `rdsp/`, `rsync` into `DeepSpeedExamples/training/rdsp/` (delete removed files with `git rm`), commit there. The user pushes.

## Review Focus

- A recipe launched on a node with fewer GPUs than it needs: expect the existing `ValidationError("stage N: cannot place ...")`, quoted in GETTING_STARTED's common failures (Task 5).
- TP that does not divide the model's key/value heads (2B with TP4): expect a `ValidationError` naming the head count before any actor starts. Test in Task 3.
- `--cuts balanced` on a text-only model: `vision_token_ratio` returns 1.0. Test in Task 3.
- A doc command whose flag no longer exists: caught by `test_docs_flags` (Task 10).
- README results drifting from the logs: Task 8 copies numbers from `bench/mimo/summarize.py` output, not by hand.

---

### Task 1: Code review

**Files:**
- Create: `docs/superpowers/reviews/2026-10-05-code-review.md`

- [ ] **Step 1:** Invoke `mattpocock-skills:code-review` on `DeepSpeedExamples`, fixed point `5a2b7f9^`, scope `training/rdsp/src`, `training/rdsp/train.py`, `training/rdsp/train_vl.py`, `training/rdsp/bench/mimo`. Standards source: `docs/CODING_STANDARDS.md` plus the skill's smell baseline. Spec source: `docs/ENGINEERING.md`.
- [ ] **Step 2:** Write the review doc: one line per finding (`file:line`, standard or smell, verdict `fix` / `skip: <reason>`).
- [ ] **Step 3:** Commit the review doc.

### Task 2: Fix review findings

**Files:** as named by each `fix` finding in the review doc; tests next to the existing ones in `tests/unit/`.

- [ ] **Step 1:** For each `fix` finding with behaviour impact: write the failing test, run it (FAIL), fix, run it (PASS). Pure renames, comment and dead-code fixes need no new test.
- [ ] **Step 2:** Run the CPU suite: all pass. `ruff check src tests train.py train_vl.py bench`: clean.
- [ ] **Step 3:** Mark each finding `fixed` in the review doc. Commit.

### Task 3: Public API contract

**Files:**
- Modify: `src/ray_deepspeed_pipeline/__init__.py`, `src/ray_deepspeed_pipeline/compiler.py`, public docstrings in `api.py`, `config.py`, `engine.py`, `losses.py`, `vocab_parallel.py`, `partition.py`
- Create: `docs/CONTRACTS.md`, `tests/unit/test_public_api.py`

**Interfaces:**
- Produces: `rdsp.StageOverride`, `rdsp.ConnectionOverride` exported; compiler raises `ValidationError` when a stage's `tp` does not divide `num_key_value_heads`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_public_api.py`:

```python
def test_layout_types_are_public():
    assert {"StageOverride", "ConnectionOverride"} <= set(rdsp.__all__)

def test_every_public_name_states_its_contract():
    for name in rdsp.__all__:
        assert (getattr(rdsp, name).__doc__ or "").strip(), name

def test_tp_must_divide_key_value_heads():
    # tiny Qwen3.5 config with num_key_value_heads=2, stage tp=4
    with pytest.raises(ValidationError, match="key/value heads"):
        lower(model, PipelineConfig(stages=2, partition=ExplicitCuts((3,)),
              stage_overrides=(StageOverride(stage=0, num_gpus=4, tp=4),)), DS)

def test_vision_token_ratio_is_neutral_without_an_encoder():
    assert vision_token_ratio(SimpleNamespace(), []) == 1.0
```

- [ ] **Step 2:** Run: `pytest tests/unit/test_public_api.py -v`. Expected: first and third FAIL.
- [ ] **Step 3:** Export the two types; add the head check in `compiler.lower` (read `num_key_value_heads` from the text config; skip when absent); give every public name a docstring: inputs, guarantees, errors raised.
- [ ] **Step 4:** Write `docs/CONTRACTS.md`: per public name, its guarantees and errors; supported layouts (from `support_matrix.ROWS`); failure semantics (from README "Failure semantics"); not supported (colocated on 32 GB without recompute, TP above key/value heads, cross-stage tied embeddings).
- [ ] **Step 5:** Run the CPU suite: all pass. Commit.

### Task 4: Recipes

**Files:**
- Modify: `train_vl.py`, `train.py` (expose `build_parser() -> argparse.ArgumentParser`, used by their `main`)
- Create: `recipes/qwen35-2b-vision1-tp2dp2.sh`, `recipes/qwen35-2b-tp2pp2dp2.sh`, `recipes/qwen35-4b-tp2pp2dp2.sh`, `recipes/qwen35-4b-tp4pp2dp1.sh`, `recipes/qwen3-text-pp.sh`, `tests/unit/test_recipes.py`
- Delete: `run.sh` (replaced by `recipes/qwen3-text-pp.sh`)

**Interfaces:**
- Produces: `train_vl.build_parser()`, `train.build_parser()`; recipes call `python "$ROOT/train_vl.py" ... "$@"` (or `train.py`).

- [ ] **Step 1: Write the failing test** `tests/unit/test_recipes.py::test_every_recipe_parses`: for each `recipes/*.sh`, join continuation lines, take the tokens after `train_vl.py` / `train.py` up to `"$@"` (via `shlex`), and assert `build_parser().parse_args(tokens)` succeeds; assert each recipe header names `GPUs:` and `Measured:`.
- [ ] **Step 2:** Run it: FAIL (no recipes, no `build_parser`).
- [ ] **Step 3:** Add `build_parser()` to both scripts. Write the five recipes. Each header: model, layout, GPUs, peak memory, measured step time from `bench/results/box_runs_20261005/` (2B vision1 8.30 s, 2B TP2×PP2×DP2 5.30 s, 4B TP2×PP2×DP2 8.69 s, 4B TP4×PP2×DP1 13.29 s). VL recipes use `--cuts balanced --prefetch --sharded-loss --drop-padding-mask --pad-per-microbatch` plus the per-layout flags in the results page's "Reproducing" section; data via `--dataset exported:DIR` or `cord-v2`. `qwen3-text-pp.sh` carries over `run.sh`.
- [ ] **Step 4:** Run the test and the CPU suite: PASS. Commit.

### Task 5: Getting-started guide

**Files:**
- Create: `docs/GETTING_STARTED.md`

- [ ] **Step 1:** Write: install (`uv pip install -e .` plus DeepSpeed pin from `pyproject.toml`), run a recipe, change the model / dataset / layout (`--stage N:gpus=..,tp=..,zero=..`, `--cuts`), read the step log line (`step N loss .. ms .. real tok/s`), common failures: not enough GPUs (quote the `ValidationError`), out of memory (fewer layers on the first stage via `--cuts`, `recompute=1`), TP above key/value heads (quote Task 3's error).
- [ ] **Step 2:** Commit.

### Task 6: Benchmark kit

**Files:**
- Create: `bench/setup_node.sh` (from the session's scratch `setup.sh`: venv, torch 2.13, DeepSpeed pin, transformers 5.17.0, ray, Liger, flash-linear-attention, causal-conv1d, `docker pull nvcr.io/nvidia/nemo:26.08`), `tests/unit/test_bench_summarize.py`
- Modify: `bench/mimo/runs.sh` (rdsp cases use `--cuts balanced` and the current flags; add `tp4pp2-megatron`, `tp4pp2-rdsp`; delete `pp4-*`, `best-rdsp`, `coloc-rdsp`, `gate`), `bench/mimo/summarize.py` (`RUNS` map to the final case names; prints the results table)

**Interfaces:**
- Produces: `runs.sh` cases `convert`, `export`, `noncoloc-mimo`, `noncoloc-rdsp`, `shared-megatron`, `shared-rdsp`, `tp4pp2-megatron`, `tp4pp2-rdsp`; `summarize.py LOGS` prints one row per pair.

- [ ] **Step 1: Write the failing test** `test_bench_summarize.py::test_pairs_megatron_and_rdsp_logs`: write two synthetic logs (Megatron `iteration` lines, rdsp `step` lines with `real` tokens) for `shared-*` into `tmp_path`; assert the printed table has the pair with both medians.
- [ ] **Step 2:** Run it: FAIL.
- [ ] **Step 3:** Update `summarize.py` and `runs.sh`; add `bench/setup_node.sh`. `bash -n` both scripts.
- [ ] **Step 4:** Run the test and the CPU suite: PASS. Commit.

### Task 7: Reproduction guide

**Files:**
- Create: `docs/REPRODUCE.md`

- [ ] **Step 1:** Write: node requirements (8 GPUs, ≥32 GB each, Docker with NVIDIA runtime, ~300 GB disk), `bench/setup_node.sh`, then per model `runs.sh convert`, `runs.sh export`, each Megatron / rdsp pair, `summarize.py`; expected numbers (the four results rows) and noise (median of steps 6–50; repeat runs within ~2%; steps 4–5 loss varies up to ~0.08 on both systems); total time and cost (~3 h node time, $5.32/h spot at time of writing).
- [ ] **Step 2:** Commit.

### Task 8: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1:** Rewrite to: one-paragraph summary; terms (TP, PP, DP, stage, cut); install; quickstart (one recipe); **Results** (the spec's table, hardware, software versions from `pyproject.toml` / container tag, settings held equal, remaining differences); supported layouts (link CONTRACTS); docs index (GETTING_STARTED, REPRODUCE, ENGINEERING, CONTRACTS, BENCHMARK_RESULTS); limitations. Drop the Qwen3-0.6B / H100 results.
- [ ] **Step 2:** Check every number against `summarize.py` output on `bench/results/box_runs_20261005/` and `box_runs_20261004b/`. Commit.

### Task 9: Design and results docs

**Files:**
- Modify: `docs/ENGINEERING.md`, `docs/BENCHMARK_RESULTS.md`

- [ ] **Step 1:** ENGINEERING: add sections for vision placements (on the first stage, vision-only stage, colocated), balanced cuts and the vision cost estimate (formula from `partition.vision_token_ratio`), memory (12 bytes per parameter, freed bf16 gradients, per-microbatch padding), prefetch, split-vocabulary loss.
- [ ] **Step 2:** BENCHMARK_RESULTS: replace with the four results, per-layout settings, and the history of what moved the numbers (the step-time chain from the results page).
- [ ] **Step 3:** Commit.

### Task 10: Docs-as-tests and final check

**Files:**
- Create: `tests/unit/test_docs_flags.py`

- [ ] **Step 1: Write the test** `test_every_documented_flag_exists`: collect `--[a-z-]+` tokens from fenced code blocks in `README.md` and `docs/*.md` that invoke `train_vl.py`, `train.py` or `recipes/`; assert each is an option of `train_vl.build_parser()` or `train.build_parser()`.
- [ ] **Step 2:** Run it; fix any doc it flags. PASS.
- [ ] **Step 3:** Run the CPU suite and `ruff`: clean. Commit.
