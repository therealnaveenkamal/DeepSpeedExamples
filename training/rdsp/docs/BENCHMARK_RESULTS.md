# rdsp vs Megatron: pipeline-parallel training on H100

*2026-09-23 · Modal H100 80GB · one repeat per cell · raw data in `bench/results/`*


## Final conclusions (2026-09-23)

Everything below was measured on one pinned-H100 machine per comparison
(`H100!`, GPU name checked on every GPU), from the same pretrained Qwen3-0.6B
weights and the same WikiText batches, with fused kernels on both sides and
gradient precision matched.

1. **Correct.** Losses match the unsplit model and Megatron at 1, 4 and 8 GPUs
   (median per-step difference ≤ 0.13%).
2. **As fast as Megatron or faster.** In GPU-bound training (8,192-token
   microbatches, Megatron with TE CE): 1.10× / 1.11× / 1.02× at 1 / 4 / 8
   GPUs, and 1.18× for balanced 4- and 5-stage layouts. With small
   microbatches, where host overhead dominates: 1.24× / 1.30× / 1.29×.
3. **Stage-local dispatch removed the Ray overhead.** rdsp's pipeline adds no
   measurable cost relative to Megatron's up to 8 stages.
4. **Uneven stages did not pay off on Qwen3-0.6B.** A 2-GPU head stage lost
   to a plain extra stage in all three fair tests (0.93×, 0.91×, 0.97×),
   because DP=2 on small per-GPU work gains only ~1.5×. Worth retesting only
   on a model large enough that one stage's work splits efficiently.

Superseded results further down (mixed H100/H200 hardware,
`pretrain_gpt.py` comparisons, the 1-GPU cross-machine estimate) are marked
there.

**Future work:** uneven stages on a larger model; error bars from repeats;
the 32-GPU topology (needs a higher GPU limit or AWS).

## Setup

- **Model:** Qwen3-0.6B shape (28 layers, untied head), bf16, Adam lr 1e-5, no gradient clipping, no activation recomputation.
- **Batch:** 2048 tokens per microbatch (4 × 512 or 1 × 2048).
- **Same software on both sides:** one NGC 25.10 image (torch 2.9, CUDA 13.0, TransformerEngine 2.8, Megatron-Core 0.19.2, DeepSpeed pinned 53a2ac4, Ray 2.58).
- **Grid:** pipeline stages {1, 2, 4} × microbatches per step {4, 8, 16} × sequence length {512, 2048}.

**Apples to apples.** The headline compares both frameworks with fused kernels on:
- **Megatron-defaults:** TE fused norms, fused SwiGLU/RoPE, fused cross-entropy, fused Adam.
- **rdsp-fused:** the same fusions for HF models via Liger kernels (RMSNorm, SwiGLU, RoPE, cross-entropy) and fused AdamW. It trains to the same losses as unfused rdsp.

Both use flash attention. A second pair, with the optional fusions *off* on both sides, gives the same picture (`bench/results/all_cells.md`).

## Headline: raw speed as GPUs are added

Thousands of tokens trained per second. Both frameworks have fused kernels
on; 16 microbatches per step, sequence length 512.

| GPUs (stages) | Megatron (defaults) | rdsp, driver dispatch (fused, RDT) | rdsp, **stage-local** (fused) |
|---|---|---|---|
| 1 | 14.3 | 23.6 | 23.6 |
| 2 | 23.7 | 37.5 | – |
| 4 | 36.0 | 43.5 | **54.4** (+51% vs Megatron) |
| 8 | 75.9 | 36.8 | **83.6** (+10% vs Megatron) |

With the original driver dispatch, rdsp led up to 4 GPUs but its lead shrank
with every stage added. At 8 stages Megatron was about 2× faster, and rdsp was
slower than at 4. The profile below traced that to per-microbatch Ray calls.

With stage-local dispatch (2026-09-23; one run per cell), rdsp scales from 4
to 8 stages again: 8-stage throughput is 2.3× what driver dispatch managed.
It is now ahead of Megatron at both 4 and 8 stages. Two cautions:

- **Megatron still gains more per added GPU.** Going from 1 to 8 GPUs gives
  Megatron 5.3× and rdsp 3.5×. rdsp's lead still rests on its faster
  single-GPU speed, which remains unexplained, and it narrows as stages are
  added (+51% at 4, +10% at 8).
- **Stage-local only has 4- and 8-stage cells.** The 1-GPU number is the same
  code path either way.

**Why:** rdsp's driver sends every stage a separate Ray call for every
microbatch, forward and backward. That's about 256 calls per step at 8
stages, and the 0.89 s step works out to about 3.5 ms per call (not
profiled). Megatron runs its whole schedule inside each GPU's process and
sends activations directly between GPUs.

**What would fix it:** each stage runs its own 1F1B loop and exchanges
activations with its neighbours directly, instead of taking per-microbatch
orders from the driver. That's a design change.

At 8 GPUs, both frameworks split the 28 layers 4-4-4-4-3-3-3-3 (Megatron via
`--pipeline-model-parallel-layout`). The rdsp 8-GPU cell ran on an image
where Liger had upgraded Triton; the Megatron cell ran on the fixed image.

## Correction: mixed GPU types (2026-09-23)

Modal sometimes fills a `gpu="H100"` request with **H200**s. Result files
record the GPU name, and checking them shows the earlier comparisons mixed
chips:

| Results file | GPU | What it holds |
|---|---|---|
| 20260922-221532 | H100 + H200 | unfused grid (pp4 m8 seq512 cells on H200) |
| 20260922-225833, -232449, -234840, -235845 | H100 | Megatron defaults grid, fused rdsp grid (first design), pp8 cells |
| 20260923-061904 (profile), -092920 (stage-local) | **H200** | first design vs stage-local, 4 and 8 stages |
| meg-bf16-4, meg-bf16-8 | **H200** | Megatron with bf16 gradient summing |
| matched_20260923-133517, meg-bf16-1 | H100 | matched comparison below |

Withdrawn: the stage-local vs Megatron headline (rdsp on H200, Megatron on
H100), the "Megatron scales better (5.3× vs 3.5×)" finding, and the
uniform-split row of the heterogeneity table (H200 vs H100 for the others).
Identical work also varied by up to ~20% between machines of the **same**
chip, so only same-machine comparisons are reliable. `modal_bench.py` now
requests `H100!` (no substitution) and asserts the GPU name in every
container before running anything.

## Matched comparison at 1, 4 and 8 GPUs (H100, same machine)

Both frameworks start from the pretrained Qwen3-0.6B weights and use the same
WikiText-103 batches (150 steps × 16 microbatches × 2,048 tokens). They also
share the same layer split (8 GPUs: 4-4-4-4-3-3-3-3), matched fused kernels
(RoPE, SwiGLU, norm, cross-entropy, Adam, flash attention), Adam at lr 1e-5
with no clipping, and fp32 master weights. Megatron runs its own 1F1B schedule
in a lean loop (`bench/megatron_hf_loop.py`). Each GPU count ran on one machine
with both frameworks back to back; the table gives the median of steps 10–149.

| GPUs | Megatron (fp32 grad summing, default) | rdsp | rdsp faster by | Megatron, bf16 grad summing |
|---|---|---|---|---|
| 1 | 1,638 ms · 20.0k tok/s | 1,308 ms · 25.0k | 1.25× | 1,455 ms · 22.5k (separate H100) → 1.11× |
| 4 | 441 ms · 74.2k | 321 ms · 101.9k | 1.37× | ran on H200: not comparable |
| 8 | 377 ms · 86.8k | 283 ms · 115.6k | 1.33× | ran on H200: not comparable |

- **Precision mismatch:** DeepSpeed ZeRO-0 bf16 sums microbatch gradients in
  bf16; Megatron defaults to fp32 (`--grad-reduce-in-bf16` matches rdsp).
- **Timing:** Megatron is timed on the GPU process; rdsp from the driver,
  including Ray's control messages.
- **Megatron-only extras:** fused weight-gradient accumulation and Python GC
  off.
- **Unstable run:** Megatron's 1-GPU fp32 run drifted from ~1,690 to
  ~1,470 ms per step.
- **Scaling, 1 → 8 GPUs:** rdsp 4.6×, Megatron 4.3×. Each GPU count ran on a
  different machine, so these ratios carry the ±20% machine spread.

Losses match at every size. Both frameworks start at 3.3371 / 3.3369 (the HF
reference is 3.3383). Final losses:

| GPUs | Megatron fp32 | Megatron bf16 | rdsp |
|---|---|---|---|
| 1 | 2.6481 | 2.6475 | 2.6475 |
| 4 | 2.6474 | 2.6479 | 2.6494 |
| 8 | 2.6474 | 2.6478 | 2.6494 |

Per-step difference vs rdsp is 0.05% at the median and 1.5% at worst. Files:
`bench/results/matched_20260923-133517.json` and `meg-bf16-{1,4,8}_*.json`.

### Fully matched, same machine, pinned H100 (samebox_20260923-144205)

Each GPU count ran on one `H100!` container, with the GPU name checked on
every GPU before running. Megatron fp32 summing, Megatron bf16 summing and
rdsp ran back to back: same weights and data, 60 steps × 16 microbatches ×
2,048 tokens, median of steps 10–59. The gradient dtype (`main_grad`) and
layer coverage were printed and checked on every rank.

| GPUs | Megatron fp32 summing | Megatron bf16 summing | rdsp | rdsp vs matched-precision Megatron |
|---|---|---|---|---|
| 8 | 379 ms · 86.5k tok/s | 372 ms · 88.1k | 288 ms · 113.6k | **1.29×** |
| 4 | 553 ms · 59.3k | 569 ms · 57.6k | 436 ms · 75.2k | **1.30×** |
| 1 | 1,486 ms · 22.1k | 1,465 ms · 22.4k | 1,179 ms · 27.8k | **1.24×** |

The 1-GPU row is from samebox_20260923-145841: 100 steps, median of steps
10–99, with no drift between the two halves of the run. It replaces the
earlier cross-machine estimate of ~1.11×, which was wrong. Because the lead
already exists on one GPU, where no pipelining happens, it comes mostly from
per-GPU speed (HF + Liger vs Megatron + TE at 0.6B), not from rdsp's pipeline
design. The pipeline keeps the lead at 4 and 8 GPUs.

- **Precision:** Megatron fp32 vs bf16 summing differ by only −3% to +2% at
  these sizes, so rdsp's lead is not a precision artefact.
- **Loss at step 59:** Megatron 2.8560 / 2.8561 (fp32 / bf16), rdsp 2.8555.
- **Machine variance:** this 4-GPU machine was ~25% slower for both
  frameworks than the matched run's 4-GPU machine (441 / 321 ms); the ratio
  moved from 1.37× to 1.27×. So we report same-machine ratios only and make no
  claim about scaling across GPU counts.

### 1-GPU profile: where rdsp's lead comes from (profile1_20260923-152602)

Setup: one pinned H100, matched-precision Megatron vs rdsp. Both used
torch.profiler (CPU + CUDA) over optimizer steps 20–29. The kernel table is
aggregated by name; Megatron's `## Call CompiledFxGraph` annotation rows
double-count their Triton kernels and are excluded.

| per step | Megatron (bf16 summing) | rdsp |
|---|---|---|
| step time, unprofiled (steps 10–19) | 1,195 ms | 1,014 ms |
| GPU kernel time | 482 ms | 437 ms |
| GPU idle | ~713 ms (60%) | ~577 ms (57%) |
| kernel launches | ~24,500 | ~34,400 |
| cross-entropy kernels | ~82 ms (5 torch.compile'd Triton kernels, "native" fusion) | ~17 ms (Liger CE + grad scale) |
| matmul / attention / optimizer | 172 / 38 / 18 ms | 181 / 37 / 18 ms |

- **Launch-bound at this shape.** With 2,048-token microbatches on a 0.6B
  model, both frameworks leave the GPU idle more than half of each step.
  Host-side Python dispatch of 25–35k kernels per step sets the pace.
- **Where rdsp's lead comes from:** about three-quarters (~136 of 181 ms) is
  less host overhead per step; about a quarter is cheaper cross-entropy.
- **Machine variance again:** this H100 ran both frameworks faster than the
  previous 1-GPU machine (ratio 1.18× here vs 1.24× there).

**Implication:** the matched benchmark measures the frameworks' dispatch
overhead more than their GPU efficiency. Two cheap follow-ups: rerun with
larger microbatches (GPU-bound regime), and try Megatron's
`cross_entropy_fusion_impl="te"`.

### 1 GPU, large microbatches (gpubound1_20260923-153735)

Same tokens per step (32,768) and data, regrouped as 4 microbatches ×
16 rows × 512 tokens (8,192 tokens per microbatch). One pinned H100, all
three setups back to back, bf16 gradient summing throughout. 60 steps; the
median excludes the profiled steps 20–30.

| | step | tok/s | GPU kernels | GPU idle | launches/step |
|---|---|---|---|---|---|
| Megatron, native CE (default) | 587 ms | 55.8k | 385 ms | 202 ms | ~7,000 |
| Megatron, TE CE (`cross_entropy_fusion_impl="te"`) | 485 ms | 67.6k | 322 ms | 163 ms | ~7,000 |
| rdsp | 442 ms | 74.1k | 342 ms | 100 ms | ~12,000 |

- **The lead shrinks:** against Megatron's TE cross-entropy, rdsp leads by
  1.10× (1.33× against the default native CE). Megatron's pure GPU work is
  slightly less than rdsp's (322 vs 342 ms). rdsp wins on idle time only.
- **Losses:** per-step difference 0.1% median, 1.5% max.
- **Implication for the headline:** the 1.24–1.30× same-machine leads were
  measured with 2,048-token microbatches and Megatron's default CE, where
  host overhead dominates.

### Pipelined, large microbatches (gpubound48_20260923-155657)

16 microbatches × 8,192 tokens per step (the headline schedule with 4× larger
microbatches), 36 steps, median of steps 10–35. Megatron used TE CE and bf16
summing, rdsp its usual setup; both ran back to back on one pinned-H100
machine per size. GPU name, gradient dtype and layer split were checked on
every rank.

| GPUs | Megatron (TE CE, bf16) | rdsp | rdsp faster by |
|---|---|---|---|
| 1 (gpubound1) | 485 ms · 67.6k tok/s | 442 ms · 74.1k | 1.10× |
| 4 | 753 ms · 174.2k | 680 ms · 192.8k | 1.11× |
| 8 | 528 ms · 248.4k | 520 ms · 251.9k | 1.02× (tie) |

Final loss: 2.7942 for Megatron, 2.7939 (4 GPUs) and 2.7949 (8 GPUs) for rdsp.

**Conclusion so far:** in GPU-bound training rdsp is on par with Megatron or
slightly faster. rdsp's pipeline adds no measurable cost over Megatron's at up
to 8 stages.

### Core idea: a second GPU for the head stage (hetero_20260923-160952)

One 8×H100 machine (H100 checked on every GPU), all five setups back to back.
16 microbatches × 8,192 tokens, 36 steps, median of steps 10–35. Megatron
used TE CE and bf16 summing.

| Setup | GPUs | step | tok/s |
|---|---|---|---|
| Megatron, 4 stages balanced (9-9-9-1) | 4 | 631 ms | 207.8k |
| rdsp, same | 4 | 535 ms | 245.1k (1.18×) |
| Megatron, 5 stages balanced (7-7-7-6-1) | 5 | 530 ms | 247.5k |
| rdsp, same | 5 | **451 ms** | **290.6k (1.18×)** |
| rdsp, 4 stages (8-8-8-4), last stage DP=2 | 5 | 484 ms | 270.8k |

All five setups reach the same loss: median per-step difference ≤ 0.13%,
last-5-step means 2.7979–2.7999.

**Result:** heterogeneous stages did **not** pay off on Qwen3-0.6B. They were
0.93× a plain 5-stage pipeline at the same GPU count. By FLOPs the head
(1024 × 151,936) is ~10 layers of work, but in time it's ~5–6: one large,
efficient GEMM, while each layer runs many small kernels. Going from 4 to 5
balanced stages sped both frameworks up, which shows the 9-layer stages, not
the head stage, were the bottleneck at 4. So a 5th equal stage balances the
pipeline well, and DP=2 on the last stage only adds its gradient all-reduce
and split microbatches. Heterogeneity should pay off where one component
can't be spread by moving layers: a huge vocabulary relative to hidden size,
a vision encoder, a stage that runs out of memory, or few microbatches.

rdsp beats Megatron's best layouts by 1.18× at both 4 and 5 GPUs.

### Uneven stages, second chance (fewmb_20260923-164633)

The prediction was stated before running. With the fair layout 7-7-7-7, the
uneven setup has the same slowest-stage work as 5 equal stages but one fewer
stage, so it should win by ~1.14× at 4 microbatches and ~1.05× at 16. Setup:
rdsp only, one 8×H100 machine, back to back, 16 rows × 512 tokens per
microbatch, 36 steps.

| | 5 equal stages (7-7-7-6-1) | 4 stages 7-7-7-7, last on 2 GPUs | uneven vs equal |
|---|---|---|---|
| 4 microbatches | 187 ms · 174.9k tok/s | 193 ms · 169.9k | 0.97× |
| 16 microbatches | 456 ms · 287.3k | 501 ms · 261.8k | 0.91× |

**The prediction was falsified.** The 2-GPU stage was the bottleneck at
~26–28 ms per pipeline slot vs ~23 ms for the others, so DP=2 made it only
~1.5–1.6× faster. Half-size microbatches, the stage's gradient all-reduce, and
the boundary split and reassembly eat the rest. The losses match.


## Matched loss comparison: same weights, same data (2026-09-23)

Both frameworks started from the same pretrained Qwen3-0.6B checkpoint and
trained on the same real text (WikiText-103, identical batches and order):
4 stages, 8 microbatches, sequence 512, Adam lr 1e-5, bf16, no clipping,
300 steps, 4×H100. Megatron ran as a lean Megatron-Core loop
(`bench/megatron_hf_loop.py`); the HF weights were mapped into its layout by
hand, with every parameter checked to receive exactly one weight.

| | step 0 loss | step 299 | mean, last 50 | median step |
|---|---|---|---|---|
| plain HF model (reference) | 3.3523 | – | – | – |
| Megatron-Core | 3.3528 (+0.02%) | 2.6795 | 2.7124 | 317 ms (51.6k tok/s) |
| rdsp, stage-local | 3.3516 (−0.02%) | 2.6797 | 2.7129 | 247 ms (66.4k tok/s) |

The per-step difference is 0.05% at the median and 1.5% at worst, which is
bf16 noise: **both train the same model to the same place.**

*(An earlier version of this section claimed `pretrain_gpt.py` carries ~35%
overhead, from 505 ms there vs 317 ms here. Those runs were on different
machines, possibly different chips, so the claim is withdrawn; see below.)*

(Megatron divides the loss by the microbatch count in place, and the first
run's logged values shared that memory, so they were exactly 1/8 of the true
loss. They were corrected ×8, and the logging now clones the value; gradients
were unaffected.)

## Why rdsp stopped scaling: profile (8×H100, 2026-09-23)

One profiled run each at 4 and 8 stages (16 microbatches, sequence 512,
fused kernels, RDT). Per-stage GPU time came from CUDA events around every
forward, backward and optimizer call; the rest from Ray's task timeline.

| | 4 stages | 8 stages |
|---|---|---|
| step time | 742 ms | 831 ms |
| each GPU busy with Ray tasks | 430–480 ms | 285–340 ms |
| each GPU idle | 270–315 ms (37–42%) | 495–550 ms (60–66%) |
| idle for this shape in textbook 1F1B | 16% | 30% |
| layer compute per forward/backward call | ~12.8 ms | ~8.4 ms |
| Ray/RDT overhead per call | ~4.7 ms | ~4.9 ms |
| stage-to-stage hop, median / p90 | 2.3 / 7.4 ms | 2.5 / 22.9 ms |

The overhead per call breaks down as about 2 ms receiving and unpacking the
input, 1.3 ms storing the output, and 1.5 ms for the RDT send/receive tasks.

**What it isn't:**
- **Slow GPUs.** rdsp's compute per layer is the same at 1, 4 and 8 GPUs
  (~3.2 ms per layer per microbatch for forward+backward), and lower than
  Megatron's.
- **The driver's submission loop.** About 70 ms per step, off the critical path.

**What it is:** every forward and backward was its own Ray task, about
944 Ray tasks per step at 8 stages (272 calls plus their RDT transfers).

- **The overhead doesn't shrink with more stages.** Adding stages halves each
  stage's compute, but the ~5 ms per call stays.
- **More stages means more slots.** A 1F1B step runs microbatches + stages − 1
  slots back to back, and the overhead sits in every one.
- **The hop tail stalls everything.** The slowest 10% of hops take 23 ms at
  8 stages, and each one delays every stage behind it.

Megatron's step time sits at its compute × textbook bubble: each GPU runs its
own schedule and sends activations directly to its neighbour.

**The fix (implemented):** stage-local dispatch. The driver sends each stage
one command per step carrying its whole 1F1B op list. Every rank runs that
list itself and exchanges activations and gradients with its neighbours over
direct point-to-point links (NCCL on GPU, Gloo on CPU): two process groups
per rank, one per direction, alongside the stage's own DeepSpeed world.
Ray calls per step drop from ~2 × microbatches × stages to about 2 × stages
(the step plus the optimizer apply).

**Measured so far, CPU only:** same-number parity with the old path, bit for
bit. With a tiny model (so time is mostly orchestration), 16 microbatches:

| stages | driver dispatch | stage-local |
|---|---|---|
| 2 | 15.5 ms | 7.2 ms |
| 4 | 23.6 ms | 7.8 ms |
| 8 | 47.4 ms | 11.9 ms |

**H100, same model and settings (16 microbatches, sequence 512):**

| stages | driver dispatch | stage-local | Megatron-defaults |
|---|---|---|---|
| 4 | 753 ms (43.5k tokens/s) | **602 ms (54.4k)** | 909 ms (36.0k) |
| 8 | 891 ms (36.8k) | **392 ms (83.6k)** | 432 ms (75.9k) |

Correctness on GPU (8×L4, NCCL links):

- 32 of 32 tests pass: Qwen3-0.6B parity, checkpoint and recovery (including
  a retry after a mid-step failure under ZeRO-0/1/2), and every P8 row with
  per-parameter update parity. The worst update errors are unchanged
  (2.4e-5; 3.0e-4 for the folded expert stage).

## Detailed results (1, 2 and 4 GPUs)

**1. Raw throughput: rdsp is faster in 10 of 12 cells.**

| stages | rdsp (object store) / Megatron | rdsp (RDT, GPU-to-GPU) / Megatron |
|---|---|---|
| 2 | 1.21–1.39× | 1.36–1.58× |
| 4 | 0.98–1.12× | 1.12–1.21× |

**2. Almost all of that lead comes from one GPU, not from pipelining.**

- **One GPU:** rdsp-fused runs 23.6k tokens/s against Megatron-defaults' 14.3k (1.65×) on the same GPU and model.
- **Pipeline efficiency** is throughput ÷ (stages × own 1-GPU throughput), so each framework is measured against itself. On that measure Megatron is better:

| stages | Megatron-defaults | rdsp-fused, object store | rdsp-fused, RDT |
|---|---|---|---|
| 2 | 0.69–0.83 | 0.53–0.73 | 0.60–0.80 |
| 4 | 0.45–0.65 | 0.28–0.45 | 0.32–0.48 |

Megatron's pipeline loses less to idle time and communication. rdsp pays a fixed cost per microbatch: Ray dispatch, and, with the object store, a GPU→CPU→GPU copy at every boundary. The faster each GPU gets (fusions), the larger that fixed cost looms, which is why fusing *lowered* rdsp's efficiency.

**Unexplained:** we did not profile why Megatron is slower on one GPU for a model this small. Plausible causes, all unverified:
- per-step logging and synchronization (`--log-interval 1`);
- fp32 master weights and gradients in its optimizer;
- TransformerEngine overheads at hidden size 1024.

The 1-GPU gap should not be read as "rdsp's kernels are better".

**3. RDT (tensors stay on GPU) is the better transport with fusions on:** 1.06–1.18× the object store in every fused cell. Without fusions it was within noise (0.96–1.14×).

**4. The project's premise did not pay off at this scale.** Both tested ways of giving the head-heavy last stage more resources were *slower* for rdsp (4 stages, 8 microbatches, sequence 512):

| rdsp configuration | step time |
|---|---|
| uniform split, 4 GPUs | 429 ms |
| a second GPU for the last stage (5 GPUs; Megatron cannot express this) | 490 ms (RDT: 473 ms) |
| fewer layers on the last stage (cuts 9/18/27) | 492 ms (RDT: 447 ms) |

The same balanced split *helped* Megatron (493 → 455 ms). With 0.6B parameters, one extra GPU halves only a small amount of work per step, while adding a gradient all-reduce and extra dispatch. These rows need a larger model, where per-stage compute dominates the fixed costs, before any conclusion about heterogeneous allocation.

## Caveats

- **One repeat per cell.** Differences under about 10% (e.g. RDT vs object store without fusions) are within run-to-run noise.
- **One small model on one node of H100s.** The plan's L4 rows and 3-repeat headline cells were not run, to save credits.
- **No cross-framework loss check.** Megatron trained from random init on mock data; rdsp trained from the pretrained checkpoint on random tokens. Both losses decrease. rdsp's numerical correctness is established separately (P6–P8 parity tests).

## Reproduce

```
modal run bench/modal_smoke.py                      # image + both frameworks work
modal run bench/modal_bench.py                      # main grid (fusions off)
modal run bench/modal_bench.py --only context       # Megatron-defaults, balanced, capability
modal run bench/modal_bench.py --only fused         # rdsp with fused kernels
modal run bench/modal_bench.py --only pp8           # the two 8-stage cells
python bench/analyze.py bench/results/*.json
```
