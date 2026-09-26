# Training one model across many GPUs, with Ray between the pieces

*Engineering writeup — `ray_deepspeed_pipeline` (rdsp), 2026-08-30*

## What this is

rdsp is a library that takes a transformer, splits its stack of layers into
contiguous chunks, puts each chunk on its own GPU worker, and trains the whole
thing as if it were still one model. The chunks are called stages. During the
forward pass, activations flow from each stage to the next; during the
backward pass, gradients flow back the other way. Inside each stage, an
ordinary DeepSpeed engine does what DeepSpeed always does — run the layers,
hold the optimizer, manage precision and memory. Between the stages, Ray
carries the tensors and messages. Neither framework is modified: the entire
pipeline lives in our code, which sits on top of both.

The user's side of it is five lines:

```python
engine, _, _, _ = rdsp.initialize(
    model=model,                      # a normal HuggingFace model
    config=deepspeed_config,          # a normal DeepSpeed config, unchanged
    pipeline_config=rdsp.PipelineConfig(stages=2,
        partition=rdsp.UniformTransformerBlocks()),
    loss_fn=my_loss)

for _ in range(steps):
    loss = engine.train_batch()       # one call = one optimizer step
```

Why bother splitting by depth at all? Two reasons. A model too big for one
GPU has to be split *somehow*, and splitting by depth moves the least data —
only activations cross between stages, once per boundary, instead of the
constant chatter that splitting individual layers requires. And second, the
different parts of a model do very different amounts of work: in our own
measurements, the stage holding the output head ran at 85% utilization while a
middle stage sat at 29%. When each stage is its own group of workers, you can
hand the busy stage more or better GPUs. That per-stage resource knob is the
reason this project exists.

## The rules the design follows

**DeepSpeed and Ray stay untouched.** Not as a preference — as a checked
invariant. DeepSpeed is pinned to one exact source revision, our GPU
environments install it *from* that revision, and a `git diff` against the
pinned checkout proving zero changes is part of the acceptance criteria. This
forces the design to be honest: if the pipeline can't be built on the public
APIs of both frameworks, the design is wrong, and we'd rather find that out
than quietly patch around it. (We never had to patch: the public APIs were
enough everywhere.)

**The driver never touches tensors.** The user's process holds no weights and
never receives an activation or gradient. There's a hard technical reason:
when PyTorch runs a forward pass it builds an invisible record of every
operation, and running backward later *requires* that record — which cannot
leave the process it was created in. So whichever worker ran a forward must
run the matching backward, the loss must be computed on the last stage (the
labels are shipped there, not the outputs shipped back), and the only things
that ever return to the driver are loss values and ok/failed statuses.

**The API doesn't pretend.** `rdsp.initialize()` looks like
`deepspeed.initialize()` on purpose, but where DeepSpeed returns an optimizer
object, we return `None` — because the real optimizers live inside the
workers, and handing back a look-alike object whose mutations silently do
nothing would be worse than returning nothing. For the same reason
`engine.forward()`, `engine.backward()`, and `engine.step()` raise errors that
point at `train_batch()`: one training step interleaves dozens of operations
across all the workers, and there is no single "the forward" for a driver-side
method to mean. (DeepSpeed's own pipeline engine disables these same methods,
for the same reason.)

## How one training step works

**Startup (once).** The library reads the config and produces a plan: which
layers go to which stage, how many microbatches per step, which stage gets how
many GPUs. Splitting is deterministic — find the model's list of transformer
blocks, cut it at computed boundaries, give everything before the blocks
(embeddings) to the first stage and everything after (final norm, output head)
to the last. Each stage's slice of the model is copied out and shipped to its
worker, where a normal `deepspeed.initialize()` wraps it.

One check at this point matters more than it looks: some models make the
input embedding and the output head share one weight tensor. Split those onto
different machines and the sharing silently becomes two diverging copies, each
receiving only half the gradient — we measured exactly this failure in an
early prototype before writing the check. The library refuses such a split at
startup rather than training it wrong; for our test model we break the
sharing explicitly (copy the embedding into the head) before splitting.

**The step itself.** A naive pipeline would push the whole batch through
stage 0, then stage 1 — each GPU idle while the other works. Instead the
batch is cut into microbatches that stream through like an assembly line:
while stage 1 works on microbatch 1, stage 0 is already on microbatch 2. The
order of operations per stage (forwards and backwards interleaved, so that
memory for each microbatch is freed as early as possible) is computed up
front as a simple list per worker.

The driver then fires all of those operations without waiting for any of
them, and correctness comes from two things Ray guarantees: a worker executes
its messages in the order they were sent, and a message that uses another
worker's output automatically waits for that output to exist. Those two rules
replace every lock and barrier a hand-rolled version would need. We verified
the ordering empirically: each worker records the operations it actually
executed, and the recorded order matches the planned order exactly.

**Backward across the process boundary.** This is the one genuinely tricky
mechanism. Stage 1's backward pass can only walk back as far as its own
input; the rest of the model lives in another process. The handoff: stage 0
sends its output with the graph record stripped (it can't be serialized
anyway); stage 1 tells PyTorch to treat that received tensor as a starting
point for gradients; after stage 1's backward, the gradient that accumulated
on that tensor — "how the loss changes per element of stage 0's output" — is
sent back, and stage 0 resumes its own backward using it as the seed. It's
the chain rule, split across two machines.

One wrinkle: DeepSpeed's public backward entry point only accepts a single
scalar. So stage 0 hands it the number `(out · g).sum()` — a quantity
constructed so that differentiating it reproduces exactly the incoming
gradient `g`. We tested this composition in isolation before building
anything on it: against an unsplit single-process model, gradients came out
bit-identical, across every DeepSpeed memory-sharding mode (ZeRO 0 through
3), in bf16, and with gradient accumulation. Two useful discoveries came out
of that test: DeepSpeed actively *blocks* the more obvious alternative
(calling `tensor.backward(gradient=...)` directly raises an error under its
default mode), and once ZeRO shards gradients, the standard `.grad` attribute
is empty — there's a public accessor function that must be used instead.

**Who presses the optimizer button.** DeepSpeed normally counts backward
calls and runs the optimizer automatically after N of them. In a pipeline
that's a hazard: it would update stage 1's weights while stage 0 might still
fail — and weight updates on separate machines can't be rolled back. So the
library keeps DeepSpeed's counter at 1, averages the microbatch losses
itself, and after all backwards are done runs an explicit two-phase finish:
first ask every stage "are you completely done?", and only when every stage
says yes, tell every stage to run its optimizer exactly once. A failure
before that point is a clean error — nothing changed, call again. A failure
*during* the update phase means some stages updated and some didn't; the
engine then locks itself and refuses every further call until all stages are
restored from the last checkpoint, because loudly refusing beats silently
training a half-updated model.

## How we know it works

One yardstick, applied at every layer of the stack: **the split model must
produce the same numbers as the unsplit model** — the same losses and the
same gradients, not just "it runs."

- The backward handoff alone, against a single-process model: gradients
  bit-identical (4 GPUs, all ZeRO modes, bf16, accumulation).
- The full library, training a small model across two Ray workers on a
  laptop, versus the same model unsplit: losses match to 1e-7 at every one
  of five steps. This is `python demo.py` — it prints both loss columns
  side by side and takes about 30 seconds, no GPU needed.
- The full library on real hardware: Qwen3-0.6B (28 layers) split across two
  GPUs, real DeepSpeed engines in bf16 — evaluation and training losses match
  the unsplit single-GPU model, and training converges. The whole acceptance
  run takes 76 seconds.
- After all of the above: `git diff` against the pinned DeepSpeed source is
  empty, and Ray was never touched.

The library also carries a large test suite for the unglamorous parts —
the exact-consumption rule (a step uses exactly N microbatches; too few fails
cleanly, extras are left untouched), the eval path (provably changes no
weight), the API's refusals, and the guarantee that importing the library
pulls in neither ray nor deepspeed until they're actually needed.

## Saving and recovering the whole pipeline

A pipeline checkpoint has to be *one* consistent moment across every
worker. Each stage still saves itself with DeepSpeed's ordinary
`save_checkpoint` (weights, optimizer, learning-rate schedule), into its own
folder. What the library adds is the part that makes those folders belong
together:

- **A manifest, written last.** When every rank of every stage reports back,
  the driver writes one small file listing each stage, each rank, a
  fingerprint (sha256) of every file, the plan it was written by, the
  training step, and where in the data the run was. A folder without that
  file is an unfinished save and is never loaded. So a crash halfway
  through saving can never leave a half-new, half-old checkpoint that looks
  valid. The previous checkpoint simply stays the latest one.
- **Everything needed to continue *exactly*.** Each worker also saves its
  random-number state (so dropout masks continue where they left off) and the
  driver records how many batches the data loader had handed out. A test
  proves the point: train, save, keep training; then build a brand-new
  pipeline from *different* random weights, load, and train. Its losses
  match the original run bit-for-bit.
- **Loading checks before it touches anything.** A checkpoint written for a
  different split of the model, or with a missing or altered file, is
  rejected before any worker changes. Each stage re-checks its own files on
  its own machine (stages on other machines keep their files on local
  disk), and only when *every* stage has passed does any stage load.
- **Recovery rebuilds everything.** If a worker died, loading tears the whole
  pipeline down and rebuilds it (preferring the same machines, so their
  local checkpoint files are still there), then loads. This is also the only
  way out of the locked state described above.

Two correctness details surfaced while building this, both now tested:

- **A failed step must leave no trace.** If a step dies halfway through its
  backward passes, some gradients have already been added up inside
  DeepSpeed. Retrying without clearing them would count them twice. Workers
  now discard everything from an abandoned step (including one internal
  running sum that DeepSpeed's own "zero the gradients" call misses) and
  rewind their random-number state to where that step began. The retried
  step's numbers equal a clean step's exactly, under every ZeRO mode.
- **Gradient clipping is off unless global.** DeepSpeed clips gradients by
  default. Inside one stage that means clipping by *that stage's* gradient
  size, which is a different algorithm from clipping a whole model. The
  library turns DeepSpeed's default off for every stage and rejects an
  explicit clipping setting, until true whole-pipeline clipping exists.

## Giving each stage its own shape

The point of the project is that stages need not be alike: the stage with
the output head does far more work, so it should get more GPUs. A stage can
now use several GPUs in any of four ways, each an ordinary DeepSpeed
feature running inside that stage alone:

| way to use more GPUs in a stage | what each GPU gets |
|---|---|
| data parallel (with any ZeRO memory mode) | a different slice of the rows |
| tensor parallel (DeepSpeed AutoTP) | the same rows; each GPU holds a slice of every weight matrix |
| sequence parallel (DeepSpeed Ulysses) | the same rows, a different stretch of the sequence |
| expert parallel (DeepSpeed AutoEP), optionally "folded" with tensor parallel | a different slice of the rows; each GPU holds some of the experts |

**Moving data between differently shaped stages.** If stage 1 splits each
batch 2 ways and stage 2 splits it 4 ways, the halves have to become
quarters on the way. Each receiving GPU asks for exactly the pieces that
overlap its own slice and stitches them together. Nothing is ever gathered
in one place. The same routine runs in reverse for gradients.

**Keeping gradients right across the boundary.** DeepSpeed *averages*
gradients over the GPUs that split rows, but *adds* them over GPUs that
split the sequence. So a gradient passed from a 4-way stage to a 2-way
stage must be rescaled, or one stage learns at the wrong speed. The
library applies that factor at every boundary.

**How the new shapes are checked.** Every row of the support table below
runs the same acceptance test on real GPUs, with the same four checks:

- the losses match an unsplit single-process model;
- each step consumes exactly the right number of batches;
- a checkpoint loads into a fresh pipeline and continues identically;
- killing one GPU's worker leads to a clean error and then full recovery.

The tensor-, sequence- and expert-parallel rows also compare, parameter by
parameter, how much each weight moved in one step against the unsplit model.

We learned to be careful about *how* to compare. The first version of the
test used the Adam optimizer. Adam ignores a gradient that is uniformly too
big or too small, so a stage receiving half its true gradient still
trained identically. A deliberately broken build passed. The test now uses
plain SGD with momentum, where any gradient error shows up. It also refuses
to run unless each step moves the loss by at least 20 times the tolerance,
so an error can't hide inside it.

That stricter test found four real problems, all fixed:

- **Only the last microbatch counted (ZeRO 1/2).** When a stage holds
  gradients sharded across GPUs, DeepSpeed was replacing, not adding, the
  gradient of each microbatch, so the optimizer saw only the last one.
- **Tensor parallel with Qwen3.** Qwen3 normalizes each attention head
  separately. With tensor parallel, each GPU only saw its own heads' share
  of that weight's gradient, and the GPUs slowly disagreed.
- **The sequence-parallel scale factor** above was first applied the wrong
  way, doubling updates.
- **Expert-parallel "folding"** leaves each GPU holding a different partial
  gradient, which only DeepSpeed's own bookkeeping reconciles. The
  gradient handed to the previous stage was one GPU's partial. It is now
  averaged across those GPUs first.

**Support table** (every row: real DeepSpeed, Modal 8×L4, fp32 tiny models
so that "matches" can mean within 1 part in 10,000):

| row | stages (GPUs) | status | how closely it matches the unsplit model |
|---|---|---|---|
| first row (P6) | 1 · 1, Qwen3-0.6B bf16 | supported | losses within bf16 rounding |
| static boundary | 1 · 2 (split rows) · 1 | supported | losses 1e-4 |
| resource mesh | 2 · 4 · 2 | supported | losses 1e-4 |
| data parallel + ZeRO | 4 (ZeRO-2) · 2 (ZeRO-1) · 2 (ZeRO-0) | supported | losses 1e-4 |
| tensor parallel | 1 · 2 (TP) · 4 (TP×DP) | supported | losses 1e-4; every weight's update within 2.4e-5 |
| sequence parallel | 2 (SP) · 4 (SP×DP) · 1 | supported | losses 1e-4; every weight's update within 2.4e-5 |
| expert parallel + folding | 1 · 4 (EP) · 2 (EP folded with TP) | supported | losses 1e-4; weight updates within 3e-4 |
| four kinds combined, 8 GPUs | 2 (SP) · 2 (TP) · 2 (EP+TP) · 2 (TP) | passes (not a table row) | losses 1e-4; weight updates within 2.4e-5 |
| the design doc's 32-GPU topology | 8 · 8 · 8 · 8 on 4 machines | **not run** | blocked: the Modal account allows 10 GPUs at once |

Each passing row also passed the checkpoint and GPU-failure checks.

## Honest edges

- **Models the splitter understands.** Plain layer stacks, and HF
  Llama-style models (Qwen, Llama) through a dedicated stage type. Other
  architectures need their own.
- **Sequence parallelism never sits on the last stage.** The loss there
  would need labels that straddle two GPUs' stretches of the sequence.
- **Tensor and sequence parallelism can't share a stage.** This is a
  DeepSpeed limitation.
- **One stage fits on one machine** (up to 8 GPUs). That keeps each stage's
  fast communication and its checkpoint files on a single machine.
- **Checkpoints on local disks survive worker crashes, not machine loss.**
  Surviving a lost machine needs a shared checkpoint folder.
- **No gradient clipping yet** (see above).

## Who runs the schedule

The first design had the driver send every forward and every backward to
every stage as a separate Ray call, and let Ray carry the tensors between
them. It was simple and correct, but profiling on 8 GPUs showed ~5 ms of
Ray overhead per call. With hundreds of calls per step, that overhead,
not the GPUs, set the pace (see `BENCHMARK_RESULTS.md`).

Now each stage gets **one** command per step: its whole list of forwards
and backwards, in 1F1B order. Every GPU works through its list on its own
and hands activations and gradients straight to its neighbours over direct
GPU-to-GPU links (two extra NCCL groups per GPU, one per direction, next to
the stage's own DeepSpeed world). The layout conversions between
differently shaped stages and the gradient scaling are the same as before,
just computed by each GPU for its own neighbours.

Guarantees that are unchanged:

- identical numbers to the old path (tested bit for bit on CPU);
- the all-stage "everyone done → everyone step" barrier;
- the failure contract. If a step fails midway, it fails cleanly, the links
  are rebuilt, and a retry equals a clean step. A dead GPU worker still
  means recovery via `load_checkpoint`.

The driver-dispatched path was removed on 2026-09-26. Before removal, its losses
were recorded as a fixed reference that the stage-local path must reproduce
exactly (`test_losses_match_recorded_reference`).

## Moving tensors without the CPU

Stage-local dispatch moves tensors GPU-to-GPU over rdsp's own NCCL groups.
The removed driver dispatch sent a stage's output GPU → CPU → Ray's object
store → CPU → next GPU, with Ray Direct Transport (`RDSP_TRANSPORT=nccl`) as
an option that kept it on the GPU. Both went with the driver path.

## Reproduce everything

From the repository root (GPU commands are billed):

```
python demo.py             # laptop, ~30s: pipeline vs unsplit, side by side
pytest -q                  # the full CPU suite
modal run scripts/modal_tests.py --gpus L4:2 --tests tests/integration/test_p6_first_row.py
modal run scripts/modal_tests.py --gpus L4:2 --tests tests/integration/test_p7_checkpoint_gpu.py
modal run scripts/modal_tests.py --gpus L4:8 --tests tests/integration/test_heterogeneous_pipeline.py
modal run scripts/modal_cluster.py --tests "tests/integration/test_heterogeneous_pipeline.py -k four-stage-mixed"
```
