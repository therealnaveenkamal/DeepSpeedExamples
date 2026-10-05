"""Train a Qwen3-VL or Qwen3.5 model with rdsp on image-caption rows.

The driver holds only an empty model skeleton; each stage reads its own
weights from the downloaded checkpoint. Rows are synthetic images of coloured
shapes with their captions, or CORD-v2 receipts with their parse as JSON,
built by the model's own processor; the loss covers the caption tokens only.

    python train_vl.py --model Qwen/Qwen3-VL-2B-Instruct --stages 4 --check
    python train_vl.py --stages 3 --stage 0:gpus=2 --stage 1:recompute=1
    python train_vl.py --model Qwen/Qwen3.5-9B --dataset cord-v2 --seq 2048 \
        --stages 2 --stage 0:gpus=4,tp=2 --stage 1:gpus=4,tp=2 --colocated-vision
    python train_vl.py --model Qwen/Qwen3.5-9B --dataset exported:/data/cord_steps \
        --rows 2 --microbatches 32 --stages 2 --cuts 16 \
        --stage 0:gpus=4,tp=2,zero=1 --stage 1:gpus=4,tp=2,zero=1

`exported:DIR` reads samples written by bench/mimo/export_bridge_batches.py
(Megatron-Bridge's own SFT pipeline), one file per step, in order.
"""

import argparse
import glob
import itertools
import json
import os
import random
import time

import torch
import torch.nn.functional as F

import ray_deepspeed_pipeline as rdsp
from train import parse_stage

COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "white", "black"]
SHAPES = ["circle", "square", "triangle"]


def shape_image(size: int, rng: random.Random):
    """(PIL image, caption) of one shape on a plain background."""
    from PIL import Image, ImageDraw
    fg, bg = rng.sample(COLORS, 2)
    shape = rng.choice(SHAPES)
    image = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(image)
    lo, hi = size // 5, size - size // 5
    if shape == "circle":
        draw.ellipse([lo, lo, hi, hi], fill=fg)
    elif shape == "square":
        draw.rectangle([lo, lo, hi, hi], fill=fg)
    else:
        draw.polygon([(size // 2, lo), (lo, hi), (hi, hi)], fill=fg)
    return image, f"A {fg} {shape} on a {bg} background."


def synthetic_rows(image_size: int, seed: int):
    rng = random.Random(seed)
    while True:
        yield shape_image(image_size, rng)


def cord_rows(max_pixels: int, seed: int):
    """CORD-v2 receipts (train split, shuffled, repeated): the image scaled
    to at most max_pixels, and its ground-truth parse as JSON text."""
    import datasets
    data = datasets.load_dataset("naver-clova-ix/cord-v2", split="train").shuffle(seed=seed)
    for row in itertools.cycle(data):
        image = row["image"].convert("RGB")
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        if scale < 1:
            image = image.resize((max(28, int(image.width * scale)),
                                  max(28, int(image.height * scale))))
        caption = json.dumps(json.loads(row["ground_truth"])["gt_parse"], ensure_ascii=False)
        yield image, caption


def encode_row(processor, image, caption: str, seq: int) -> tuple[dict, torch.Tensor]:
    """One row: inputs padded to `seq` tokens, labels on the caption only;
    None if it does not fit."""
    user = {"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": "Describe the image."}]}
    answer = {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    full = processor.apply_chat_template([user, answer], tokenize=False)
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    enc = processor(text=[full], images=[image], return_tensors="pt")
    n_prompt = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
    ids = enc["input_ids"][0]
    if len(ids) > seq:
        return None
    pad = seq - len(ids)
    labels = ids.clone()
    labels[:n_prompt] = -100
    row = {
        "input_ids": F.pad(ids, (0, pad), value=processor.tokenizer.pad_token_id),
        "attention_mask": F.pad(enc["attention_mask"][0], (0, pad)),
        "mm_token_type_ids": F.pad(enc["mm_token_type_ids"][0], (0, pad)),
        "pixel_values": enc["pixel_values"],
        "image_grid_thw": enc["image_grid_thw"],
    }
    return row, F.pad(labels, (0, pad), value=-100)


def microbatches(processor, rows: int, seq: int, source):
    """Endless (inputs, labels) microbatches from (image, caption) pairs:
    row-shaped values stacked, images given per row (rdsp concatenates each
    rank's rows). Pairs too long for `seq` are skipped."""
    encoded_rows = (encode_row(processor, image, caption, seq) for image, caption in source)
    fitting = (row for row in encoded_rows if row is not None)
    while True:
        encoded = [next(fitting) for _ in range(rows)]
        inputs = {k: torch.stack([r[k] for r, _ in encoded])
                  for k in ("input_ids", "attention_mask", "mm_token_type_ids")}
        inputs["pixel_values"] = [r["pixel_values"] for r, _ in encoded]
        inputs["image_grid_thw"] = [r["image_grid_thw"] for r, _ in encoded]
        yield inputs, torch.stack([labels for _, labels in encoded])


_EXPORTED_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")


def exported_microbatches(directory: str, rows: int, pad_multiple: int = 0,
                          pad_per_microbatch: bool = False):
    """(inputs, labels) microbatches of `rows` consecutive samples from an
    exported directory, step file by step file: the order and grouping
    Megatron's sequential sampler gives a global microbatch. pad_multiple:
    cut each step's padding to its longest row rounded up to this multiple
    (a step's microbatches share one length). pad_per_microbatch: cut each
    microbatch to its own longest row instead, as Megatron's collate does
    (the pipeline then sends every boundary with its shape)."""
    for path in sorted(glob.glob(os.path.join(directory, "step_*.pt"))):
        samples = torch.load(path, weights_only=False)
        if len(samples) % rows:
            raise ValueError(f"{path}: {len(samples)} samples do not split into "
                             f"microbatches of {rows} rows")
        exported = samples[0]["input_ids"].shape[1]
        length = _padded_length(samples, pad_multiple, exported)
        for start in range(0, len(samples), rows):
            group = samples[start:start + rows]
            if pad_per_microbatch:
                length = _padded_length(group, pad_multiple, exported)
            inputs = {k: torch.cat([s[k][:, :length] for s in group]) for k in _EXPORTED_KEYS}
            inputs["pixel_values"] = [s["pixel_values"] for s in group]
            inputs["image_grid_thw"] = [s["image_grid_thw"] for s in group]
            yield inputs, torch.cat([s["labels"][:, :length] for s in group])


def drop_padding_mask(inputs: dict) -> dict:
    """inputs without their attention mask, for rows padded only at the end:
    causal attention already keeps real tokens off later (padding) positions,
    and without a mask attention runs its causal flash kernel. Raises
    ValueError when a row has padding before a real token."""
    mask = inputs["attention_mask"]
    lengths = mask.sum(dim=1, keepdim=True)
    if not torch.equal(mask.bool(), torch.arange(mask.shape[1]) < lengths):
        raise ValueError("--drop-padding-mask needs rows padded on the right only")
    return {k: v for k, v in inputs.items() if k != "attention_mask"}


def _padded_length(samples, pad_multiple: int, exported: int) -> int:
    """The samples' longest real row rounded up to pad_multiple, at most the
    exported length; the exported length when pad_multiple is 0."""
    if not pad_multiple:
        return exported
    longest = max(int(s["attention_mask"].sum()) for s in samples)
    return min(exported, -(-longest // pad_multiple) * pad_multiple)


# summed next-token loss over the caption tokens of some rows
token_loss_sum = rdsp.next_token_loss_sum


def token_loss_sum_liger(logits, labels):
    """token_loss_sum with Liger's cross entropy: computed in fp32 chunk by
    chunk from bf16 logits, gradient written in place, so no fp32 copy of
    the (vocabulary-wide) logits."""
    from liger_kernel.transformers import LigerCrossEntropyLoss
    return LigerCrossEntropyLoss(ignore_index=-100, reduction="sum")(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1).to(logits.device))


def apply_liger(model):
    """Liger's fused RMSNorm and SwiGLU kernels, patched into this model's
    module instances: the patched forwards travel with the stage modules to
    their workers (a class-level patch would stay in this process)."""
    from liger_kernel.transformers import _apply_liger_kernel_to_instance
    _apply_liger_kernel_to_instance(model=model, rms_norm=True, swiglu=True, rope=False,
                                    cross_entropy=False, fused_linear_cross_entropy=False)


def token_count(labels) -> int:
    return int((labels[:, 1:] != -100).sum())


# what training averages over; either way the reported step loss is the
# mean over every caption token of the step, as Megatron logs it
LOSSES = {
    "token-mean": False,       # every token of the step (calculate_per_token_loss)
    "microbatch-mean": True,   # each rank's microbatch, then the mean of those
                               # (Megatron's default)
}


def loss_function(args):
    """The stage loss for --loss / --liger / --sharded-loss."""
    if args.sharded_loss:
        token_sum = rdsp.next_token_loss_sum
    elif args.liger:
        token_sum = token_loss_sum_liger
    else:
        token_sum = token_loss_sum
    return rdsp.TokenMeanLoss(token_sum, token_count, per_microbatch=LOSSES[args.loss])


def unsplit_loss(weights: str, batches) -> float:
    """Token-mean loss of the whole model, unsplit: the reference for --check.
    Spread over the GPUs when it does not fit one (32B); freed afterwards."""
    import transformers
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        weights, dtype=torch.bfloat16, device_map="auto").eval()
    sums, counts = [], []
    with torch.no_grad():
        for inputs, labels in batches:
            full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
            full = {k: v.to(model.device) for k, v in full.items()}
            sums.append(float(token_loss_sum(model(**full, use_cache=False).logits, labels)))
            counts.append(max(token_count(labels), 1))
    del model
    torch.cuda.empty_cache()
    return sum(sums) / sum(counts)


def ds_config(args) -> dict:
    offload = any(o.offload_optimizer for o in args.stage)
    beta1, beta2 = (float(b) for b in args.betas.split(","))
    conf = {
        "train_batch_size": args.rows * args.microbatches,
        "gradient_accumulation_steps": args.microbatches,
        "bf16": {"enabled": True, **({"immediate_grad_update": False}
                                     if args.keep_bf16_grads else {})},
        # ZeRO's default 500M-element communication buckets cost ~2 GB each in
        # fp32; smaller buckets change how gradients are batched, not the math
        "zero_optimization": {"stage": args.zero, "reduce_bucket_size": int(5e7),
                              "allgather_bucket_size": int(5e7)},
        "gradient_clipping": 0.0,
        # DeepSpeed's CPU Adam under optimizer offload, PyTorch's AdamW otherwise
        "optimizer": {"type": "AdamW", "params": {
            "lr": args.lr, "betas": [beta1, beta2], "eps": args.eps,
            "weight_decay": args.weight_decay, "torch_adam": not offload}},
        "steps_per_print": 10**9,
    }
    if args.grad_dtype == "fp32":
        # accumulate and all-reduce gradients in fp32 (Megatron's
        # grad_reduce_in_fp32 with fp32 main grads)
        conf["data_types"] = {"grad_accum_dtype": "fp32"}
        conf["communication_data_type"] = "fp32"
    return conf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--stages", type=int, default=2)
    p.add_argument("--cuts", default="balanced", help="'balanced', 'even', or e.g. 9,18")
    p.add_argument("--stage", type=parse_stage, action="append", default=[],
                   help="per-stage layout, as in train.py")
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--rows", type=int, default=4, help="rows per microbatch")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--image-size", type=int, default=448, help="synthetic images' side")
    p.add_argument("--dataset", default="synthetic",
                   help="synthetic, cord-v2, or exported:DIR (export_bridge_batches.py)")
    p.add_argument("--loss", choices=tuple(LOSSES), default="token-mean",
                   help="what training averages over; the step loss printed is the "
                        "token mean either way")
    p.add_argument("--liger", action="store_true",
                   help="Liger fused RMSNorm/SwiGLU kernels and cross entropy")
    p.add_argument("--pad-multiple", type=int, default=0,
                   help="exported data: trim each step's padding to its longest row, "
                   "rounded up to this multiple (0 keeps the exported length)")
    p.add_argument("--betas", default="0.9,0.999", help="AdamW betas")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-dtype", choices=("bf16", "fp32"), default="fp32",
                   help="gradient accumulation and reduction dtype")
    p.add_argument("--attn", default="sdpa", help="attention implementation (sdpa, "
                   "flash_attention_2, ...)")
    p.add_argument("--max-pixels", type=int, default=512 * 512,
                   help="cord-v2: scale images down to at most this many pixels")
    p.add_argument("--untie-embeddings", action="store_true",
                   help="train the input embedding and the output head as two matrices; "
                        "needed when a tied pair would land on different stages")
    p.add_argument("--keep-bf16-grads", action="store_true",
                   help="keep DeepSpeed's per-parameter bf16 gradients (accumulated into "
                        "fp32 after each backward) instead of freeing each one as it lands")
    p.add_argument("--pad-per-microbatch", action="store_true",
                   help="with --pad-multiple: pad each microbatch to its own longest row "
                        "(as Megatron does) instead of each step to its longest")
    p.add_argument("--drop-padding-mask", action="store_true",
                   help="leave out the attention mask of right-padded rows, so attention "
                        "can use its causal flash kernel")
    p.add_argument("--sharded-loss", action="store_true",
                   help="token-mean loss over vocab-parallel logits: tensor-parallel "
                        "stages skip gathering the full logits")
    p.add_argument("--prefetch", action="store_true",
                   help="read and ship the next step's batches while this step runs")
    p.add_argument("--colocated-vision", action="store_true",
                   help="run the vision encoder on every GPU (rdsp.ColocatedVision)")
    p.add_argument("--vision-per-microbatch", action="store_true",
                   help="with --colocated-vision: encode and backpropagate images per "
                        "microbatch inside the pipeline (less memory, can be slower)")
    p.add_argument("--vision-compile", action="store_true",
                   help="with --colocated-vision: torch.compile the encoder's blocks")
    p.add_argument("--vision-recompute", action="store_true",
                   help="with --colocated-vision: recompute the encoder's blocks")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--zero", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true",
                   help="first compare the pipeline's loss with the unsplit model on one GPU")
    p.add_argument("--profile", action="store_true",
                   help="print per-stage phase times (RDSP_PROFILE=1; slows steps)")
    args = p.parse_args(argv)

    import accelerate
    import huggingface_hub
    import ray
    import transformers

    weights = huggingface_hub.snapshot_download(
        args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    processor = transformers.AutoProcessor.from_pretrained(weights)
    if args.dataset.startswith("exported:"):
        data = exported_microbatches(args.dataset.split(":", 1)[1], args.rows,
                                     args.pad_multiple, args.pad_per_microbatch)
    else:
        source = (cord_rows(args.max_pixels, args.seed) if args.dataset == "cord-v2"
                  else synthetic_rows(args.image_size, args.seed))
        data = microbatches(processor, args.rows, args.seq, source)
    first = [next(data) for _ in range(args.microbatches)]

    reference = unsplit_loss(weights, first) if args.check else None

    config = transformers.AutoConfig.from_pretrained(weights)
    if args.untie_embeddings:  # tied parameters must sit on one stage
        config.tie_word_embeddings = config.get_text_config().tie_word_embeddings = False
    config._attn_implementation = args.attn
    with accelerate.init_empty_weights():
        skeleton = transformers.AutoModelForImageTextToText.from_config(config,
                                                                         dtype=torch.bfloat16)
    if args.cuts == "balanced":
        # the encoder's cost per text token, measured from the first step
        from ray_deepspeed_pipeline.partition import vision_token_ratio
        partition = rdsp.BalancedTransformerBlocks(
            vision_token_ratio=vision_token_ratio(config, first))
    elif args.cuts == "even":
        partition = rdsp.UniformTransformerBlocks()
    else:
        partition = rdsp.ExplicitCuts(tuple(int(c) for c in args.cuts.split(",")))

    if args.profile:
        os.environ["RDSP_PROFILE"] = "1"  # before ray.init: actors inherit it
    loss_fn = loss_function(args)
    if args.liger:
        apply_liger(skeleton)
    ray.init(ignore_reinit_error=True)
    pipeline_config = rdsp.PipelineConfig(
        stages=args.stages, partition=partition, stage_overrides=tuple(args.stage),
        colocated_vision=(rdsp.ColocatedVision(recompute=args.vision_recompute,
                                               compile=args.vision_compile,
                                               encode_per_microbatch=args.vision_per_microbatch)
                          if args.colocated_vision else None),
        prefetch=args.prefetch)
    engine, _, _, _ = rdsp.initialize(
        model=skeleton, config=ds_config(args), loss_fn=loss_fn, weights=weights,
        pipeline_config=pipeline_config)
    print(f"stages (blocks): {engine.stage_blocks}", flush=True)

    if reference is not None:
        piped = float(engine.eval_batch(data_iter=iter(first)))
        print(f"check: pipeline {piped:.5f}  unsplit {reference:.5f}  "
              f"rel diff {abs(piped - reference) / reference:.2e}", flush=True)

    padded = args.rows * args.seq * args.microbatches
    counts = {}  # step -> (real, supervised), filled as its batches are read

    def stream():
        """Every step's batches through one iterator (prefetch reads ahead)."""
        for step in range(args.steps):
            entries = first if step == 0 else [next(data) for _ in range(args.microbatches)]
            # real tokens: what the attention mask covers (padding excluded)
            counts[step] = (sum(int(x["attention_mask"].sum()) for x, _ in entries),
                            sum(token_count(y) for _, y in entries))
            if args.drop_padding_mask:
                entries = [(drop_padding_mask(x), y) for x, y in entries]
            yield from entries

    batches = stream()
    for step in range(args.steps):
        start = time.perf_counter()
        loss = float(engine.train_batch(data_iter=batches))
        real, supervised = counts.pop(step)
        ms = (time.perf_counter() - start) * 1e3
        print(f"step {step} loss {loss:.4f} {ms:.0f} ms {real / ms * 1e3:.0f} real tok/s "
              f"{padded / ms * 1e3:.0f} padded tok/s real {real} supervised {supervised}",
              flush=True)
    return engine


if __name__ == "__main__":
    main()
