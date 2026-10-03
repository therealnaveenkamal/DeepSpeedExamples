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


def exported_microbatches(directory: str, rows: int, pad_multiple: int = 0):
    """(inputs, labels) microbatches of `rows` consecutive samples from an
    exported directory, step file by step file: the order and grouping
    Megatron's sequential sampler gives a global microbatch. pad_multiple:
    cut each step's padding to its longest row rounded up to this multiple
    (a step's microbatches share one length)."""
    for path in sorted(glob.glob(os.path.join(directory, "step_*.pt"))):
        samples = torch.load(path, weights_only=False)
        if len(samples) % rows:
            raise ValueError(f"{path}: {len(samples)} samples do not split into "
                             f"microbatches of {rows} rows")
        length = samples[0]["input_ids"].shape[1]
        if pad_multiple:
            longest = max(int(s["attention_mask"].sum()) for s in samples)
            length = min(length, -(-longest // pad_multiple) * pad_multiple)
        for start in range(0, len(samples), rows):
            group = samples[start:start + rows]
            inputs = {k: torch.cat([s[k][:, :length] for s in group]) for k in _EXPORTED_KEYS}
            inputs["pixel_values"] = [s["pixel_values"] for s in group]
            inputs["image_grid_thw"] = [s["image_grid_thw"] for s in group]
            yield inputs, torch.cat([s["labels"][:, :length] for s in group])


def token_loss_sum(logits, labels):
    """Summed next-token loss over the caption tokens of some rows."""
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1).to(logits.device), ignore_index=-100,
                           reduction="sum")


def token_count(labels) -> int:
    return int((labels[:, 1:] != -100).sum())


def microbatch_mean_loss(logits, labels):
    """Mean over one rank's caption tokens in one microbatch; the step loss
    is then the mean over microbatches and data-parallel ranks (Megatron's
    default, calculate_per_token_loss=False)."""
    return token_loss_sum(logits, labels) / max(token_count(labels), 1)


LOSSES = {
    # averaged over every caption token of the step (calculate_per_token_loss)
    "token-mean": rdsp.TokenMeanLoss(token_loss_sum, token_count),
    "microbatch-mean": microbatch_mean_loss,
}


def unsplit_loss(weights: str, batches, loss: str) -> float:
    """Loss of the whole model, unsplit: the reference for --check (for
    microbatch-mean, equal to the pipeline's when the last stage has one
    data-parallel rank). Spread over the GPUs when it does not fit one
    (32B); freed afterwards."""
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
    if loss == "token-mean":
        return sum(sums) / sum(counts)
    return sum(s / c for s, c in zip(sums, counts)) / len(sums)


def ds_config(args) -> dict:
    offload = any(o.offload_optimizer for o in args.stage)
    beta1, beta2 = (float(b) for b in args.betas.split(","))
    conf = {
        "train_batch_size": args.rows * args.microbatches,
        "gradient_accumulation_steps": args.microbatches,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": args.zero},
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
    p.add_argument("--loss", choices=tuple(LOSSES), default="token-mean")
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
    p.add_argument("--colocated-vision", action="store_true",
                   help="run the vision encoder on every GPU (rdsp.ColocatedVision)")
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
                                     args.pad_multiple)
    else:
        source = (cord_rows(args.max_pixels, args.seed) if args.dataset == "cord-v2"
                  else synthetic_rows(args.image_size, args.seed))
        data = microbatches(processor, args.rows, args.seq, source)
    first = [next(data) for _ in range(args.microbatches)]

    reference = unsplit_loss(weights, first, args.loss) if args.check else None

    config = transformers.AutoConfig.from_pretrained(weights)
    # rdsp needs tied parameters untied: embedding and head sit on different stages
    config.tie_word_embeddings = config.get_text_config().tie_word_embeddings = False
    config._attn_implementation = args.attn
    with accelerate.init_empty_weights():
        skeleton = transformers.AutoModelForImageTextToText.from_config(config,
                                                                         dtype=torch.bfloat16)
    if args.cuts == "balanced":
        # the vision encoder's cost scales with the patches it sees per text token
        patches = sum(p.shape[0] for p in first[0][0]["pixel_values"])
        partition = rdsp.BalancedTransformerBlocks(
            vision_token_ratio=patches / (args.rows * args.seq))
    elif args.cuts == "even":
        partition = rdsp.UniformTransformerBlocks()
    else:
        partition = rdsp.ExplicitCuts(tuple(int(c) for c in args.cuts.split(",")))

    if args.profile:
        import os
        os.environ["RDSP_PROFILE"] = "1"  # before ray.init: actors inherit it
    ray.init(ignore_reinit_error=True)
    engine, _, _, _ = rdsp.initialize(
        model=skeleton, config=ds_config(args), loss_fn=LOSSES[args.loss], weights=weights,
        pipeline_config=rdsp.PipelineConfig(
            stages=args.stages, partition=partition, stage_overrides=tuple(args.stage),
            colocated_vision=(rdsp.ColocatedVision(recompute=args.vision_recompute)
                              if args.colocated_vision else None)))
    cuts = [(s.block_start, s.block_stop) for s in engine._coordinator._plan.stages]
    print(f"stages (blocks): {cuts}", flush=True)

    if reference is not None:
        piped = float(engine.eval_batch(data_iter=iter(first)))
        print(f"check: pipeline {piped:.5f}  unsplit {reference:.5f}  "
              f"rel diff {abs(piped - reference) / reference:.2e}", flush=True)

    padded = args.rows * args.seq * args.microbatches
    for step in range(args.steps):
        entries = first if step == 0 else [next(data) for _ in range(args.microbatches)]
        # real tokens: what the attention mask covers (padding excluded)
        real = sum(int(x["attention_mask"].sum()) for x, _ in entries)
        supervised = sum(token_count(y) for _, y in entries)
        start = time.perf_counter()
        loss = float(engine.train_batch(data_iter=iter(entries)))
        ms = (time.perf_counter() - start) * 1e3
        print(f"step {step} loss {loss:.4f} {ms:.0f} ms {real / ms * 1e3:.0f} real tok/s "
              f"{padded / ms * 1e3:.0f} padded tok/s real {real} supervised {supervised}",
              flush=True)
    return engine


if __name__ == "__main__":
    main()
