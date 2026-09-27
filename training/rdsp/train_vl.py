"""Train a Qwen3-VL model with rdsp on image-caption rows.

The driver holds only an empty model skeleton; each stage reads its own
weights from the downloaded checkpoint. Rows are synthetic images of coloured
shapes with their captions, built by the model's own processor; the loss
covers the caption tokens only.

    python train_vl.py --model Qwen/Qwen3-VL-2B-Instruct --stages 4 --check
    python train_vl.py --stages 3 --stage 0:gpus=2 --stage 1:recompute=1
"""

import argparse
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


def encode_row(processor, image, caption: str, seq: int) -> tuple[dict, torch.Tensor]:
    """One row: inputs padded to `seq` tokens, labels on the caption only."""
    user = {"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": "Describe the image."}]}
    answer = {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    full = processor.apply_chat_template([user, answer], tokenize=False)
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    enc = processor(text=[full], images=[image], return_tensors="pt")
    n_prompt = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
    ids = enc["input_ids"][0]
    if len(ids) > seq:
        raise ValueError(f"a row needs {len(ids)} tokens; raise --seq")
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


def microbatches(processor, rows: int, seq: int, image_size: int, seed: int):
    """Endless (inputs, labels) microbatches: row-shaped values stacked,
    images given per row (rdsp concatenates each rank's rows)."""
    rng = random.Random(seed)
    while True:
        encoded = [encode_row(processor, *shape_image(image_size, rng), seq)
                   for _ in range(rows)]
        inputs = {k: torch.stack([r[k] for r, _ in encoded])
                  for k in ("input_ids", "attention_mask", "mm_token_type_ids")}
        inputs["pixel_values"] = [r["pixel_values"] for r, _ in encoded]
        inputs["image_grid_thw"] = [r["image_grid_thw"] for r, _ in encoded]
        yield inputs, torch.stack([labels for _, labels in encoded])


def loss_fn(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1).to(logits.device), ignore_index=-100)


def unsplit_loss(weights: str, batches) -> float:
    """Mean loss of the whole model on one GPU: the reference for --check."""
    import transformers
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        weights, dtype=torch.bfloat16).cuda().eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in batches:
            full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
            full = {k: v.cuda() for k, v in full.items()}
            losses.append(float(loss_fn(model(**full, use_cache=False).logits, labels)))
    del model
    torch.cuda.empty_cache()
    return sum(losses) / len(losses)


def ds_config(args) -> dict:
    offload = any(o.offload_optimizer for o in args.stage)
    return {
        "train_batch_size": args.rows * args.microbatches,
        "gradient_accumulation_steps": args.microbatches,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": args.zero},
        "gradient_clipping": 0.0,
        # DeepSpeed's CPU Adam under optimizer offload, PyTorch's AdamW otherwise
        "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                  "torch_adam": not offload}},
        "steps_per_print": 10**9,
    }


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
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--zero", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true",
                   help="first compare the pipeline's loss with the unsplit model on one GPU")
    args = p.parse_args(argv)

    import accelerate
    import huggingface_hub
    import ray
    import transformers

    weights = huggingface_hub.snapshot_download(
        args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    processor = transformers.AutoProcessor.from_pretrained(weights)
    data = microbatches(processor, args.rows, args.seq, args.image_size, args.seed)
    first = [next(data) for _ in range(args.microbatches)]

    reference = unsplit_loss(weights, first) if args.check else None

    config = transformers.AutoConfig.from_pretrained(weights)
    # rdsp needs tied parameters untied: embedding and head sit on different stages
    config.tie_word_embeddings = config.get_text_config().tie_word_embeddings = False
    with accelerate.init_empty_weights():
        skeleton = transformers.AutoModelForImageTextToText.from_config(config,
                                                                         dtype=torch.bfloat16)
    if args.cuts == "balanced":
        partition = rdsp.BalancedTransformerBlocks()
    elif args.cuts == "even":
        partition = rdsp.UniformTransformerBlocks()
    else:
        partition = rdsp.ExplicitCuts(tuple(int(c) for c in args.cuts.split(",")))

    ray.init(ignore_reinit_error=True)
    engine, _, _, _ = rdsp.initialize(
        model=skeleton, config=ds_config(args), loss_fn=loss_fn, weights=weights,
        pipeline_config=rdsp.PipelineConfig(stages=args.stages, partition=partition,
                                            stage_overrides=tuple(args.stage)))
    cuts = [(s.block_start, s.block_stop) for s in engine._coordinator._plan.stages]
    print(f"stages (blocks): {cuts}", flush=True)

    if reference is not None:
        piped = float(engine.eval_batch(data_iter=iter(first)))
        print(f"check: pipeline {piped:.5f}  unsplit {reference:.5f}  "
              f"rel diff {abs(piped - reference) / reference:.2e}", flush=True)

    tokens = args.rows * args.seq * args.microbatches
    batches = iter(first)
    for step in range(args.steps):
        start = time.perf_counter()
        loss = float(engine.train_batch(data_iter=batches if step == 0 else data))
        ms = (time.perf_counter() - start) * 1e3
        print(f"step {step} loss {loss:.4f} {ms:.0f} ms {tokens / ms * 1e3:.0f} tok/s", flush=True)
    return engine


if __name__ == "__main__":
    main()
