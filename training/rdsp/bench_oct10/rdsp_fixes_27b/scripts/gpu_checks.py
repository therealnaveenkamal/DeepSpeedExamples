"""On the H100 box, one GPU, before the 27B runs: what the CPU tests cannot run.

1. varlen: --vision-attn varlen on Qwen3.5-27B's vision encoder (real weights, the first exported
   microbatches' images) against SDPA: features, weight gradients, and forward + backward time.
2. remainder: --master-dtype remainder against the fp32 master, through rdsp's DeepSpeed engine
   factory (ZeRO-1, bf16 gradients, fused AdamW, groups of 250M) on a 1.6B-parameter stack:
   memory held after the first step, the step's peak, the optimizer step's time, and the two
   masters after 5 steps.

Writes <out>/varlen.ok and <out>/remainder.ok when a check passes (the queue reads them).

    python gpu_checks.py <exported dir> <Qwen3.5-27B snapshot dir or Hub id> <out dir>
"""

import copy
import glob
import json
import os
import sys
import time

import torch

RDSP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, RDSP)
EXPORTED, SNAPSHOT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(SNAPSHOT):  # a Hub id: the cached snapshot
    import huggingface_hub

    SNAPSHOT = huggingface_hub.snapshot_download(
        SNAPSHOT, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]
    )


def gib(n):
    return round(n / 2**30, 2)


def timed(fn, repeat):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat * 1e3


def vision_encoder():
    import transformers
    from safetensors import safe_open

    config = transformers.AutoConfig.from_pretrained(SNAPSHOT)
    vision = config.vision_config
    vision._attn_implementation = "sdpa"
    with torch.device("cuda"):
        encoder = transformers.Qwen3_5VisionModel._from_config(vision, dtype=torch.bfloat16)
    state = {}
    for path in sorted(glob.glob(os.path.join(SNAPSHOT, "*.safetensors"))):
        with safe_open(path, "pt", device="cuda") as f:
            for key in f.keys():
                if ".visual." in f".{key}":
                    state[key.split("visual.", 1)[1]] = f.get_tensor(key)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return encoder


def check_varlen():
    import train_vl
    from ray_deepspeed_pipeline.varlen_attention import use_varlen_vision_attention

    sdpa = vision_encoder()
    holder = torch.nn.Module()
    holder.visual = copy.deepcopy(sdpa)
    use_varlen_vision_attention(holder)
    varlen = holder.visual
    data = train_vl.exported_microbatches(EXPORTED, rows=2)
    batches = []
    for _ in range(4):
        inputs, _ = next(data)
        batches.append(
            (
                torch.cat(inputs["pixel_values"]).cuda().to(torch.bfloat16),
                torch.cat(inputs["image_grid_thw"]).cuda(),
            )
        )

    def step(encoder, pixels, grid):
        encoder.zero_grad(set_to_none=True)
        out = encoder(pixels, grid_thw=grid).pooler_output
        out.float().square().mean().backward()
        return out

    def run(encoder, pixels, grid):
        out = step(encoder, pixels, grid)
        return out.detach().float(), {n: p.grad.float() for n, p in encoder.named_parameters()}

    report = {
        "images_per_microbatch": [int(g.shape[0]) for _, g in batches],
        "patches_per_microbatch": [int(g.prod(-1).sum()) for _, g in batches],
    }
    worst_out = worst_grad = 0.0
    for pixels, grid in batches:
        a, ga = run(sdpa, pixels, grid)
        b, gb = run(varlen, pixels, grid)
        worst_out = max(worst_out, float((a - b).norm() / a.norm()))
        worst_grad = max(
            worst_grad,
            max(float((ga[n] - gb[n]).norm() / ga[n].norm().clamp_min(1e-30)) for n in ga),
        )
    report["features_rel_diff_max"] = worst_out
    report["grad_rel_diff_max"] = worst_grad
    for name, encoder in (("sdpa", sdpa), ("varlen", varlen)):
        report[f"{name}_fwd_bwd_ms"] = round(
            timed(lambda: [step(encoder, p, g) for p, g in batches], repeat=5) / len(batches), 1
        )
    # bf16 kernels differ in summation order: a few 1e-3 relative is expected
    report["pass"] = worst_out < 2e-2 and worst_grad < 5e-2
    print("varlen", json.dumps(report), flush=True)
    if report["pass"]:
        open(os.path.join(OUT, "varlen.ok"), "w").write(json.dumps(report) + "\n")
    del sdpa, varlen, holder
    torch.cuda.empty_cache()


def check_remainder():
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29561", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"
    )
    import deepspeed

    import train_vl
    from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory
    from ray_deepspeed_pipeline.remainder_adamw import join_master

    deepspeed.init_distributed(dist_backend="nccl")
    width, depth = 8192, 24  # 1.61B parameters

    def engine(master):
        flags = [
            "--stages",
            "1",
            "--stage",
            "0:gpus=1",
            "--zero",
            "1",
            "--lr",
            "1e-5",
            "--betas",
            "0.9,0.95",
            "--rows",
            "1",
            "--microbatches",
            "1",
            "--grad-dtype",
            "bf16",
            "--fused-adam",
            "--optimizer-group-numel",
            "250000000",
            "--master-dtype",
            master,
        ]
        conf = train_vl.ds_config(train_vl.build_parser().parse_args(flags))
        conf.update(train_micro_batch_size_per_gpu=1, train_batch_size=1)
        torch.manual_seed(0)
        with torch.device("cuda"):
            model = torch.nn.Sequential(
                *[torch.nn.Linear(width, width, bias=False) for _ in range(depth)]
            ).to(torch.bfloat16)
        return _deepspeed_engine_factory(model, conf)

    def train(e, steps=5):
        g = torch.Generator(device="cuda").manual_seed(1)
        times, peaks = [], []
        for _ in range(steps):
            x = torch.randn(16, width, generator=g, device="cuda").to(torch.bfloat16)
            loss = e(x).float().pow(2).mean()
            e.backward(loss)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            e.step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e3)
            peaks.append(torch.cuda.max_memory_allocated())
        return times, peaks

    def masters(e):
        opt = e.optimizer
        out = []
        for m in opt.single_partition_of_fp32_groups:
            state = opt.optimizer.state.get(m, {})
            out.append(
                (
                    join_master(m.detach(), state["remainder"])
                    if "remainder" in state
                    else m.detach().float()
                ).cpu()
            )
        return out

    report, kept = {"parameters": width * width * depth}, {}
    for master in ("fp32", "remainder"):
        e = engine(master)
        times, peaks = train(e)
        report[master] = {
            "held_gib": gib(torch.cuda.memory_allocated()),
            "step_peak_gib": gib(max(peaks)),
            "step_ms_median": round(sorted(times)[len(times) // 2], 1),
        }
        kept[master] = masters(e)
        del e
        torch.cuda.empty_cache()
    diff = max(float((a - b).abs().max()) for a, b in zip(kept["fp32"], kept["remainder"]))
    report["master_max_abs_diff"] = diff
    saved = report["fp32"]["held_gib"] - report["remainder"]["held_gib"]
    report["held_saving_gib"] = round(saved, 2)
    expected = gib(2 * report["parameters"])  # 2 bytes per parameter of master weights
    report["pass"] = abs(saved - expected) < 0.25 * expected and diff < 1e-3
    print("remainder", json.dumps(report), flush=True)
    if report["pass"]:
        open(os.path.join(OUT, "remainder.ok"), "w").write(json.dumps(report) + "\n")


for check in (check_varlen, check_remainder):
    try:
        check()
    except Exception as e:  # report and go on: the queue drops a failed fix's flag
        import traceback

        traceback.print_exc()
        print(f"{check.__name__} FAILED: {e!r}", flush=True)
