"""The GPU checks again, corrected (gpu_checks.py's first run, 2026-10-17 17:56, failed both):

1. remainder: each master in its own process, memory read before anything else is allocated
   (2026-10-17 18:03 this script itself read it after making the samples' fp32 copies: +0.5 B). gpu_checks.py built the fp32 engine, then the
   remainder engine in the same process; the first engine was never freed, so the second
   one's "held" memory included it (39.06 = 21.06 + 18.0 GiB). Here: memory held after 5
   steps, the optimizer step's peak and time, and the two masters compared on every 997th
   element. Writes <out>/remainder.ok on pass (the queue reads it).
2. varlen: one attention call (q, k, v strided as Hugging Face's vision attention passes
   them) and the whole Qwen3.5-27B vision encoder, varlen and SDPA both in bf16, each
   against SDPA in fp32: the first run compared two bf16 runs with each other, which says
   nothing about which is off. Reported only: the runs no longer use varlen.

    python gpu_checks2.py <exported dir> <Qwen3.5-27B snapshot dir or Hub id> <out dir>
    python gpu_checks2.py <...> <out dir> --engine fp32|remainder   (one engine, internal)
"""
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RDSP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, RDSP)
EXPORTED, SNAPSHOT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)


def gib(n):
    return round(n / 2**30, 2)


def rel(a, b):
    """|a - b| / |b|, over all elements."""
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def one_engine(master):
    """Build, train 5 steps, report; this process's only engine."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29571", RANK="0", WORLD_SIZE="1",
                      LOCAL_RANK="0")
    import deepspeed

    import train_vl
    from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory
    from ray_deepspeed_pipeline.remainder_adamw import join_master

    deepspeed.init_distributed(dist_backend="nccl")
    width, depth = 8192, 24  # 1.61B parameters
    flags = ["--stages", "1", "--stage", "0:gpus=1", "--zero", "1", "--lr", "1e-5",
             "--betas", "0.9,0.95", "--rows", "1", "--microbatches", "1",
             "--grad-dtype", "bf16", "--fused-adam", "--optimizer-group-numel", "250000000",
             "--master-dtype", master]
    conf = train_vl.ds_config(train_vl.build_parser().parse_args(flags))
    conf.update(train_micro_batch_size_per_gpu=1, train_batch_size=1)
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = torch.nn.Sequential(*[torch.nn.Linear(width, width, bias=False)
                                      for _ in range(depth)]).to(torch.bfloat16)
    before = torch.cuda.memory_allocated()  # the bf16 model itself
    e = _deepspeed_engine_factory(model, conf)
    g = torch.Generator(device="cuda").manual_seed(1)
    times, peaks = [], []
    for _ in range(5):
        x = torch.randn(16, width, generator=g, device="cuda").to(torch.bfloat16)
        e.backward(e(x).float().pow(2).mean())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        e.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
        peaks.append(torch.cuda.max_memory_allocated())
    held = torch.cuda.memory_allocated()  # before the samples below make fp32 copies
    opt = e.optimizer
    sample = []
    for m in opt.single_partition_of_fp32_groups:
        state = opt.optimizer.state.get(m, {})
        full = join_master(m.detach(), state["remainder"]) if "remainder" in state else m.detach()
        sample.append(full.float()[::997].cpu())
    torch.save(torch.cat(sample), os.path.join(OUT, f"master-sample-{master}.pt"))
    params = width * width * depth
    print("ENGINE_REPORT", json.dumps({"model_gib": gib(before), "held_gib": gib(held),
                      "held_bytes_per_param": round(held / params, 2),
                      "step_peak_gib": gib(max(peaks)),
                      "step_ms_median": round(sorted(times)[2], 1)}), flush=True)


def check_remainder():
    report = {}
    for master in ("fp32", "remainder"):
        done = subprocess.run([sys.executable, __file__, EXPORTED, SNAPSHOT, OUT, "--engine", master],
                              capture_output=True, text=True)
        lines = [l for l in done.stdout.splitlines() if l.startswith("ENGINE_REPORT ")]
        if done.returncode or not lines:
            raise RuntimeError(f"{master} engine failed: {done.stderr[-2000:]}")
        report[master] = json.loads(lines[-1].split(" ", 1)[1])
    a, b = (torch.load(os.path.join(OUT, f"master-sample-{m}.pt")) for m in ("fp32", "remainder"))
    report["master_max_abs_diff"] = float((a - b).abs().max())
    saved = report["fp32"]["held_bytes_per_param"] - report["remainder"]["held_bytes_per_param"]
    report["held_saving_bytes_per_param"] = round(saved, 2)
    # 2 bytes per parameter less; masters equal up to bf16 ties (none expected in 5 steps)
    report["pass"] = abs(saved - 2.0) < 0.3 and report["master_max_abs_diff"] < 1e-3
    print("remainder", json.dumps(report), flush=True)
    if report["pass"]:
        open(os.path.join(OUT, "remainder.ok"), "w").write(json.dumps(report) + "\n")


def vision_encoder():
    """Qwen3.5-27B's vision encoder, bf16, real weights, SDPA (as gpu_checks.py)."""
    import glob

    import huggingface_hub
    import transformers
    from safetensors import safe_open

    snapshot = SNAPSHOT if os.path.isdir(SNAPSHOT) else huggingface_hub.snapshot_download(
        SNAPSHOT, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    vision = transformers.AutoConfig.from_pretrained(snapshot).vision_config
    vision._attn_implementation = "sdpa"
    with torch.device("cuda"):
        encoder = transformers.Qwen3_5VisionModel._from_config(vision, dtype=torch.bfloat16)
    state = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(path, "pt", device="cuda") as f:
            for key in f.keys():
                if ".visual." in f".{key}":
                    state[key.split("visual.", 1)[1]] = f.get_tensor(key)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return encoder


def check_varlen():
    import copy

    import torch.nn.functional as F
    from torch.nn.attention.varlen import varlen_attn

    import train_vl
    from ray_deepspeed_pipeline.varlen_attention import use_varlen_vision_attention

    inputs, _ = next(train_vl.exported_microbatches(EXPORTED, rows=2))
    grid = torch.cat(inputs["image_grid_thw"]).cuda()
    pixels = torch.cat(inputs["pixel_values"]).cuda()
    report = {"patches": [int(x) for x in grid.prod(-1)]}

    # (a) one call, q/k/v strided as HF passes them (unbound from one qkv tensor)
    lens = grid.prod(-1).to(torch.int32)
    cu = torch.nn.functional.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
    tokens, heads, dim = int(lens.sum()), 16, 72
    qkv32 = torch.randn(tokens, 3, heads, dim, device="cuda", generator=torch.Generator(
        device="cuda").manual_seed(0))
    grad_out = torch.randn(tokens, heads, dim, device="cuda")

    def per_image(qkv):
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        out = [F.scaled_dot_product_attention(*(t[a:b].transpose(0, 1)[None] for t in (q, k, v)),
                                              scale=dim**-0.5)[0].transpose(0, 1)
               for a, b in zip(cu[:-1].tolist(), cu[1:].tolist())]
        return torch.cat(out)

    def varlen(qkv):
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        return varlen_attn(q, k, v, cu, cu, int(lens.max()), int(lens.max()), scale=dim**-0.5)

    results = {}
    for name, fn, dtype in (("ref_fp32", per_image, torch.float32),
                            ("sdpa_bf16", per_image, torch.bfloat16),
                            ("varlen_bf16", varlen, torch.bfloat16)):
        qkv = qkv32.detach().to(dtype).clone().requires_grad_(True)
        out = fn(qkv)
        out.backward(grad_out.to(dtype))
        results[name] = (out.detach(), qkv.grad)
    ref_out, ref_grad = results["ref_fp32"]
    for name in ("sdpa_bf16", "varlen_bf16"):
        out, grad = results[name]
        report[f"call_{name}_out_err"] = rel(out, ref_out)
        report[f"call_{name}_grad_err"] = rel(grad, ref_grad)

    # (b) the whole encoder, real weights and images
    bf16 = vision_encoder()
    fp32 = copy.deepcopy(bf16).float()
    holder = torch.nn.Module()
    holder.visual = copy.deepcopy(bf16)
    use_varlen_vision_attention(holder)

    def run(encoder, dtype):
        encoder.zero_grad(set_to_none=True)
        out = encoder(pixels.to(dtype), grid_thw=grid).pooler_output
        out.float().square().mean().backward()
        return out.detach().float(), torch.cat([p.grad.float().flatten()
                                                for p in encoder.parameters()])

    ref_out, ref_grad = run(fp32, torch.float32)
    for name, encoder in (("sdpa_bf16", bf16), ("varlen_bf16", holder.visual)):
        out, grad = run(encoder, torch.bfloat16)
        report[f"encoder_{name}_out_err"] = rel(out, ref_out)
        report[f"encoder_{name}_grad_err"] = rel(grad, ref_grad)
    print("varlen", json.dumps(report), flush=True)


if "--engine" in sys.argv:
    one_engine(sys.argv[sys.argv.index("--engine") + 1])
else:
    for check in (check_remainder, check_varlen):
        try:
            check()
        except Exception as e:  # report and go on
            import traceback
            traceback.print_exc()
            print(f"{check.__name__} FAILED: {e!r}", flush=True)
