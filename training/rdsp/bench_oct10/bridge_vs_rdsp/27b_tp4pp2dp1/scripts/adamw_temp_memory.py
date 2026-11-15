"""CPU check: the memory PyTorch's AdamW allocates during one step on a single
flat fp32 parameter (as DeepSpeed's ZeRO optimizers hand it one flat master
partition), foreach (the default) vs fused. Profiles the second step, when
Adam's two moments already exist.

    .venv/bin/python rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/adamw_temp_memory.py
"""

import torch
from torch.profiler import ProfilerActivity, profile

N = 25_000_000  # 100 MB of fp32
print(f"one flat fp32 parameter of {N:,} elements = {N * 4 / 2**20:.0f} MiB")
for kind in ("foreach", "fused"):
    p = torch.nn.Parameter(torch.randn(N))
    p.grad = torch.randn(N)
    opt = torch.optim.AdamW(
        [p], lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, **{kind: True}
    )
    opt.step()  # creates exp_avg and exp_avg_sq
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        opt.step()
    allocs = [
        (e.key, e.self_cpu_memory_usage)
        for e in prof.key_averages()
        if e.self_cpu_memory_usage > 1 << 20
    ]
    peak = sum(b for _, b in allocs)
    print(
        f"  {kind:8s} new memory over 1 MiB, by the op that allocated it: "
        + (", ".join(f"{k} {b / 2**20:.0f} MiB" for k, b in allocs) or "none")
        + f"  -> {peak / (N * 4):.2f}x the master size"
    )
