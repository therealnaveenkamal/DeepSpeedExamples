"""CPU check (Mac, DeepSpeed 0.19.6 in the project venv): rdsp's engine factory with
--grad-dtype bf16 --fused-adam under ZeRO-1, one optimizer group vs groups of at most
N elements (--optimizer-group-numel). Same training expected: prints DeepSpeed's group
count, the losses, and the largest weight difference from the one-group run.

    .venv/bin/python rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/optimizer_groups_check.py
"""

import os
import sys

os.environ.update(
    MASTER_ADDR="127.0.0.1", MASTER_PORT="29533", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"
)
import deepspeed
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4)
)  # the repo root
import train_vl
from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory

deepspeed.init_distributed(dist_backend="gloo")


def run(group_numel):
    flags = [
        "--stages",
        "1",
        "--stage",
        "0:gpus=1,zero=1",
        "--lr",
        "1e-3",
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
        str(group_numel),
    ]
    conf = train_vl.ds_config(train_vl.build_parser().parse_args(flags))
    conf.update(train_micro_batch_size_per_gpu=1, train_batch_size=1)
    conf["zero_optimization"]["stage"] = 1
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(256, 256, bias=True) for _ in range(6)]).to(
        torch.bfloat16
    )
    engine = _deepspeed_engine_factory(model, conf)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(20):
        x = torch.randn(8, 256, generator=g).to(torch.bfloat16).to(engine.device)
        loss = engine(x).float().pow(2).mean()
        engine.backward(loss)
        engine.step()
        losses.append(loss.item())
    groups = len(engine.optimizer.bit16_groups)
    return (
        type(engine.optimizer).__name__,
        groups,
        losses,
        [p.detach().float().cpu() for p in engine.module.parameters()],
    )


base = run(0)
print(
    f"one group: {base[0]}, {base[1]} group(s), loss step 1 {base[2][0]:.6f} step 20 {base[2][-1]:.6f}"
)
for n in (70_000, 140_000):
    name, groups, losses, weights = run(n)
    diff = max(float((a - b).abs().max()) for a, b in zip(base[3], weights))
    print(
        f"groups of <= {n:,}: {name}, {groups} groups, loss step 20 {losses[-1]:.6f}, "
        f"max weight difference from one group {diff:.3g}"
    )
