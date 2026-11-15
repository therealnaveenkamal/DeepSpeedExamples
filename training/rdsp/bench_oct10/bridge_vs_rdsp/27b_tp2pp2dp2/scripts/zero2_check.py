"""CPU check (Mac, DeepSpeed 0.19.6 in the project venv): rdsp's engine factory with the D2 optimizer
settings (--grad-dtype bf16 --fused-adam --optimizer-group-numel), ZeRO-1 against ZeRO-2, 4 microbatches
per step accumulated the way rdsp drives the engine (gradient_accumulation_steps 1, only each step's last
backward marked as the boundary). Same weights expected.

    .venv/bin/python rdsp/bench_oct10/bridge_vs_rdsp/27b_tp2pp2dp2/scripts/zero2_check.py
"""

import os
import sys

os.environ.update(
    MASTER_ADDR="127.0.0.1", MASTER_PORT="29537", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"
)
import deepspeed
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4)
)  # the repo root
import train_vl
from ray_deepspeed_pipeline.deepspeed_adapter import _deepspeed_engine_factory

deepspeed.init_distributed(dist_backend="gloo")
MICRO = 4


def run(zero):
    flags = [
        "--stages",
        "1",
        "--stage",
        f"0:gpus=1,zero={zero}",
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
        "70000",
    ]
    conf = train_vl.ds_config(train_vl.build_parser().parse_args(flags))
    conf.update(train_micro_batch_size_per_gpu=1, train_batch_size=1)
    conf["zero_optimization"]["stage"] = zero
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(256, 256) for _ in range(6)]).to(torch.bfloat16)
    start = [p.detach().float().clone() for p in model.parameters()]
    engine = _deepspeed_engine_factory(model, conf)
    g = torch.Generator().manual_seed(1)
    for _ in range(10):
        for m in range(MICRO):
            engine.set_gradient_accumulation_boundary(m == MICRO - 1)
            x = torch.randn(8, 256, generator=g).to(torch.bfloat16).to(engine.device)
            engine.backward(engine(x).float().pow(2).mean() / MICRO)
        engine.step()
    end = [p.detach().float().cpu() for p in engine.module.parameters()]
    moved = max(float((a - b).abs().max()) for a, b in zip(start, end))
    return type(engine.optimizer).__name__, end, moved


name1, w1, moved = run(1)
name2, w2, _ = run(2)
diff = max(float((a - b).abs().max()) for a, b in zip(w1, w2))
print(
    f"ZeRO-1 {name1} vs ZeRO-2 {name2}, {MICRO} microbatches x 10 steps: weights moved up to "
    f"{moved:.3g} from the start; max ZeRO-1 / ZeRO-2 difference {diff:.3g} (one process: DP1)"
)
