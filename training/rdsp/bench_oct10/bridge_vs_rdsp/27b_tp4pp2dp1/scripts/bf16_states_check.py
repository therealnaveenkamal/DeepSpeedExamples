"""CPU check (Mac, DeepSpeed 0.19.6 in the project venv): rdsp's DeepSpeed config
under ZeRO-1: fp32 (default), bf16grads (--grad-dtype bf16),
bf16grads_fused (+ --fused-adam), bf16
(--master-dtype bf16 --grad-dtype bf16). Prints the optimizer DeepSpeed
picks, the dtypes of master weights and Adam moments, and how far 100 Adam
steps at lr 1e-5 move weights of std 0.02 (a 512x512 + 512x1 toy model).

    .venv/bin/python rdsp/bench_oct10/bridge_vs_rdsp/27b_tp4pp2dp1/scripts/bf16_states_check.py fp32|bf16grads|bf16grads_fused|bf16
"""

import os
import sys

os.environ.update(
    MASTER_ADDR="127.0.0.1", MASTER_PORT="29531", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0"
)
import deepspeed
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4)
)  # the repo root
import train_vl

mode = sys.argv[1]
flags = [
    "--stages",
    "1",
    "--stage",
    "0:gpus=1,zero=1",
    "--lr",
    "1e-5",
    "--betas",
    "0.9,0.95",
    "--rows",
    "1",
    "--microbatches",
    "1",
]
flags += {
    "fp32": [],
    "bf16grads": ["--grad-dtype", "bf16"],
    "bf16grads_fused": ["--grad-dtype", "bf16", "--fused-adam"],
    "bf16": ["--master-dtype", "bf16", "--grad-dtype", "bf16"],
}[mode]
args = train_vl.build_parser().parse_args(flags)
assert train_vl.optimizer_problem(args) is None
conf = train_vl.ds_config(args)
conf["zero_optimization"]["stage"] = 1
conf["train_micro_batch_size_per_gpu"] = 1
conf["gradient_accumulation_steps"] = 1
conf["train_batch_size"] = 1
torch.manual_seed(0)
# weights of the size a 27B linear layer has (std ~0.02)
model = torch.nn.Sequential(
    torch.nn.Linear(512, 512, bias=False), torch.nn.Linear(512, 1, bias=False)
)
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0, 0.02)
engine, opt, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config=conf
)
o = engine.optimizer
print(
    mode, "optimizer:", type(o).__name__, "| AdamW fused:", o.optimizer.param_groups[0].get("fused")
)
master = getattr(o, "single_partition_of_fp32_groups", None) or getattr(
    o, "fp32_groups_flat_partition", None
)
print("  master dtype:", master[0].dtype if master else None)
before = [p.detach().clone() for p in engine.module.parameters()]
g = torch.Generator().manual_seed(1)
target = torch.randn(512, 1, generator=g) * 0.02
for step in range(100):
    x = torch.randn(16, 512, generator=g)
    y = (x @ target).to(torch.bfloat16).to(engine.device)
    loss = (engine(x.to(torch.bfloat16).to(engine.device)).float() - y.float()).pow(2).mean()
    engine.backward(loss)
    engine.step()
    if step in (0, 9, 99):
        print(f"  step {step + 1:3d} loss {loss.item():.6f}")
st = o.optimizer.state[o.optimizer.param_groups[0]["params"][0]]
print("  adam moments:", st["exp_avg"].dtype, st["exp_avg_sq"].dtype)
changed = sum(int((a != b).sum()) for a, b in zip(before, engine.module.parameters()))
total = sum(b.numel() for b in before)
moved = (
    sum(
        float((b.float() - a.float()).norm() ** 2)
        for a, b in zip(before, engine.module.parameters())
    )
    ** 0.5
)
print(f"  distance weights moved: {moved:.3e}")
print(f"  weights changed after 100 lr=1e-5 steps: {changed}/{total} = {changed / total:.1%}")
