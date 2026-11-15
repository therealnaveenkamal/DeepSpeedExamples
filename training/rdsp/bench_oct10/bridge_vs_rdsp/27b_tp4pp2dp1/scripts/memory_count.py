# Exact per-GPU parameter counts for Qwen3.5-27B at TP4xPP2 (meta device, no weights),
# split as rdsp now splits it: TP-plan linears, GDN (all but its norm), vocab rows and
# vision blocks / tp; norms and other small tensors whole. Cut: even 32|32, and 30|34.
import re, sys, torch, transformers
from huggingface_hub import hf_hub_download

cfg = transformers.AutoConfig.from_pretrained(
    hf_hub_download("Qwen/Qwen3.5-27B", "config.json").rsplit("/", 1)[0]
)
with torch.device("meta"):
    model = transformers.Qwen3_5ForConditionalGeneration(cfg)
TP = 4
split = re.compile(
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|linear_attn\.(in_proj_\w+|out_proj|conv1d|A_log|dt_bias)"
    r"|embed_tokens|lm_head|visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc\d))\.?"
)


def per_gpu(name, n):
    return n / TP if split.search(name) else n


layers = {}
other = {"first": 0.0, "last": 0.0}
total = 0
for name, p in model.named_parameters():
    total += p.numel()
    m = re.search(r"language_model\.layers\.(\d+)\.", name)
    if m:
        layers[int(m.group(1))] = layers.get(int(m.group(1)), 0) + per_gpu(name, p.numel())
    elif "lm_head" in name or name.endswith("language_model.norm.weight"):
        other["last"] += per_gpu(name, p.numel())
    elif "mtp" in name:
        pass
    else:
        other["first"] += per_gpu(name, p.numel())
print(f"total params {total / 1e9:.2f}B, decoder layers {len(layers)}")
for cut in (32, 30):
    s0 = other["first"] + sum(v for k, v in layers.items() if k < cut)
    s1 = other["last"] + sum(v for k, v in layers.items() if k >= cut)
    for s, n in ((0, s0), (1, s1)):
        print(
            f"cut {cut}|{64 - cut} stage {s}: {n / 1e9:.2f}B params per GPU -> {n * 18 / 1e9:.1f} GB (weights+fp32 grads+Adam, DP1)"
        )
