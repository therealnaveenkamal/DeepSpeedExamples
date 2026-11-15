"""The vision encoder's forward pass on stage 0, split into parts, in seconds per step
(share of training-step samples x the system's median step time).

    python scripts/vision_parts.py > results/pyspy_vision_parts.txt   (from the layout directory)

A sample belongs to the first part whose frame is on its stack. "wait" is the line
where the encoder's CPU thread waits for the GPU once per layer (Bridge: Transformer
Engine's AttentionParams.__eq__; rdsp: lengths.tolist() in Hugging Face's attention).
"""

import re

from analyze_pyspy import RULES, STEP

PROF = "profiles"
RUNS = {  # stage-0 processes, median step (s), the encoder's root frame, its parts
    "bridge": (
        ["254456", "254458"],
        18.24,
        r"forward \(.*vision_model\.py",
        (
            ("attention: per-layer wait", r"__eq__ \(transformer_engine"),
            ("attention: rotary embedding", r"apply_rotary_pos_emb_absolute"),
            ("attention: the rest", r"_forward_attention \(core/transformer/transformer_layer"),
            ("MLP", r"forward \(core/transformer/transformer_layer\.py:755"),
            ("outside the layers", r"."),
        ),
    ),
    "rdsp": (
        ["266531", "266532", "266540", "266541"],
        14.63,
        r"get_image_features",
        (
            ("attention: per-layer wait", r"modeling_qwen3_5\.py:1070\)"),
            ("attention: rotary embedding", r"apply_rotary_pos_emb_vision"),
            (
                "attention: the rest",
                r"forward \(transformers/models/qwen3_5/modeling_qwen3_5\.py:1114\)",
            ),
            ("MLP", r"forward \(transformers/models/qwen3_5/modeling_qwen3_5\.py:1120\)"),
            ("outside the layers", r"."),
        ),
    ),
}
BACKWARD = RULES[2][1]

out = {}
for system, (pids, step_s, root, parts) in RUNS.items():
    got, total = dict.fromkeys([p for p, _ in parts], 0), 0
    for pid in pids:
        for line in open(f"{PROF}/pyspy-{system}-{pid}.txt"):
            stack, _, n = line.rstrip("\n").rpartition(" ")
            if not n.isdigit() or not STEP.search(stack):
                continue
            total += int(n)
            if not re.search(root, stack) or BACKWARD.search(stack):
                continue
            got[next(p for p, rx in parts if re.search(rx, stack))] += int(n)
    out[system] = {p: v / total * step_s for p, v in got.items()}
names = list(out["bridge"])
print(f"{'vision forward, s per step':32} {'Bridge':>7} {'rdsp':>7} {'gap':>7}")
for p in names + ["total"]:
    b = sum(out["bridge"].values()) if p == "total" else out["bridge"][p]
    r = sum(out["rdsp"].values()) if p == "total" else out["rdsp"][p]
    print(f"{p:32} {b:7.2f} {r:7.2f} {b - r:+7.2f}")
