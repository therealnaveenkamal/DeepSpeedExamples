"""Seconds per step of each GPU process's training thread, by role and by what it is
doing, from py-spy raw stacks (25 Hz, 40 s while steady), for MIMO and rdsp.

    python scripts/pyspy_roles.py <layout dir> > <layout dir>/results/pyspy_roles.txt

Shares of training-step samples are turned into seconds with each system's median
step from the unprofiled 50-step run (<layout>/logs/{mimo,rdsp}.log).
Role: "vision" if the stacks reach the vision encoder; a language process is "last"
if its stacks reach the loss, else "first"; a layout without a language pipeline has
only "language". What the thread is doing (first match wins):
  waiting    blocked on the GPU or on another stage: sync calls (.tolist(), .item(),
             synchronize) and pipeline send/receive, including MIMO's vision<->language
             hand-off (bridge_communicator)
  backward   inside autograd's backward
  forward    the rest of the forward pass; for the vision process split into the parts
             of the encoder (as in bridge_vs_rdsp/9b_tp4pp2dp1/scripts/vision_parts.py)
  other      optimizer, schedule bookkeeping
"""

import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../bridge_vs_rdsp/9b_tp4pp2dp1/scripts"))
from analyze_pyspy import RULES, STEP, VISION  # noqa: E402

WAIT = re.compile(RULES[0][1].pattern + "|" + RULES[1][1].pattern + r"|bridge_communicator")
BACKWARD, FORWARD = RULES[2][1], RULES[3][1]
LOSS = re.compile(r"loss_func \(|/losses\.py|cross_entropy")  # not "liger": rdsp uses Liger norms on every stage
PARTS = {  # vision forward parts, first match wins
    "mimo": (("wait for the GPU, once per layer", r"__eq__ \(transformer_engine"),
             ("rotary embedding", r"apply_rotary_pos_emb_absolute"),
             ("attention, the rest", r"_forward_attention \(core/transformer/transformer_layer"),
             ("MLP", r"forward \(core/transformer/transformer_layer\.py:755"),
             ("outside the layers", r".")),
    "rdsp": (("wait for the GPU, once per layer", r"modeling_qwen3_5\.py:1070\)"),
             ("rotary embedding", r"apply_rotary_pos_emb_vision"),
             ("attention, the rest", r"forward \(transformers/models/qwen3_5/modeling_qwen3_5\.py:1114\)"),
             ("MLP", r"forward \(transformers/models/qwen3_5/modeling_qwen3_5\.py:1120\)"),
             ("outside the layers", r".")),
}


def median_step(log, system):
    text = open(log).read()
    if system == "mimo":
        t = [float(m[2]) for m in re.finditer(r"iteration +(\d+)/ +\d+ .*?per iteration \(ms\): ([\d.]+)", text)
             if int(m[1]) >= 6]
    else:
        t = [float(m[2]) for m in re.finditer(r"^step (\d+) loss [\d.]+ (\d+) ms", text, re.M) if int(m[1]) >= 5]
    return statistics.median(t) / 1e3


def process(path, system):
    cats, total, vision, loss = {}, 0, False, False
    for line in open(path, errors="replace"):  # a few frames carry non-UTF-8 bytes
        stack, _, n = line.rstrip("\n").rpartition(" ")
        if not n.isdigit() or not STEP.search(stack):
            continue
        n = int(n)
        total += n
        vision |= bool(VISION.search(stack))
        loss |= bool(LOSS.search(stack))
        if WAIT.search(stack):
            label = "waiting"
        elif BACKWARD.search(stack):
            label = "backward"
        elif FORWARD.search(stack):
            label = "forward"
            if VISION.search(stack):
                label = "forward: " + next(p for p, rx in PARTS[system] if re.search(rx, stack))
        else:
            label = "other"
        cats[label] = cats.get(label, 0) + n
    return total, vision, loss, cats


def main() -> None:
    d = sys.argv[1].rstrip("/")
    rows, roles = {}, {}
    for system in ("mimo", "rdsp"):
        step = median_step(f"{d}/logs/{system}.log", system)
        procs = [process(p, system) for p in sorted(glob.glob(f"{d}/profiles/pyspy-{system}-*.txt"))]
        procs = [p for p in procs if p[0]]
        pipeline = any(p[2] for p in procs) and not all(p[2] for p in procs if not p[1])
        for total, vision, loss, cats in procs:
            role = "vision" if vision else ("last" if loss else "first") if pipeline else "language"
            acc = rows.setdefault(role, {}).setdefault(system, {})
            roles.setdefault(role, {}).setdefault(system, 0)
            roles[role][system] += 1
            for k, v in cats.items():  # mean over the role's processes
                acc[k] = acc.get(k, 0) + v / total * step
        for role in rows:
            if system in rows[role] and roles[role][system] and not rows[role][system].get("_done"):
                rows[role][system] = {k: v / roles[role][system] for k, v in rows[role][system].items()}
                rows[role][system]["_done"] = 1
        print(f"{system}: median step {step:.2f} s (unprofiled run)")
    order = ["forward: " + p for p, _ in PARTS["mimo"]] + ["forward", "backward", "other", "waiting"]
    for role in ("vision", "first", "last", "language"):
        if role not in rows:
            continue
        print(f"\n== {role} GPU process (mean of MIMO {roles[role].get('mimo', 0)}, "
              f"rdsp {roles[role].get('rdsp', 0)} processes), seconds per step")
        print(f"  {'':40} {'MIMO':>6} {'rdsp':>6} {'MIMO-rdsp':>10}")
        m, r = rows[role].get("mimo", {}), rows[role].get("rdsp", {})
        for k in order:
            if k in m or k in r:
                print(f"  {k:40} {m.get(k, 0):6.2f} {r.get(k, 0):6.2f} {m.get(k, 0) - r.get(k, 0):+10.2f}")
        busy = lambda x: sum(v for k, v in x.items() if k not in ("waiting", "_done"))  # noqa: E731
        print(f"  {'busy (all but waiting)':40} {busy(m):6.2f} {busy(r):6.2f} {busy(m) - busy(r):+10.2f}")


if __name__ == "__main__":
    main()
