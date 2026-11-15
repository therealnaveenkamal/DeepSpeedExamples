"""Median step time (steps 6-50) and loss agreement for one layout's logs.

    python medians.py <layout>/logs   (bridge.log or mimo.log, and rdsp.log or rdsp-equal.log)
"""

import re
import statistics
import sys

d = sys.argv[1].rstrip("/")
out = [d.split("/")[-2]]
b = []
for name in ("bridge.log", "mimo.log"):  # the Megatron side: Bridge or MegatronMIMO
    try:
        b = [(int(m[1]), float(m[2]), float(m[3])) for m in re.finditer(
            r"iteration +(\d+)/ +\d+ .*?per iteration \(ms\): ([\d.]+).*?lm loss: ([\d.E+-]+)",
            open(f"{d}/{name}").read())]
        break
    except OSError:
        pass
r = []
for name in ("rdsp.log", "rdsp-equal.log"):
    try:
        r = [(int(m[1]) + 1, float(m[3]), float(m[2])) for m in re.finditer(
            r"^step (\d+) loss ([\d.]+) (\d+) ms", open(f"{d}/{name}").read(), re.M)]
        break
    except OSError:
        pass
for name, xs in (("megatron", b), ("rdsp", r)):
    t = [x[1] for x in xs if x[0] >= 6]
    out.append(f"{name}: {len(xs)} steps, median {statistics.median(t) / 1e3:.2f} s" if t
               else f"{name}: {len(xs)} steps, no median")
lb, lr = {x[0]: x[2] for x in b}, {x[0]: x[2] for x in r}
diff = [lr[s] - lb[s] for s in range(2, 51) if s in lb and s in lr]
if diff:
    out.append(f"loss step1 {lb.get(1, 0):.4f}/{lr.get(1, 0):.4f}, "
               f"mean diff {sum(diff) / len(diff):+.4f}")
print(" | ".join(out))
