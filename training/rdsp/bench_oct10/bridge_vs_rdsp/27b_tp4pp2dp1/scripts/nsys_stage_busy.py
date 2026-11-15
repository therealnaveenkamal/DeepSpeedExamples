"""Per-GPU time in an Nsight Systems sqlite export, by pipeline stage: how much of the traced window
each GPU spends running any kernel, and in matmuls, NCCL (communication, including waiting for a
peer), the vision encoder's patch convolution, and everything else; plus kernel counts (a proxy for
launch overhead). For TP4 x PP2: Bridge's GPUs 0-3 are stage 0 and 4-7 stage 1; every rdsp worker
sees its GPU as device 0, so its stage 0 processes are the ones that run the vision convolution.

    python3 nsys_stage_busy.py <report.sqlite> [bridge|rdsp]
"""

import collections
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
system = sys.argv[2] if len(sys.argv) > 2 else "rdsp"
q = (
    "select k.globalPid / 16777216 % 16777216, k.deviceId, k.start, k.end, s.value "
    "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by 1, 3"
)
procs = collections.defaultdict(lambda: {"iv": [], "cat": collections.Counter(), "dev": None})
lo, hi = None, None
for pid, dev, s, e, name in db.execute(q):
    p = procs[pid]
    p["iv"].append((s, e))
    p["dev"] = dev
    n = name.lower()
    cat = (
        "nccl"
        if "nccl" in n
        else "conv"
        if "conv" in n
        else "matmul"
        if any(x in n for x in ("gemm", "nvjet", "cutlass", "xmma", "sm90_", "cublas"))
        else "other"
    )
    p["cat"][cat] += e - s
    lo = s if lo is None else min(lo, s)
    hi = e if hi is None else max(hi, e)
window = (hi - lo) / 1e9


def busy(iv):
    total, cs, ce = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + ce - cs


print(f"{system}: traced window {window:.1f} s, {len(procs)} GPU processes")
print("stage  dev   kernels  any-kernel  matmul    nccl    conv   other   (seconds of the window)")
rows = []
for pid, p in procs.items():
    if system == "bridge":
        stage = 0 if p["dev"] < 4 else 1
    else:
        stage = 0 if p["cat"]["conv"] > 0 else 1
    rows.append((stage, p["dev"], pid, p))
for stage, dev, pid, p in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
    c = p["cat"]
    print(
        f"{stage:5d}  {dev:3d}  {len(p['iv']):8d}  {busy(p['iv']) / 1e9:10.1f}  {c['matmul'] / 1e9:6.1f}"
        f"  {c['nccl'] / 1e9:6.1f}  {c['conv'] / 1e9:6.2f}  {c['other'] / 1e9:6.1f}"
    )
