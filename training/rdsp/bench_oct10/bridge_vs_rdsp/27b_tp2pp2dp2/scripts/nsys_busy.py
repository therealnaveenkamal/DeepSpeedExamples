"""Per-process GPU time in an Nsight Systems sqlite export of an rdsp run: kernels, share of the window
with any kernel running, and matmul / NCCL / conv / other kernel time (NCCL time includes waiting for peers).
Every rdsp worker sees its GPU as device 0, so processes, not devices, are the unit.

    python3 nsys_busy.py <report.sqlite>
"""

import sqlite3, sys, collections

db = sqlite3.connect(sys.argv[1])
rows = collections.defaultdict(list)
for pid, s, e in db.execute(
    "select globalPid / 16777216 % 16777216, start, end from CUPTI_ACTIVITY_KIND_KERNEL order by 1, start"
):
    rows[pid].append((s, e))


def merged(iv):
    out = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            out.append((cs, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    out.append((cs, ce))
    return out


kind = collections.defaultdict(collections.Counter)
q = (
    "select k.globalPid / 16777216 % 16777216, s.value, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k "
    "join StringIds s on s.id = k.shortName"
)
for pid, name, d in db.execute(q):
    n = name.lower()
    cat = (
        "nccl"
        if "nccl" in n
        else "conv (vision)"
        if "conv" in n
        else "matmul"
        if any(x in n for x in ("gemm", "nvjet", "cutlass", "xmma"))
        else "other"
    )
    kind[pid][cat] += d
for pid in sorted(rows, key=lambda p: -len(rows[p])):
    iv = rows[pid]
    span = iv[-1][1] - iv[0][0]
    busy = sum(e - s for s, e in merged(iv))
    k = kind[pid]
    print(
        f"pid {pid}: {len(iv):7d} kernels | GPU running anything {busy / span:5.1%} | matmul {k['matmul'] / span:5.1%} "
        f"nccl {k['nccl'] / span:5.1%} conv {k['conv (vision)'] / span:4.1%} other {k['other'] / span:5.1%}"
    )
