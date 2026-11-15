"""GPU time per training step from an Nsight Systems sqlite export, per GPU and per
kernel category, plus the kernels that take the most time.

    python analyze_nsys.py run.sqlite --step-ms 22600 > breakdown.txt

--step-ms is the step time INSIDE the traced window (tracing slows the CPU side, so
wall-clock under Nsight is not the benchmark; GPU kernel time per step is what
compares across systems). Busy time is the union of a GPU's kernel intervals, so
communication that overlaps compute is not counted twice; category columns are each
category's own kernel time. A GPU is one (process, device) pair: Ray workers each see
their GPU as device 0. compute = union of non-NCCL kernels; nccl-only = time only NCCL
kernels run (an NCCL kernel spins while it waits for its peers, so this is mostly
waiting); idle = no kernel at all (the GPU waits on its CPU). Stage markers: vision
(convolution) kernels only run on the first stage, the loss kernels on the last.
"""

import argparse
import re
import sqlite3
from collections import defaultdict

CATEGORIES = (  # first match wins
    ("allreduce", r"nccl.*AllReduce"),
    ("reducescat", r"nccl.*ReduceScatter"),
    ("allgather", r"nccl.*AllGather"),
    ("sendrecv", r"nccl.*(SendRecv|Send|Recv)"),
    ("nccl-other", r"nccl"),
    (
        "gdn",
        r"gated_delta|delta_rule|causal_conv1d|chunk_|recompute_w_u|kkt|solve_tril"
        r"|l2norm|local_cumsum|fused_recurrent",
    ),
    ("attention", r"flash|fmha|attention|attn|sdpa"),
    ("matmul", r"gemm|xmma|nvjet|cutlass|cublas|wgmma|sm90_|sm100_|matmul"),
    ("optimizer", r"adam|multi_tensor|optimizer"),
    ("loss", r"cross_entropy|softmax|logsumexp|liger|_ce_"),
    (
        "elementwise",
        r"elementwise|reduce|norm|rms|silu|gelu|swiglu|copy|cat|fill|index"
        r"|gather|scatter|triton|where|mul|add",
    ),
)
COLUMNS = [c for c, _ in CATEGORIES] + ["other"]
VISION = re.compile(r"conv(?!.*sdpa)|im2col|implicit", re.I)
LOSS = re.compile(r"cross_entropy|logsumexp|log_softmax|liger", re.I)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "other"


def union_ns(intervals) -> int:
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total, end = total + e - s, e
        elif e > end:
            total, end = total + e - end, e
    return total


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--step-ms", type=float, required=True)
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args(argv)
    db = sqlite3.connect(a.sqlite)
    lo, hi = db.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    steps = (hi - lo) / 1e6 / a.step_ms
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    per_dev = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    markers = defaultdict(lambda: [0, 0])
    by_kernel = defaultdict(lambda: [0.0, 0])
    for pid, gpu, sid, tot, n in db.execute(
        "SELECT globalPid, deviceId, shortName, sum(end - start), count(*) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid, deviceId, shortName"
    ):
        dev = (pid, gpu)
        name = names.get(sid, "?")
        per_dev[dev][category(name)] += tot
        counts[dev] += n
        markers[dev][0] += n if VISION.search(name) else 0
        markers[dev][1] += n if LOSS.search(name) else 0
        by_kernel[name][0] += tot
        by_kernel[name][1] += n
    busy, compute = {}, {}
    nccl_ids = {i for i, v in names.items() if "nccl" in v.lower()}
    for dev in per_dev:
        rows = db.execute(
            "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE globalPid = ? AND deviceId = ?",
            dev,
        ).fetchall()
        busy[dev] = union_ns((s, e) for s, e, _ in rows)
        compute[dev] = union_ns((s, e) for s, e, n in rows if n not in nccl_ids)
    copies = defaultdict(float)
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            query = (
                f"SELECT globalPid, deviceId, sum(end - start) FROM {table} "
                "GROUP BY globalPid, deviceId"
            )
            for pid, gpu, tot in db.execute(query):
                copies[(pid, gpu)] += tot
        except sqlite3.OperationalError:
            pass
    db.close()

    ms = lambda ns: ns / 1e6 / steps  # noqa: E731
    print(
        f"window {(hi - lo) / 1e9:.1f} s = {steps:.2f} steps of {a.step_ms / 1e3:.2f} s; "
        f"all values ms per step"
    )
    head = (
        f"{'gpu':>3} {'stage':>5} {'compute':>7} {'ncclonly':>8} {'idle':>6} {'kernels':>7} "
        f"{'vis':>5} {'loss':>5} {'copies':>6} "
    ) + " ".join(f"{c:>10}" for c in COLUMNS)
    print(head)
    order = sorted(per_dev, key=lambda d: (-markers[d][0], markers[d][1], d))
    for i, dev in enumerate(order):
        vis, loss = markers[dev]
        stage = "first" if vis > loss else ("last" if loss > vis else "?")
        print(
            f"{i:>3} {stage:>5} {ms(compute[dev]):>7.0f} {ms(busy[dev] - compute[dev]):>8.0f} "
            f"{a.step_ms - ms(busy[dev]):>6.0f} {counts[dev] / steps:>7.0f} "
            f"{vis / steps:>5.0f} {loss / steps:>5.0f} {ms(copies[dev]):>6.0f} "
            + " ".join(f"{ms(per_dev[dev][c]):>10.0f}" for c in COLUMNS)
        )
    print(f"\ntop {a.top} kernels, all GPUs (ms per step summed over GPUs, launches per step):")
    for name, (tot, n) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"{ms(tot):>8.0f} ms {n / steps:>8.0f} x  {category(name):>11}  {name[:110]}")


if __name__ == "__main__":
    main()
