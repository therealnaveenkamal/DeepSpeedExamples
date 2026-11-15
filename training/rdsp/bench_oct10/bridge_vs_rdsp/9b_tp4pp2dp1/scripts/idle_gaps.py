"""Why each GPU sits idle: every gap between kernels in an Nsight Systems sqlite export,
attributed to what held up the kernel that ended it.

    python idle_gaps.py run.sqlite --step-ms 22600

For each gap (no kernel of this process running on its GPU), take the next kernel and
the CUDA call that launched it (matched by correlationId):
  dependency   the launch was issued before the gap began, yet the kernel waited:
               a stream dependency (an event from NCCL or another stream)
  after sync   the launch came late, and a sync call (stream / device / event
               synchronize) of this process returned inside the gap: the CPU was
               blocked until the GPU drained, then had to start queueing again
  cpu          the launch came late with no sync in between: the CPU was busy
               elsewhere (Python, framework code, launch overhead)
Totals are ms per step per process. Kernels inside NCCL count as busy, so time an NCCL
kernel spends waiting for peers is not in these gaps (see analyze_nsys.py).
"""

import argparse
import bisect
import collections
import sqlite3

SYNC_WORDS = ("Synchronize",)
PID_MASK = ~((1 << 24) - 1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--step-ms", type=float, required=True)
    a = p.parse_args(argv)
    db = sqlite3.connect(a.sqlite)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    lo, hi = db.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    steps = (hi - lo) / 1e6 / a.step_ms
    procs = [r[0] for r in db.execute("SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    print(f"{steps:.2f} steps in the window; ms per step per process")
    print(f"{'proc':>4} {'kernels':>8} {'idle':>7} {'dependency':>10} {'after sync':>10} "
          f"{'cpu':>7}   after-sync by call (ms)")
    rows = []
    for pid in procs:
        kernels = db.execute(
            "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE globalPid = ? ORDER BY start", (pid,)).fetchall()
        calls = db.execute(
            "SELECT start, end, correlationId, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE (globalTid & ?) = ?", (PID_MASK, pid & PID_MASK)).fetchall()
        launch = {c: (s, e) for s, e, c, _, _ in calls}
        syncs = sorted((e, names.get(n, "?").split("_v")[0])
                       for s, e, _, n, _ in calls if "Synchronize" in names.get(n, ""))
        sync_ends = [e for e, _ in syncs]
        totals = collections.Counter()
        by_call = collections.Counter()
        end = None
        for s, e, corr in kernels:
            if end is not None and s > end:
                gap = s - end
                ls, le = launch.get(corr, (None, None))
                if ls is not None and le <= end:
                    totals["dependency"] += gap
                else:
                    issue = ls if ls is not None else s
                    i = bisect.bisect_left(sync_ends, end)
                    if i < len(syncs) and syncs[i][0] <= issue:
                        totals["after sync"] += gap
                        by_call[syncs[i][1]] += gap
                    else:
                        totals["cpu"] += gap
                totals["idle"] += gap
            end = e if end is None else max(end, e)
        ms = lambda ns: ns / 1e6 / steps  # noqa: E731
        rows.append((len(kernels) / steps, totals, by_call, ms))
    for i, (n, t, c, ms) in sorted(enumerate(rows), key=lambda r: -r[1][0]):
        print(f"{i:>4} {n:>8.0f} {ms(t['idle']):>7.0f} {ms(t['dependency']):>10.0f} "
              f"{ms(t['after sync']):>10.0f} {ms(t['cpu']):>7.0f}   "
              + "  ".join(f"{k}: {ms(v):.0f}" for k, v in c.most_common(3)))
    db.close()


if __name__ == "__main__":
    main()
