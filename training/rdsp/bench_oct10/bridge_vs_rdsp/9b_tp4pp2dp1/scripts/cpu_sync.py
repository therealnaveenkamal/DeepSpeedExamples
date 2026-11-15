"""CPU-side CUDA calls per training step from an Nsight Systems sqlite export: how many
kernels each process launches, and how often (and how long) its CPU blocks on the GPU.

    python cpu_sync.py run.sqlite --step-ms 22600
"""

import argparse
import collections
import sqlite3


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--step-ms", type=float, required=True)
    a = p.parse_args(argv)
    db = sqlite3.connect(a.sqlite)
    lo, hi = db.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    steps = (hi - lo) / 1e6 / a.step_ms
    per = collections.defaultdict(dict)
    for pid, fn, n, tot in db.execute(
            "SELECT r.globalTid, s.value, count(*), sum(r.end - r.start) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
            "WHERE s.value LIKE '%Synchronize%' OR s.value LIKE '%LaunchKernel%' "
            "OR s.value LIKE '%EventQuery%' GROUP BY r.globalTid, s.value"):
        per[pid][fn] = (n / steps, tot / 1e6 / steps)
    db.close()
    print(f"{steps:.2f} steps in the window; per step, per CPU thread that launches kernels")
    for pid in sorted(per):
        launches = sum(n for fn, (n, _) in per[pid].items() if "LaunchKernel" in fn)
        if launches < 1000:
            continue
        blocking = {fn.split("_v")[0]: v for fn, v in per[pid].items() if "LaunchKernel" not in fn}
        print(f"  launches {launches:>8.0f}  " + "  ".join(
            f"{fn}: {n:.0f}x {ms:.0f} ms" for fn, (n, ms) in sorted(blocking.items())))


if __name__ == "__main__":
    main()
