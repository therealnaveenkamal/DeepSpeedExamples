"""The Nsight facts quoted in the MIMO report and on its screenshot labels.

    python scripts/nsys_facts.py > nsys_facts.txt      (from mimo_vs_rdsp/)

1. Layout a, one microbatch cycle per system (the windows in the screenshots): when
   each thread is busy, the long waits, and how long the vision GPU computes.
2. Layouts a and b: per traced step, how long the vision GPU's CPU threads are blocked
   waiting on the language GPUs, and how long the vision GPU computes.
Process ids: vision = the process with the patch-embedding conv kernel.
"""

import sqlite3

HERE = "."


def busy_periods(db, pid, tid, lo, hi, gap=2e6):
    rows = db.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid/16777216 % 16777216=? "
        "AND globalTid % 16777216=? AND start BETWEEN ? AND ? ORDER BY start",
        (pid, tid, lo, hi),
    )
    out, cur = [], None
    for s, e in rows:
        if cur and s - cur[1] < gap:
            cur[1] = max(cur[1], e)
        else:
            if cur:
                out.append(cur)
            cur = [s, e]
    out.append(cur)
    return [
        f"{a / 1e9:.4f}-{b / 1e9:.4f} s ({(b - a) / 1e6:.1f} ms)" for a, b in out if b - a > 1e6
    ]


def long_calls(db, pid, lo, hi, min_ms=5):
    return [
        f"{n} on thread {t}: {s / 1e9:.4f}-{e / 1e9:.4f} s ({(e - s) / 1e6:.1f} ms)"
        for s, e, n, t in db.execute(
            "SELECT r.start, r.end, s.value, r.globalTid % 16777216 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s "
            "ON s.id=r.nameId WHERE r.globalTid/16777216 % 16777216=? AND r.start BETWEEN ? AND ? AND r.end-r.start>? "
            "ORDER BY r.start",
            (pid, lo, hi, min_ms * 1e6),
        )
    ]


def compute_ms(db, pid, lo, hi):
    return db.execute(
        "SELECT sum(min(k.end,?)-max(k.start,?))/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
        "ON s.id=k.shortName WHERE k.globalPid/16777216 % 16777216=? AND k.end>? AND k.start<? "
        "AND s.value NOT LIKE '%nccl%'",
        (hi, lo, pid, lo, hi),
    ).fetchone()[0]


def vision_pid(db):
    return db.execute(
        "SELECT k.globalPid/16777216 % 16777216 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
        "ON s.id=k.shortName WHERE s.value LIKE '%implicit_convolve%' LIMIT 1"
    ).fetchone()[0]


def cycle(name, path, threads, lo, hi):
    db = sqlite3.connect(path)
    lo, hi = int(lo * 1e9), int(hi * 1e9)
    print(
        f"\n== layout a, {name}, one cycle {lo / 1e9:.4f}-{hi / 1e9:.4f} s ({(hi - lo) / 1e6:.1f} ms)"
    )
    for label, pid, tid in threads:
        print(f"  {label} (pid {pid}, thread {tid}) busy: {busy_periods(db, pid, tid, lo, hi)}")
    for label, pid, _ in threads[:1] + threads[-1:]:
        for c in long_calls(db, pid, lo, hi):
            print(f"  {label} long call: {c}")
    vp = vision_pid(db)
    print(
        f"  vision GPU computing: {compute_ms(db, vp, lo, hi):.1f} ms of {(hi - lo) / 1e6:.1f} ms"
    )


def per_step(layout, system, traced_step_s):
    db = sqlite3.connect(f"{layout}/profiles/{system}.sqlite")
    vp = vision_pid(db)
    lo, hi = db.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    steps = (hi - lo) / 1e9 / traced_step_s
    print(
        f"\n== {layout} {system}: vision pid {vp}, {steps:.2f} traced steps of {traced_step_s} s; per traced step:"
    )
    for t, n, c, ms in db.execute(
        "SELECT r.globalTid % 16777216, s.value, count(*), sum(r.end-r.start)/1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId WHERE r.globalTid/16777216 % 16777216=? AND (s.value LIKE "
        "'cudaStreamSynchronize%' OR s.value LIKE 'cudaMemcpy%') GROUP BY 1, 2 HAVING sum(r.end-r.start) > 1e8 * ?",
        (vp, steps),
    ):
        print(
            f"  thread {t} ({'main' if t == vp else 'other, e.g. autograd'}): {n} {c / steps:.0f}x, {ms / steps / 1e3:.2f} s"
        )
    print(f"  vision GPU computing: {compute_ms(db, vp, lo, hi) / steps / 1e3:.2f} s")


if __name__ == "__main__":
    a = "9b_v1_tp2pp2dp1"
    cycle(
        "MIMO",
        f"{a}/profiles/mimo.sqlite",
        [
            ("vision main", 1088, 1088),
            ("vision autograd", 1088, 10581),
            ("first language main", 1084, 1084),
        ],
        30.7694,
        30.9550,
    )
    cycle(
        "rdsp",
        f"{a}/profiles/rdsp.sqlite",
        [
            ("vision main", 470178, 470178),
            ("vision autograd", 470178, 482020),
            ("first language main", 470179, 470179),
        ],
        30.7854,
        30.9694,
    )
    # traced step lengths: median step time in the nsys run's log over the traced window
    for layout, system, step in (
        (a, "mimo", 11.9),
        (a, "rdsp", 12.8),
        ("9b_v1_tp2pp1dp2", "mimo", 9.6),
        ("9b_v1_tp2pp1dp2", "rdsp", 12.0),
    ):
        per_step(layout, system, step)
