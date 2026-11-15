"""Compact GPU timelines from an Nsight Systems sqlite export, for plotting.

    python timeline_extract.py run.sqlite out.json

Picks the GPU process with the most kernels (a stage-0 GPU: vision encoder + 16
layers) and the one with the fewest (a stage-1 GPU). For each: every kernel in a
1-second window 30 s into the trace (start, end, compute or NCCL), and for the
first 25 s, the fraction of each 50 ms bin spent in compute kernels, NCCL kernels,
or no kernel at all.
"""

import json
import sqlite3
import sys

ZOOM_AT, ZOOM_LEN, SPAN, BIN = 30.0, 1.0, 25.0, 0.05  # seconds


def busy_fraction(intervals, lo, hi, n_bins):
    out = [0.0] * n_bins
    for s, e in intervals:
        s, e = max(s, lo), min(e, hi)
        b = int((s - lo) / BIN)  # step bins by index: float edges never stall the loop
        while s < e and b < n_bins:
            out[b] += max(0.0, min(e, lo + (b + 1) * BIN) - max(s, lo + b * BIN))
            b += 1
            if lo + b * BIN >= e:
                break
    return [x / BIN for x in out]


def main() -> None:
    db = sqlite3.connect(sys.argv[1])
    nccl = {i for i, v in db.execute("SELECT id, value FROM StringIds") if "nccl" in v.lower()}
    t0 = db.execute("SELECT min(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    procs = db.execute("SELECT globalPid, deviceId, count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                       "GROUP BY globalPid, deviceId ORDER BY count(*) DESC").fetchall()
    out = {}
    for label, (pid, dev, _) in (("stage0", procs[0]), ("stage1", procs[-1])):
        rows = db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                          "WHERE globalPid = ? AND deviceId = ? ORDER BY start",
                          (pid, dev)).fetchall()
        sec = [((s - t0) / 1e9, (e - t0) / 1e9, n in nccl) for s, e, n in rows]
        zoom = [[round(s - ZOOM_AT, 6), round(e - ZOOM_AT, 6), "nccl" if c else "compute"]
                for s, e, c in sec if s < ZOOM_AT + ZOOM_LEN and e > ZOOM_AT]
        n_bins = int(SPAN / BIN)
        comp = busy_fraction([(s, e) for s, e, c in sec if not c], 0.0, SPAN, n_bins)
        comm = busy_fraction([(s, e) for s, e, c in sec if c], 0.0, SPAN, n_bins)
        out[label] = {"kernels_in_trace": len(rows), "zoom": zoom, "bin_s": BIN,
                      "compute": [round(x, 3) for x in comp], "nccl": [round(x, 3) for x in comm]}
    json.dump(out, open(sys.argv[2], "w"))
    print({k: (v["kernels_in_trace"], len(v["zoom"])) for k, v in out.items()})


if __name__ == "__main__":
    main()
