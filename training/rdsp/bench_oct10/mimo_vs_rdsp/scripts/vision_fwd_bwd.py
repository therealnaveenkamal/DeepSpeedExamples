"""Layout a, vision GPU: forward and backward time per microbatch, with rdsp's waits for
the gradient taken out of its backward, and what the forward pass launches.

    python scripts/vision_fwd_bwd.py > 9b_v1_tp2pp2dp1/results/vision_fwd_bwd.txt   (from mimo_vs_rdsp/)

Traced runs (Nsight), so absolute times are inflated by tracing (MIMO step 11.9 s vs 9.5 s
normal, rdsp 12.8 s vs 9.1 s); compare the two systems side by side, not to the normal run.
- MIMO: forward = a busy period of the vision process's main thread, backward = a busy
  period of its autograd thread (pt_autograd_4); the two alternate.
- rdsp: forward / backward = the NVTX ranges DeepSpeedEngine.forward / .backward on the vision
  process; its wait for the gradient = cudaStreamSynchronize calls over 1 ms on its autograd
  thread (pt_autograd_0) inside the backward range.
Kernels "launched by the forward" = kernels whose launch call ran on the forward thread
during a forward period (matched by correlationId).
"""

import collections
import re
import sqlite3
import statistics

L = "9b_v1_tp2pp2dp1/profiles"
CATS = (
    ("conv (patch embedding)", r"convolve"),
    ("attention (cuDNN fused)", r"cudnn_generated|fort_native_sd|flash|fmha"),
    ("attention setup (TE: seeds, seqlens, TMA)", r"extract_seed|cu_seqlens|qkv_tma"),
    ("matmul", r"gemm|nvjet|cutlass|xmma|cublas"),
    ("NCCL send (GPU time includes waiting for the receiver)", r"nccl"),
    ("layer norm", r"ln_fwd|layer_norm"),
    ("concat / copy", r"CatArray|copy|memcpy|memset"),
    ("elementwise", r"elementwise"),
    ("other small kernels", r"."),
)


def periods(db, pid, tid, lo, hi, gap=2e6):
    out, cur = [], None
    for s, e in db.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid/16777216 % 16777216=? "
        "AND globalTid % 16777216=? AND start BETWEEN ? AND ? ORDER BY start",
        (pid, tid, lo, hi),
    ):
        if cur and s - cur[1] < gap:
            cur[1] = max(cur[1], e)
        else:
            if cur:
                out.append(tuple(cur))
            cur = [s, e]
    out.append(tuple(cur))
    return [p for p in out if p[1] - p[0] > 10e6]


def launched(db, pid, tid, spans):
    """Kernel names (and GPU time) launched from thread tid during the given spans."""
    names, gpu, calls = collections.Counter(), collections.Counter(), 0
    for lo, hi in spans:
        for n, d in db.execute(
            "SELECT s.value, k.end-k.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN CUPTI_ACTIVITY_KIND_KERNEL k "
            "ON k.correlationId=r.correlationId AND k.globalPid/16777216 % 16777216=? JOIN StringIds s ON s.id=k.shortName "
            "WHERE r.globalTid/16777216 % 16777216=? AND r.globalTid % 16777216=? AND r.start BETWEEN ? AND ?",
            (pid, pid, tid, lo, hi),
        ):
            cat = next(c for c, rx in CATS if re.search(rx, n, re.I))
            names[cat] += 1
            gpu[cat] += d
        calls += db.execute(
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId "
            "WHERE r.globalTid/16777216 % 16777216=? AND r.globalTid % 16777216=? AND r.start BETWEEN ? AND ? "
            "AND s.value LIKE 'cudaLaunchKernel%'",
            (pid, tid, lo, hi),
        ).fetchone()[0]
    # one microbatch = 27 vision layers = 27 fused attention kernels (busy periods can split a pass)
    k = names["attention (cuDNN fused)"] / 27
    return {c: (names[c] / k, gpu[c] / k / 1e6) for c, _ in CATS}, calls / k


def med(xs):
    return statistics.median(xs)


def main() -> None:
    m = sqlite3.connect(f"{L}/mimo.sqlite")
    lo, hi = m.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    fwd = periods(m, 1088, 1088, lo + 5e9, hi - 5e9)
    bwd = periods(m, 1088, 10581, lo + 5e9, hi - 5e9)
    fwd = [p for p in fwd if p[1] - p[0] > 40e6]  # the forward passes (short setup bursts dropped)
    bwd = [p for p in bwd if p[1] - p[0] > 40e6]
    r = sqlite3.connect(f"{L}/rdsp.sqlite")
    nv = r.execute(
        "SELECT n.start, n.end, coalesce(n.text, s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId "
        "WHERE n.globalTid/16777216 % 16777216=470178 AND n.start BETWEEN ? AND ? "
        "AND coalesce(n.text, s.value) IN ('DeepSpeedEngine.forward','DeepSpeedEngine.backward')",
        (lo + 5e9, hi - 5e9),
    ).fetchall()
    rf = [(s, e) for s, e, n in nv if n.endswith("forward")]
    rb = [(s, e) for s, e, n in nv if n.endswith("backward")]
    waits = r.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE "
        "r.globalTid/16777216 % 16777216=470178 AND r.globalTid % 16777216=482020 AND s.value LIKE "
        "'cudaStreamSynchronize%' AND r.end-r.start > 1e6"
    ).fetchall()

    def wait_in(s, e):
        return sum(max(0, min(e, we) - max(s, ws)) for ws, we in waits)

    rb_wait = [wait_in(s, e) for s, e in rb]
    print("== Vision GPU, per microbatch, median over the traced window (ms)")
    print(
        f"   MIMO: {len(fwd)} forward passes, {len(bwd)} backward passes; rdsp: {len(rf)} and {len(rb)}"
    )
    print(f"   {'':34} {'MIMO':>7} {'rdsp':>7}")
    print(
        f"   {'forward':34} {med([e - s for s, e in fwd]) / 1e6:7.1f} {med([e - s for s, e in rf]) / 1e6:7.1f}"
    )
    print(
        f"   {'backward, as measured':34} {med([e - s for s, e in bwd]) / 1e6:7.1f} {med([e - s for s, e in rb]) / 1e6:7.1f}"
    )
    print(f"   {'  of which waiting for the gradient':34} {'0':>7} {med(rb_wait) / 1e6:7.1f}")
    print(
        f"   {'  backward work (measured - wait)':34} {med([e - s for s, e in bwd]) / 1e6:7.1f} "
        f"{med([e - s - w for (s, e), w in zip(rb, rb_wait)]) / 1e6:7.1f}"
    )

    print("\n== What one forward pass launches (per microbatch: totals / (attention kernels / 27))")
    mk, mc = launched(m, 1088, 1088, fwd)
    rk, rc = launched(r, 470178, 470178, rf)
    print(f"   cudaLaunchKernel calls from the forward thread: MIMO {mc:.0f}, rdsp {rc:.0f}")
    print(f"   {'kernel type':38} {'MIMO n':>7} {'rdsp n':>7} {'MIMO ms':>8} {'rdsp ms':>8}")
    for c, _ in CATS:
        print(f"   {c:38} {mk[c][0]:7.0f} {rk[c][0]:7.0f} {mk[c][1]:8.1f} {rk[c][1]:8.1f}")
    print(
        f"   {'total':38} {sum(v[0] for v in mk.values()):7.0f} {sum(v[0] for v in rk.values()):7.0f} "
        f"{sum(v[1] for v in mk.values()):8.1f} {sum(v[1] for v in rk.values()):8.1f}"
    )


if __name__ == "__main__":
    main()
