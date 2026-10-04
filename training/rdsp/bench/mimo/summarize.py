"""Summarize the matched runs from bench/mimo/runs.sh logs.

Both systems train on the same exported steps, so a step's real (unpadded)
token count comes from the rdsp log and applies to the Megatron run's same
step. Step time is the median over steps 6-50 (the first five compile and
warm up).

    python summarize.py ~/runs --price 5.00
"""

import argparse
import json
import re
import statistics
from pathlib import Path

RUNS = {  # run name -> (system, active GPUs)
    "shared-megatron": ("Megatron-Bridge", 8), "shared-rdsp": ("rdsp", 8),
    "noncoloc-mimo": ("MegatronMIMO", 6), "noncoloc-rdsp": ("rdsp", 6),
    "coloc-rdsp": ("rdsp colocated", 8),
}
_MEGATRON = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?elapsed time per iteration \(ms\): ([\d.]+)"
                       r".*?lm loss: ([\d.E+-]+)")
_RDSP = re.compile(r"^step (\d+) loss ([\d.]+) (\d+) ms .* real (\d+) supervised (\d+)")
_MEM = re.compile(r"mem-max-allocated-gigabytes: ([\d.]+)")


def parse(path: Path):
    """{step: (ms, loss)}, real tokens per step (rdsp logs only), peak GB."""
    steps, real, peak = {}, {}, None
    for line in path.read_text(errors="replace").splitlines():
        m = _MEGATRON.search(line)
        if m:
            steps[int(m[1]) - 1] = (float(m[2]), float(m[3]))
        m = _RDSP.search(line)
        if m:
            steps[int(m[1])] = (float(m[3]), float(m[2]))
            real[int(m[1])] = int(m[4])
        for g in _MEM.findall(line):
            peak = max(peak or 0.0, float(g))
    return steps, real, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs")
    p.add_argument("--price", type=float, default=5.00, help="instance $/hour")
    p.add_argument("--first", type=int, default=5, help="steps skipped as warm-up")
    args = p.parse_args()
    logs = Path(args.logs)
    parsed = {name: parse(logs / f"{name}.log") for name in RUNS if (logs / f"{name}.log").exists()}
    real = next((r for _, r, _ in parsed.values() if r), {})
    rows = []
    for name, (steps, _, peak) in parsed.items():
        system, gpus = RUNS[name]
        timed = [s for s in sorted(steps) if s >= args.first and s in real]
        if not timed:
            continue
        ms = statistics.median(steps[s][0] for s in timed)
        tok_s = statistics.median(real[s] / steps[s][0] * 1e3 for s in timed)
        rows.append({
            "run": name, "system": system, "gpus": gpus, "steps": len(timed),
            "median_step_s": round(ms / 1e3, 2), "real_tok_s": round(tok_s),
            "tok_s_per_gpu": round(tok_s / gpus), "usd_per_m_tokens":
            round(args.price * gpus / 8 / 3600 / tok_s * 1e6, 3),
            "peak_gb": peak, "loss_first": steps[min(steps)][1],
            "loss_last": steps[max(steps)][1]})
    print(f"{'run':16} {'system':16} {'gpus':>4} {'step s':>7} {'tok/s':>7} {'tok/s/gpu':>9} "
          f"{'$/M tok':>8} {'peak GB':>8} {'loss 0':>7} {'loss end':>8}")
    for r in rows:
        print(f"{r['run']:16} {r['system']:16} {r['gpus']:>4} {r['median_step_s']:>7} "
              f"{r['real_tok_s']:>7} {r['tok_s_per_gpu']:>9} {r['usd_per_m_tokens']:>8} "
              f"{r['peak_gb'] if r['peak_gb'] is not None else '-':>8} "
              f"{r['loss_first']:>7.4f} {r['loss_last']:>8.4f}")
    curves = {n: {s: v[1] for s, v in sorted(st.items())} for n, (st, _, _) in parsed.items()}
    (logs / "summary.json").write_text(json.dumps({"rows": rows, "loss": curves}, indent=1))


if __name__ == "__main__":
    main()
