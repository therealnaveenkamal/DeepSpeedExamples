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

RUNS = {  # run name -> (pair, system, active GPUs)
    "shared-megatron": ("TP2 PP2 DP2", "Megatron-Bridge", 8),
    "shared-rdsp": ("TP2 PP2 DP2", "rdsp", 8),
    "noncoloc-mimo": ("non-colocated vision", "MegatronMIMO", 5),
    "noncoloc-rdsp": ("non-colocated vision", "rdsp", 5),
    "pp4-megatron": ("PP4 DP2", "Megatron-Bridge", 8),
    "pp4-rdsp": ("PP4 DP2", "rdsp", 8),
    "best-rdsp": ("best", "rdsp (colocated)", 8),
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


def _smi_peaks(logs: Path, extra_spans: dict) -> dict:
    """Peak memory.used (GB, over GPUs) per run, from gpu_mem.csv (nvidia-smi
    every 5 s) within the run's start/end lines in seq.log."""
    seq, csv = logs / "seq.log", logs / "gpu_mem.csv"
    if not seq.exists() or not csv.exists():
        return {}
    spans, start = dict(extra_spans), {}
    for line in seq.read_text().splitlines():
        m = re.match(r"=== (start|end) (\S+) (\d\d:\d\d:\d\d)", line)
        if m and m[1] == "start":
            start[m[2]] = m[3]
        elif m:
            spans[m[2]] = (start.get(m[2]), m[3])
    samples = []
    for line in csv.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[2].endswith("MiB"):
            samples.append((parts[0].split()[-1][:8], float(parts[2].split()[0]) / 1024))
    return {name: round(max((gb for t, gb in samples if a <= t <= b), default=0.0), 1)
            for name, (a, b) in spans.items() if a}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs")
    p.add_argument("--price", type=float, default=5.00, help="instance $/hour")
    p.add_argument("--first", type=int, default=5, help="steps skipped as warm-up")
    p.add_argument("--span", action="append", default=[],
                   help="run=HH:MM:SS-HH:MM:SS for a run missing from seq.log")
    args = p.parse_args()
    logs = Path(args.logs)
    parsed = {name: parse(logs / f"{name}.log") for name in RUNS if (logs / f"{name}.log").exists()}
    real = next((r for _, r, _ in parsed.values() if r), {})
    extra = dict(span.split("=") for span in args.span)
    smi_peak = _smi_peaks(logs, {k: tuple(v.split("-")) for k, v in extra.items()})
    rows = []
    for name, (steps, _, _) in parsed.items():
        pair, system, gpus = RUNS[name]
        timed = [s for s in sorted(steps) if s >= args.first and s in real]
        if not timed:
            continue
        ms = statistics.median(steps[s][0] for s in timed)
        tok_s = statistics.median(real[s] / steps[s][0] * 1e3 for s in timed)
        rows.append({
            "run": name, "pair": pair, "system": system, "gpus": gpus, "steps": len(timed),
            "median_step_s": round(ms / 1e3, 2), "real_tok_s": round(tok_s),
            "tok_s_per_gpu": round(tok_s / gpus), "usd_per_m_tokens":
            round(args.price * gpus / 8 / 3600 / tok_s * 1e6, 3),
            # one measure for both systems: nvidia-smi's memory.used
            "peak_gb": smi_peak.get(name), "loss_first": steps[min(steps)][1],
            "loss_last": steps[max(steps)][1]})
    print(f"{'run':16} {'system':18} {'gpus':>4} {'step s':>7} {'tok/s':>7} {'tok/s/gpu':>9} "
          f"{'$/M tok':>8} {'peak GB':>8} {'loss 0':>7} {'loss end':>8}")
    for r in rows:
        print(f"{r['run']:16} {r['system']:18} {r['gpus']:>4} {r['median_step_s']:>7} "
              f"{r['real_tok_s']:>7} {r['tok_s_per_gpu']:>9} {r['usd_per_m_tokens']:>8} "
              f"{r['peak_gb'] if r['peak_gb'] is not None else '-':>8} "
              f"{r['loss_first']:>7.4f} {r['loss_last']:>8.4f}")
    curves = {n: {s: v[1] for s, v in sorted(st.items())} for n, (st, _, _) in parsed.items()}
    (logs / "summary.json").write_text(json.dumps({"rows": rows, "loss": curves}, indent=1))


if __name__ == "__main__":
    main()
