"""Results table from bench/mimo/runs.sh logs: one row per Megatron / rdsp
pair, median step time over steps 6-50 (the first five compile and warm up).

Both systems train on the same samples, so a step's real (unpadded) token
count comes from the rdsp log and applies to the Megatron run's same step.

    python summarize.py ~/runs --price 5.32
"""

import argparse
import re
import statistics
from pathlib import Path

PAIRS = (  # (layout, GPUs, Megatron run, rdsp run)
    ("vision 1 GPU + language TP2xDP2", 5, "noncoloc-mimo", "noncoloc-rdsp"),
    ("TP2xPP2xDP2", 8, "shared-megatron", "shared-rdsp"),
    ("TP4xPP2xDP1", 8, "tp4pp2-megatron", "tp4pp2-rdsp"),
)
_MEGATRON = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?elapsed time per iteration \(ms\): ([\d.]+)")
_RDSP = re.compile(r"^step (\d+) loss [\d.]+ (\d+) ms .* real (\d+) supervised \d+", re.M)


def parse(path: Path) -> tuple[dict, dict]:
    """({step: ms}, {step: real tokens}); real tokens only in rdsp logs."""
    text = path.read_text(errors="replace")
    ms = {int(m[1]) - 1: float(m[2]) for m in _MEGATRON.finditer(text)}
    real = {}
    for m in _RDSP.finditer(text):
        ms[int(m[1])] = float(m[2])
        real[int(m[1])] = int(m[3])
    return ms, real


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs")
    p.add_argument("--price", type=float, default=5.32, help="8-GPU node, $/hour")
    p.add_argument("--first", type=int, default=5, help="warm-up steps left out")
    args = p.parse_args(argv)
    logs = Path(args.logs)
    print(f"{'layout':34} {'GPUs':>4} {'Megatron s':>10} {'rdsp s':>7} {'step':>6} "
          f"{'Megatron tok/s':>14} {'rdsp tok/s':>10} {'rdsp $/M tok':>12}")
    for layout, gpus, megatron, rdsp in PAIRS:
        if not (logs / f"{megatron}.log").exists() or not (logs / f"{rdsp}.log").exists():
            continue
        (m_ms, _), (r_ms, real) = parse(logs / f"{megatron}.log"), parse(logs / f"{rdsp}.log")
        timed = [s for s in sorted(r_ms) if s >= args.first and s in m_ms and s in real]
        if not timed:
            continue
        m_s = statistics.median(m_ms[s] for s in timed) / 1e3
        r_s = statistics.median(r_ms[s] for s in timed) / 1e3
        m_tok = statistics.median(real[s] / m_ms[s] * 1e3 for s in timed)
        r_tok = statistics.median(real[s] / r_ms[s] * 1e3 for s in timed)
        cost = args.price * gpus / 8 / 3600 / r_tok * 1e6
        print(f"{layout:34} {gpus:>4} {m_s:>10.2f} {r_s:>7.2f} {(r_s / m_s - 1):>+6.0%} "
              f"{m_tok:>14.0f} {r_tok:>10.0f} {cost:>12.3f}")


if __name__ == "__main__":
    main()
