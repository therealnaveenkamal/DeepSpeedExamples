"""bench/mimo/summarize.py pairs each Megatron run with its rdsp run and
prints one results row per pair."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench", "mimo"))

import summarize  # noqa: E402


def _megatron_log(ms: float) -> str:
    return "".join(
        f" iteration {i + 1:8d}/      50 | consumed samples: {64 * (i + 1)} | "
        f"elapsed time per iteration (ms): {ms:.1f} | lm loss: 5.0E-01 |\n" for i in range(50))


def _rdsp_log(ms: int) -> str:
    return "".join(f"step {i} loss 0.5000 {ms} ms 1 real tok/s 2 padded tok/s "
                   f"real 65000 supervised 13000\n" for i in range(50))


def test_pairs_megatron_and_rdsp_logs(tmp_path, capsys):
    (tmp_path / "shared-megatron.log").write_text(_megatron_log(10000.0))
    (tmp_path / "shared-rdsp.log").write_text(_rdsp_log(8000))
    summarize.main([str(tmp_path)])
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if "TP2xPP2xDP2" in line)
    assert "10.00" in row and "8.00" in row and "-20%" in row
    assert "TP4xPP2xDP1" not in out  # pairs without logs are left out
