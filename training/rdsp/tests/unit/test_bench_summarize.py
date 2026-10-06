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


def test_rdsp_only_layout_gets_a_row_without_megatron(tmp_path, capsys):
    """Colocated vision has no Megatron counterpart: its row shows rdsp alone."""
    (tmp_path / "coloc-rdsp.log").write_text(_rdsp_log(5000))
    summarize.main([str(tmp_path)])
    row = next(line for line in capsys.readouterr().out.splitlines() if "colocated" in line)
    assert "5.00" in row and row.count("-") >= 2


def test_published_logs_give_the_readme_numbers(capsys):
    """bench/published holds the logs behind the README table; summarize.py
    on them prints every step time the README reports."""
    published = os.path.join(HERE, "..", "..", "bench", "published")
    rows = []
    for size in ("2b", "4b"):
        summarize.main([os.path.join(published, size)])
        rows += capsys.readouterr().out.splitlines()[1:]
    got = [tuple(line.split()[-6:-4]) for line in rows]
    assert got == [("9.64", "8.30"), ("7.39", "5.30"), ("10.04", "8.69"), ("15.93", "13.29")]
    with open(os.path.join(HERE, "..", "..", "README.md")) as f:
        readme = f.read()
    for megatron, rdsp in got:
        assert f"| {megatron} s | {rdsp} s |" in readme


def test_published_h100_logs_give_the_benchmark_results_numbers(capsys):
    """bench/published/h100 holds the 8x H100 logs; summarize.py on them prints
    every step time the README and BENCHMARK_RESULTS.md report for H100."""
    summarize.main([os.path.join(HERE, "..", "..", "bench", "published", "h100")])
    rows = capsys.readouterr().out.splitlines()[1:]
    got = [tuple(line.split()[-6:-4]) for line in rows]
    assert got == [("7.99", "7.89"), ("8.79", "6.49"), ("18.24", "12.30"), ("-", "5.62")]
    for name in ("README.md", os.path.join("docs", "BENCHMARK_RESULTS.md")):
        with open(os.path.join(HERE, "..", "..", name)) as f:
            doc = f.read()
        for megatron, rdsp in got:
            cell = "—" if megatron == "-" else f"{megatron} s"
            assert f"| {cell} | {rdsp} s |" in doc, (name, megatron, rdsp)
