# ruff: noqa: E501  (inline HTML and SVG strings)
"""Build summary/report.html: every 9B run of 2026-10-11, rdsp vs Megatron-Bridge and
rdsp vs MegatronMIMO, with step times, per-GPU throughput and links to the analyses.
Also writes summary/results.md (the same table, for the repo).

    python scripts/make_report.py      (from summary/)

Numbers come from each run's logs: median step time over steps 6-50, real tokens per
step from rdsp's log (both systems train on the same rows, so the count is shared).
"""

import html
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bridge_vs_rdsp/9b_tp4pp2dp1/scripts"))
from make_report import CSS  # noqa: E402

BRIDGE_URL = "https://claude.ai/artifact/Paf5zLXpjbUUMJJBEXNqyv"
MIMO_URL = "https://claude.ai/artifact/E7zBPxyi8oz5n92VAmXpjM"
PADDED = 64 * 2048  # tokens per step, every row padded to 2,048 on both systems

RUNS = (  # system, layout, directory, Megatron log, rdsp log, GPUs, layer split (Megatron, rdsp), why, link
    (
        "Megatron-Bridge",
        "TP4 × PP2 × DP1",
        "bridge_vs_rdsp/9b_tp4pp2dp1",
        "bridge.log",
        "rdsp-equal.log",
        8,
        ("16 | 16", "16 | 16"),
        "Bridge's vision encoder is 3.6 s per step slower. Profiled.",
        BRIDGE_URL,
    ),
    (
        "Megatron-Bridge",
        "TP2 × PP2 × DP2",
        "bridge_vs_rdsp/9b_tp2pp2dp2",
        "bridge.log",
        "rdsp.log",
        8,
        ("16 | 16", "16 | 16"),
        "Not profiled.",
        None,
    ),
    (
        "Megatron-Bridge",
        "TP2 × DP4, no pipeline",
        "bridge_vs_rdsp/9b_tp2pp1dp4",
        "bridge.log",
        "rdsp.log",
        8,
        ("one stage", "one stage"),
        "Not profiled. No pipeline, so this compares the model code: Megatron + Transformer Engine vs Hugging Face + DeepSpeed.",
        None,
    ),
    (
        "MegatronMIMO",
        "vision 1 GPU + TP2 × PP2",
        "mimo_vs_rdsp/9b_v1_tp2pp2dp1",
        "mimo.log",
        "rdsp.log",
        5,
        ("not printed", "18 | 14"),
        "MIMO is held back by its vision GPU: slower vision encoder and a hand-off that blocks. Profiled.",
        MIMO_URL,
    ),
    (
        "MegatronMIMO",
        "vision 1 GPU + TP2 × DP2",
        "mimo_vs_rdsp/9b_v1_tp2pp1dp2",
        "mimo.log",
        "rdsp.log",
        5,
        ("one stage", "one stage"),
        "Both are held back by their language GPUs, which run at about the same speed. Profiled.",
        MIMO_URL,
    ),
)
# (group, setting, Megatron side, rdsp, flag) -- flag: "" held equal, "design" differs by design,
# "side" differs as a side effect of each system's defaults, "check" not confirmed equal.
# Sources: bridge_vs_rdsp/9b_tp4pp2dp1/results/settings.md, the runs' launch scripts,
# bench/mimo/runs.sh (STD_TRAIN, MIMO_TRAIN), MIMO's own config dump in its log.
SETTINGS = (
    (
        "Hardware and model",
        "Hardware",
        "8× H100 SXM 80 GB, NVLink, one AWS p5.48xlarge",
        "Same machine, same session",
        "",
    ),
    (
        "Hardware and model",
        "Model",
        "Qwen/Qwen3.5-9B from Hugging Face, converted to Megatron format",
        "Qwen/Qwen3.5-9B from Hugging Face, used as is",
        "",
    ),
    (
        "Hardware and model",
        "Trained parts",
        "Vision encoder, projection, language model",
        "Same",
        "",
    ),
    (
        "Hardware and model",
        "Output layer, MTP head",
        "Own output weight (not tied). MTP head: off in Bridge (<code>mtp_num_layers=null</code>); not printed by MIMO",
        "Own output weight; no MTP head",
        "check",
    ),
    ("Data and batch", "Dataset", "CORD-v2 receipts", "Same", ""),
    (
        "Data and batch",
        "Which rows, in what order",
        "Bridge: its own sampler. MIMO: its own CORD-v2 loader",
        "Replays the rows Bridge exported: same rows, order and microbatch groups as Bridge",
        "check",
    ),
    ("Data and batch", "Rows per step", "64", "64", ""),
    (
        "Data and batch",
        "Microbatches per step",
        "64 ÷ DP: 64 (DP1), 32 (DP2), 16 (DP4); 1 row per DP copy",
        "Same",
        "",
    ),
    (
        "Data and batch",
        "Sequence length and padding",
        "2,048; every row padded to 2,048 (Bridge forces it under PP and was told to for TP2 × DP4; MIMO's default)",
        "2,048; every row padded to 2,048 (<code>--pad-multiple 0</code>)",
        "",
    ),
    ("Parallel layout", "TP / PP / DP grid", "As listed for each run", "Same grid in each run", ""),
    (
        "Parallel layout",
        "Language layers per stage",
        "Bridge: even, 16 | 16. MIMO: not printed (Megatron's default is even)",
        "Picks its own (<code>--cuts balanced</code>): 16 | 16 in the Bridge runs, 18 | 14 in MIMO layout a",
        "design",
    ),
    (
        "Parallel layout",
        "Vision encoder placement",
        "Bridge: on stage 0, split with the language TP. MIMO: on its own GPU",
        "Same as the opponent in each run",
        "",
    ),
    (
        "Parallel layout",
        "Tensor-parallel details",
        "Bridge: no sequence parallel, no TP communication overlap. MIMO: not printed (Megatron's defaults are off)",
        "No sequence parallel",
        "check",
    ),
    ("Parallel layout", "Embedding and output layer", "Split by vocabulary over TP", "Same", ""),
    (
        "Parallel layout",
        "GDN (linear attention) layers",
        "Split by heads over TP, FLA kernels",
        "Same",
        "",
    ),
    (
        "Training math",
        "Loss averaging",
        "Bridge: mean over tokens. MIMO: mean per microbatch",
        "Matches each opponent: <code>--loss token-mean</code> vs Bridge, <code>--loss microbatch-mean</code> vs MIMO",
        "",
    ),
    (
        "Training math",
        "Step-1 loss",
        "Bridge 0.8412–0.8413. MIMO 0.8413 (a), 0.8412 (b)",
        "0.8412–0.8413 vs Bridge; 0.8408 vs MIMO",
        "check",
    ),
    (
        "Training math",
        "Optimizer",
        "AdamW: lr 1e-5 constant, betas 0.9 / 0.95, eps 1e-8, weight decay 0, no clipping, no warm-up",
        "Same",
        "",
    ),
    (
        "Training math",
        "Precision",
        "bf16 weights; fp32 master weights, gradients and gradient reduce",
        "bf16 weights; fp32 master weights and gradients (<code>--grad-dtype fp32</code>)",
        "",
    ),
    (
        "Training math",
        "Optimizer state",
        "Split over DP (distributed optimizer)",
        "Split over DP (ZeRO-1)",
        "",
    ),
    (
        "Training math",
        "Recompute, CUDA graphs",
        "Bridge: off, off. MIMO: not printed (Megatron's defaults are off)",
        "Off, off",
        "check",
    ),
    ("Training math", "Steps and timing", "50 steps; median step time of steps 6–50", "Same", ""),
    (
        "Differs by design: what the comparison measures",
        "Model code",
        "Megatron modules and Transformer Engine layers; fused cross entropy; fp32 gradient accumulation fused into the matrix multiply",
        "Hugging Face modules; Liger norm and SwiGLU kernels; loss split by vocabulary; fp32 accumulation as a separate add",
        "design",
    ),
    (
        "Differs by design: what the comparison measures",
        "Attention kernel",
        "Transformer Engine, <code>auto</code> backend",
        "PyTorch SDPA (flash), no padding mask",
        "design",
    ),
    (
        "Differs by design: what the comparison measures",
        "Vision encoder code",
        "Megatron / Transformer Engine layers",
        "Hugging Face blocks, eager: no <code>torch.compile</code> (it crashed compiled under TP4)",
        "design",
    ),
    (
        "Differs by design: what the comparison measures",
        "Processes",
        "One process per GPU (torchrun)",
        "One Ray actor per GPU, plus a CPU driver",
        "design",
    ),
    (
        "Differs by design: what the comparison measures",
        "Sends between stages",
        "Bridge: the CPU waits after every send and receive. MIMO: each vision-language hand-off first reads the tensor's shape back to the CPU",
        "Asynchronous; waited for at the end of the step",
        "design",
    ),
    (
        "Differs as a side effect: each system's own default",
        "Data loading",
        "Tokenizes text and processes images in its data loader during training",
        "Loads already-processed rows; the next step's rows are sent during this step (<code>--prefetch</code>)",
        "side",
    ),
    (
        "Differs as a side effect: each system's own default",
        "NaN check",
        "Scans gradients every step (on in both Bridge and MIMO)",
        "None",
        "side",
    ),
    (
        "Differs as a side effect: each system's own default",
        "Python garbage collector",
        "Bridge: automatic collection off, one collection every 100 steps (none within these 50). MIMO: not checked",
        "Automatic collection on",
        "side",
    ),
)
TAGS = {"design": "differs by design", "side": "differs: default", "check": "not confirmed equal"}

COLORS = """
:root { --meg: #b5651d; --rd: #1769aa; --diff-bg: #fbeedd; --diff-line: #c98a4a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --meg: #c97a35; --rd: #2f8fd0; --diff-bg: #33271a; --diff-line: #a8743c; } }
:root[data-theme="dark"] { --meg: #c97a35; --rd: #2f8fd0; --diff-bg: #33271a; --diff-line: #a8743c; }
svg .bar.meg { fill: var(--meg); } svg .bar.rd { fill: var(--rd); }
.key { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.92rem; }
.key span::before { content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.key .meg::before { background: var(--meg); } .key .rd::before { background: var(--rd); }
th[colspan] { text-align: center; border-bottom: none; }
.big { font: 600 2rem/1 var(--font-head); }
table.settings td { white-space: normal; vertical-align: top; }
table.settings td:first-child { width: 22%; font-weight: 600; }
table.settings tr.grp th { text-transform: none; letter-spacing: 0; font: 600 0.95rem var(--font-head); color: var(--fg); padding-top: 16px; }
table.settings tr.flag td { background: var(--diff-bg); }
.tag { display: inline-block; margin-top: 3px; font: 600 0.7rem/1.4 var(--font-mono); text-transform: uppercase; letter-spacing: 0.04em; padding: 1px 6px; border-radius: 999px; border: 1px solid var(--diff-line); color: var(--fg); background: var(--surface); }
details pre { margin-top: 8px; }
summary { cursor: pointer; font-weight: 600; }
"""


def esc(s):
    return html.escape(str(s), quote=True)


def load(run):
    system, layout, d, mlog, rlog, gpus, split, why, link = run
    mt = open(f"{ROOT}/{d}/logs/{mlog}").read()
    meg = {
        int(m[1]): (float(m[2]) / 1e3, float(m[3]))
        for m in re.finditer(
            r"iteration +(\d+)/ +\d+ .*?per iteration \(ms\): ([\d.]+).*?lm loss: ([\d.E+-]+)", mt
        )
    }
    rt = open(f"{ROOT}/{d}/logs/{rlog}").read()
    rd = {
        int(m[1]) + 1: (float(m[3]) / 1e3, float(m[2]), int(m[4]))
        for m in re.finditer(
            r"^step (\d+) loss ([\d.]+) (\d+) ms .*? real (\d+) supervised", rt, re.M
        )
    }
    ms = statistics.median(v[0] for s, v in meg.items() if s >= 6)
    rs = statistics.median(v[0] for s, v in rd.items() if s >= 6)
    real = statistics.mean(v[2] for s, v in rd.items() if s >= 6)
    diff = statistics.mean(rd[s][1] - meg[s][1] for s in range(2, 51) if s in rd and s in meg)
    return dict(
        system=system,
        layout=layout,
        dir=d,
        gpus=gpus,
        split=split,
        why=why,
        link=link,
        meg=ms,
        rdsp=rs,
        steps=(len(meg), len(rd)),
        real=real,
        loss1=(meg[1][1], rd[1][1]),
        loss_diff=diff,
        tput=(real / ms / gpus, real / rs / gpus),
        padded=(PADDED / ms / gpus, PADDED / rs / gpus),
    )


def svg_tput(rows):
    """Grouped horizontal bars: real tokens per second per GPU, Megatron side vs rdsp."""
    W, L, R, bar, gap, group = 760, 230, 70, 16, 2, 22
    top = max(max(r["tput"]) for r in rows)
    step = 200 if top < 1500 else 500
    xmax = (int(top / step) + 1) * step
    x = lambda v: L + v / xmax * (W - L - R)  # noqa: E731
    H = 26 + len(rows) * (2 * bar + gap + group)
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Real tokens per second per GPU, each run, both systems">'
    ]
    for v in range(0, xmax + 1, step):
        out.append(
            f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="0" y2="{H - 24}"/>'
            f'<text class="axis" x="{x(v):.1f}" y="{H - 8}" text-anchor="middle">{v:,}</text>'
        )
    for i, r in enumerate(rows):
        y0 = 4 + i * (2 * bar + gap + group)
        short = "Bridge" if r["system"] == "Megatron-Bridge" else "MIMO"
        out.append(
            f'<text class="rowlabel" x="{L - 10}" y="{y0 + bar}" text-anchor="end">{esc(short)}: {esc(r["layout"])}</text>'
            f'<text class="axis" x="{L - 10}" y="{y0 + 2 * bar + 1}" text-anchor="end">{r["gpus"]} GPUs</text>'
        )
        for j, (cls, name, v) in enumerate(
            (("meg", short, r["tput"][0]), ("rd", "rdsp", r["tput"][1]))
        ):
            y = y0 + j * (bar + gap)
            w = x(v) - L
            # 4 px rounded data end, square at the baseline
            out.append(
                f"<g><title>{esc(short)}: {esc(r['layout'])}. {name}: {v:,.0f} real tokens/s per GPU</title>"
                f'<path class="bar {cls}" d="M{L},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bar - 8} a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z"/>'
                f'<text class="axis" x="{x(v) + 6:.1f}" y="{y + bar - 4}">{v:,.0f}</text></g>'
            )
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    rows = [load(r) for r in RUNS]
    faster = sum(1 for r in rows if (r["meg"] - r["rdsp"]) / r["meg"] > 0.02)

    def pct(r):
        p = 100 * (r["meg"] - r["rdsp"]) / r["meg"]
        return (
            f"rdsp {p:.0f}% faster"
            if p > 2
            else ("about even" if p > -2 else f"rdsp {-p:.0f}% slower")
        )

    def name(r):
        return ("Bridge" if r["system"] == "Megatron-Bridge" else "MIMO") + " · " + r["layout"]

    trs = []
    for r in rows:
        trs.append(
            f'<tr><td>{esc(name(r))}</td><td class="num">{r["gpus"]}</td>'
            f'<td class="num">{r["meg"]:.2f}</td><td class="num">{r["rdsp"]:.2f}</td>'
            f'<td class="num">{r["tput"][0]:,.0f}</td><td class="num">{r["tput"][1]:,.0f}</td><td>{pct(r)}</td></tr>'
        )
    whys = "".join(
        f"<li><b>{esc(name(r))}:</b> {esc(r['why'])}"
        + (f' <a href="{r["link"]}">Analysis →</a>' if r["link"] else "")
        + "</li>"
        for r in rows
    )
    srows = "".join(
        f'<tr><td>{esc(name(r))}</td><td class="num">{r["padded"][0]:,.0f}</td><td class="num">{r["padded"][1]:,.0f}</td>'
        f'<td>{esc(r["split"][0])}</td><td>{esc(r["split"][1])}</td><td class="num">{r["loss_diff"]:+.4f}</td></tr>'
        for r in rows
    )
    real_mean = statistics.mean(r["real"] for r in rows)
    cfg, group = [], None
    for g, name, meg, rd, flag in SETTINGS:
        if g != group:
            cfg.append(f'<tr class="grp"><th colspan="3">{esc(g)}</th></tr>')
            group = g
        tag = f'<br><span class="tag">{TAGS[flag]}</span>' if flag else ""
        cfg.append(
            f"<tr{' class="flag"' if flag else ''}><td>{esc(name)}{tag}</td><td>{meg}</td><td>{rd}</td></tr>"
        )
    ndiff = sum(1 for x in SETTINGS if x[4])
    nflag = {k: sum(1 for x in SETTINGS if x[4] == k) for k in TAGS}

    page = f"""<title>9B Benchmark Summary</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}{COLORS}</style>
<main>
<header style="display:grid;gap:12px">
  <p class="eyebrow">Qwen3.5-9B · H100 SXM · 2026-10-11</p>
  <h1>rdsp vs Megatron on 9B: all results</h1>
  <p class="headline"><span class="big">{faster} of {len(rows)}</span> layouts: rdsp is faster. In the other one it is about even.</p>
  <p class="muted">Every run does the same work on both systems: same model, same rows, 64 rows per step, every row padded to 2,048 tokens. Step time is the median of steps 6–50.</p>
</header>

<section id="results">
  <h2>Results</h2>
  <div class="tablewrap"><table>
    <tr><th rowspan="2">Run</th><th rowspan="2" class="num">GPUs</th><th colspan="2" class="num">Step time, s</th><th colspan="2" class="num">Real tokens/s per GPU</th><th rowspan="2">Result</th></tr>
    <tr><th class="num">Megatron</th><th class="num">rdsp</th><th class="num">Megatron</th><th class="num">rdsp</th></tr>
    {"".join(trs)}
  </table></div>
  <p class="muted">Real tokens/s per GPU = real (non-padding) tokens trained per second, divided by the GPUs the run used. Both systems train on the same rows, so they process the same real tokens: on average {real_mean:,.0f} per step out of {PADDED:,} padded.</p>
  <h3 style="font-family:var(--font-head);margin:6px 0 0">Why, per run</h3>
  <ul>{whys}</ul>

</section>

<section id="throughput">
  <h2>Per-GPU throughput</h2>
  <div class="key"><span class="meg">Megatron (Bridge or MIMO)</span><span class="rd">rdsp</span></div>
  <figure><div class="chart">{svg_tput(rows)}</div>
  <figcaption>Real tokens per second per GPU. Longer is better. The MIMO runs use 5 GPUs, the Bridge runs 8.</figcaption></figure>
</section>

<section id="reports">
  <h2>The analyses</h2>
  <ul>
    <li><a href="{BRIDGE_URL}"><b>Bridge vs rdsp, TP4 × PP2 × DP1</b></a>: why rdsp is 20% faster. Nsight timelines, py-spy flame graphs, the code behind it.</li>
    <li><a href="{MIMO_URL}"><b>MIMO vs rdsp, both layouts</b></a>: why rdsp is 4% faster in one layout and even in the other, and why more vision GPUs made MIMO slower.</li>
  </ul>
</section>

<section id="same">
  <h2>Exact settings, side by side</h2>
  <p>Every setting that could change speed or the result, {len(SETTINGS)} in all. Shaded rows: {nflag["design"]} differ by design (what the comparison measures), {nflag["side"]} differ because each system keeps its own default, {nflag["check"]} are not confirmed equal (MIMO's log does not print them, or the numbers differ slightly). The other {len(SETTINGS) - ndiff} are held equal.</p>
  <div class="tablewrap"><table class="settings">
    <tr><th>Setting</th><th>Megatron (Bridge or MIMO)</th><th>rdsp</th></tr>
    {"".join(cfg)}
  </table></div>
  <details><summary>Exact commands</summary>
  <pre>Megatron-Bridge (NeMo container 26.08, Bridge r0.6.0), 8 GPUs:
torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe qwen35_vl_9b_sft_4gpu_h100_bf16_config --step_func qwen3_vl_step \
  model.seq_length=2048 dataset.seq_length=2048 dataset.source.dataset_name=cord_v2 \
  dataset.dataloader_type=single train.global_batch_size=64 train.micro_batch_size=1 \
  train.train_iters=50 model.mtp_num_layers=null model.calculate_per_token_loss=true \
  model.recompute_granularity=null optimizer.lr=1e-5 optimizer.min_lr=1e-5 \
  optimizer.adam_beta1=0.9 optimizer.adam_beta2=0.95 optimizer.adam_eps=1e-8 \
  optimizer.weight_decay=0.0 optimizer.clip_grad=0.0 scheduler.lr_warmup_iters=0 \
  scheduler.lr_decay_style=constant ddp.grad_reduce_in_fp32=true \
  ddp.use_distributed_optimizer=true ddp.average_in_collective=false \
  model.tensor_model_parallel_size=TP model.pipeline_model_parallel_size=PP
  (TP2 × DP4 also: dataset.pad_to_max_length=true)

MegatronMIMO (same container), 5 GPUs, layout a (b: --micro-batch-size 2, language pp=1,dp=2):
torchrun --nproc_per_node=5 examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py \
  --hf-model Qwen/Qwen3.5-9B --dataset-name cord_v2 --seq-length 2048 \
  --global-batch-size 64 --micro-batch-size 1 --train-iters 50 --dataloader-type single \
  --lr 1e-5 --min-lr 1e-5 --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.0 \
  --start-weight-decay 0.0 --end-weight-decay 0.0 --clip-grad 0.0 --lr-warmup-iters 0 \
  --component language=tp=2,pp=2,dp=1,rank_offset=0 --component images=tp=1,pp=1,dp=1,rank_offset=4

rdsp, e.g. TP4 × PP2 × DP1:
python train_vl.py --model Qwen/Qwen3.5-9B --dataset exported:&lt;Bridge's rows&gt; \
  --stages 2 --cuts balanced --stage 0:gpus=4,tp=4,zero=1 --stage 1:gpus=4,tp=4,zero=1 \
  --rows 1 --microbatches 64 --seq 2048 --pad-multiple 0 --steps 50 \
  --lr 1e-5 --betas 0.9,0.95 --eps 1e-8 --weight-decay 0.0 --grad-dtype fp32 \
  --loss token-mean --liger --prefetch --sharded-loss --drop-padding-mask
rdsp vs MIMO layout a: --stages 3 --cuts 0,balanced --stage 0:gpus=1,zero=1 \
  --stage 1:gpus=2,tp=2,zero=1 --stage 2:gpus=2,tp=2,zero=1 --loss microbatch-mean</pre>
  <p class="muted">Every run's script is in its folder under <code>rdsp/bench_oct10/</code> (<code>scripts/</code>).</p>
  </details>
  <h3 style="font-family:var(--font-head);margin:6px 0 0">Per-run details</h3>
  <div class="tablewrap"><table>
    <tr><th rowspan="2">Run</th><th colspan="2" class="num">Padded tokens/s per GPU</th><th colspan="2">Layer split</th><th rowspan="2" class="num">Loss diff</th></tr>
    <tr><th class="num">Megatron</th><th class="num">rdsp</th><th>Megatron</th><th>rdsp</th></tr>
    {srows}
  </table></div>
  <p class="muted">Layer split = language layers per pipeline stage. Loss diff = mean of rdsp minus Megatron over steps 2–50.</p>
</section>

<section id="limits">
  <h2>Limits</h2>
  <ul>
    <li>One run per system per layout. No repeats.</li>
    <li>3 of the 5 comparisons were profiled: Bridge TP4 × PP2 × DP1 and both MIMO layouts. The other two have step times only.</li>
    <li>The MIMO layouts use 5 of the 8 GPUs; per-GPU numbers make them comparable to the 8-GPU runs.</li>
    <li>Not run: 27B (fits only with full recompute on both systems) and the planned 4B layouts.</li>
  </ul>
  <p class="muted">Files: <code>rdsp/bench_oct10/</code> (<code>bridge_vs_rdsp/</code>, <code>mimo_vs_rdsp/</code>, one folder per run). This page: <code>summary/scripts/make_report.py</code>.</p>
</section>
</main>"""
    os.makedirs(HERE, exist_ok=True)
    open(f"{HERE}/report.html", "w").write(page)
    md = [
        "# 9B results, 2026-10-11",
        "",
        "Median step time over steps 6–50; tok/s/GPU = real tokens per second per GPU.",
        "",
        "| Run | GPUs | Megatron step | rdsp step | Result | Megatron tok/s/GPU | rdsp tok/s/GPU |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['system']}: {r['layout']} | {r['gpus']} | {r['meg']:.2f} s | {r['rdsp']:.2f} s | {pct(r)} | {r['tput'][0]:,.0f} | {r['tput'][1]:,.0f} |"
        )
    open(f"{HERE}/results.md", "w").write("\n".join(md) + "\n")
    for r in rows:
        print(
            f"{r['system'][:6]} {r['layout']:28} steps {r['steps']} meg {r['meg']:.2f} rdsp {r['rdsp']:.2f} "
            f"real/step {r['real']:,.0f} tput {r['tput'][0]:,.0f}/{r['tput'][1]:,.0f} loss1 {r['loss1']} diff {r['loss_diff']:+.4f}"
        )


if __name__ == "__main__":
    main()
