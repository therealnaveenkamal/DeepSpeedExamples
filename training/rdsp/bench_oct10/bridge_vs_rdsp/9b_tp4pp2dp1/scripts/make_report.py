# ruff: noqa: E501  (inline HTML, CSS and SVG strings)
"""Build results/report.html: the 9B TP4xPP2xDP1 evidence. The step-time and stage-split
charts are inline SVG drawn from the measured data (logs, py-spy samples); the flame
graphs are py-spy's own (inferno) rendering (scripts/flamegraphs.sh) and the timelines
are Nsight Systems screenshots, all in results/screens/ (linked, not inlined).

    python scripts/make_report.py      (run from the layout directory, after flamegraphs.sh)
"""

import html
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pyspy import PARTS, RULES, STEP  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS, PROF, RES = (os.path.join(HERE, d) for d in ("logs", "profiles", "results"))
STAGE0 = {"bridge": ["254456", "254458"], "rdsp": ["266531", "266532", "266540", "266541"]}
NAMES = {"bridge": "Megatron-Bridge", "rdsp": "rdsp"}


def esc(s):
    return html.escape(str(s), quote=True)


# ---------- data ----------------------------------------------------------------

def step_times():
    b = {int(m[1]): float(m[2]) / 1e3 for m in re.finditer(
        r"iteration +(\d+)/ +50 .*?per iteration \(ms\): ([\d.]+)", open(f"{LOGS}/bridge.log").read())}
    r = {int(m[1]) + 1: float(m[2]) / 1e3 for m in re.finditer(
        r"^step (\d+) loss [\d.]+ (\d+) ms", open(f"{LOGS}/rdsp-equal.log").read(), re.M)}
    return b, r


def stage0_split(system, step_s):
    """Seconds per step of the stage-0 training thread, by part (py-spy shares)."""
    cats, total = {}, 0
    for pid in STAGE0[system]:
        for line in open(f"{PROF}/pyspy-{system}-{pid}.txt"):
            stack, _, n = line.rstrip("\n").rpartition(" ")
            if not n.isdigit() or not STEP.search(stack):
                continue
            n = int(n)
            total += n
            label = next((c for c, rx in RULES if rx.search(stack)), "other")
            if label == "forward":
                label = "fwd:" + next((c for c, rx in PARTS if rx.search(stack)), "rest")
            cats[label] = cats.get(label, 0) + n
    share = {k: v / total for k, v in cats.items()}
    parts = {
        "Vision encoder, forward": share.get("fwd:vision", 0),
        "Language layers, forward": share.get("fwd:language", 0),
        "Loading the batch": share.get("fwd:data", 0),
        "Waiting after send/receive": share.get("sync", 0) + share.get("p2p", 0),
        "Backward pass": share.get("backward", 0),
        "Other": sum(share.get(k, 0) for k in ("fwd:rest", "fwd:checks", "other")),
    }
    return {k: v * step_s for k, v in parts.items()}, total


def vision_parts():
    """[(part, bridge s, rdsp s)] from results/pyspy_vision_parts.txt (vision_parts.py)."""
    rows = []
    for line in open(f"{RES}/pyspy_vision_parts.txt").read().splitlines()[1:]:
        m = re.match(r"(.+?)\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+$", line)
        if m[1].strip() != "total":  # the page adds its own total row
            rows.append((m[1].strip(), float(m[2]), float(m[3])))
    return rows


# ---------- SVG -----------------------------------------------------------------

def svg_steps(b, r):
    W, H, L, R, T, B = 760, 300, 52, 176, 18, 40
    steps = range(6, 51)
    lo, hi = 12, 21
    x = lambda s: L + (s - 6) / 44 * (W - L - R)  # noqa: E731
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Step time for steps 6 to 50">']
    for v in range(lo, hi + 1, 2):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="axis" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v} s</text>')
    for s in (6, 10, 20, 30, 40, 50):
        out.append(f'<text class="axis" x="{x(s):.1f}" y="{H - 16}" text-anchor="middle">{s}</text>')
    out.append(f'<text class="axis" x="{(L + W - R) / 2:.0f}" y="{H - 2}" text-anchor="middle">step</text>')
    for sysname, d in (("bridge", b), ("rdsp", r)):
        pts = " ".join(f"{x(s):.1f},{y(d[s]):.1f}" for s in steps if s in d)
        med = statistics.median(d[s] for s in steps if s in d)
        out.append(f'<polyline class="line {sysname}" points="{pts}"/>'
                   f'<line class="median {sysname}" x1="{L}" x2="{W - R}" y1="{y(med):.1f}" y2="{y(med):.1f}"/>'
                   f'<text class="label {sysname}" x="{W - R + 8}" y="{y(med) + 4:.1f}">'
                   f'{NAMES[sysname]} {med:.2f} s</text>')
    out.append("</svg>")
    return "".join(out)


def svg_split(splits, steps):
    keys = list(next(iter(splits.values())))
    W, L, R, bar, gap = 760, 128, 86, 34, 30
    H = 40 + 2 * (bar + gap) + 26 * ((len(keys) + 2) // 3)
    top = max(steps.values())
    x = lambda v: L + v / top * (W - L - R)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Stage 0 time per step by part">']
    for v in range(0, int(top) + 1, 2):
        out.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="14" y2="{14 + 2 * (bar + gap) - gap}"/>'
                   f'<text class="axis" x="{x(v):.1f}" y="{20 + 2 * (bar + gap) - gap}" text-anchor="middle">{v} s</text>')
    for i, sysname in enumerate(("bridge", "rdsp")):
        y0 = 14 + i * (bar + gap)
        out.append(f'<text class="rowlabel" x="{L - 10}" y="{y0 + bar / 2 + 5}" text-anchor="end">'
                   f'{NAMES[sysname]}</text>')
        acc = 0.0
        for j, k in enumerate(keys):
            v = splits[sysname][k]
            out.append(f'<rect class="part p{j}" x="{x(acc):.1f}" y="{y0}" width="{max(x(acc + v) - x(acc), 0):.1f}" '
                       f'height="{bar}"><title>{esc(k)}: {v:.2f} s</title></rect>')
            if x(acc + v) - x(acc) > 46:
                out.append(f'<text class="inbar" x="{(x(acc) + x(acc + v)) / 2:.1f}" y="{y0 + bar / 2 + 5}" '
                           f'text-anchor="middle">{v:.1f}</text>')
            acc += v
        out.append(f'<text class="axis" x="{x(acc) + 6:.1f}" y="{y0 + bar / 2 + 5}">= {steps[sysname]:.2f} s</text>')
    ly = 44 + 2 * (bar + gap) - gap
    for j, k in enumerate(keys):
        cx, cy = L + (j % 3) * 210, ly + (j // 3) * 26
        out.append(f'<rect class="part p{j}" x="{cx}" y="{cy - 11}" width="14" height="14"/>'
                   f'<text class="legend" x="{cx + 20}" y="{cy}">{esc(k)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------- page ----------------------------------------------------------------

CSS = """
/* Layout: one reading column, figures full width of it, evidence in reading order */
:root {
  --bg: #f6f7f8; --surface: #ffffff; --fg: #1d232b; --muted: #5b6672; --line: #d9dee3;
  --bridge: #b5651d; --rdsp: #1f6f8b; --hot: #c9302c;
  --compute: #2b8a6e; --nccl: #7b62b8; --idle: #ffffff;
  --p0: #c9302c; --p1: #3e7fb0; --p2: #d39a2c; --p3: #8a5cc2; --p4: #8c96a1; --p5: #c5ccd3;
  --font-body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --font-head: "IBM Plex Sans Condensed", "Arial Narrow", Arial, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12161b; --surface: #1a2027; --fg: #e4e8ec; --muted: #9aa5b1; --line: #2c343d;
  --bridge: #e39a55; --rdsp: #5fb3d3; --hot: #ff6b63; --compute: #4cc29d; --nccl: #a58be0; --idle: #1a2027;
  --p0: #ff6b63; --p1: #6aa9dc; --p2: #e8b54d; --p3: #b08ee6; --p4: #7d8894; --p5: #4a535d;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #12161b; --surface: #1a2027; --fg: #e4e8ec; --muted: #9aa5b1; --line: #2c343d;
  --bridge: #e39a55; --rdsp: #5fb3d3; --hot: #ff6b63; --compute: #4cc29d; --nccl: #a58be0; --idle: #1a2027;
  --p0: #ff6b63; --p1: #6aa9dc; --p2: #e8b54d; --p3: #b08ee6; --p4: #7d8894; --p5: #4a535d;
  color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font: 16px/1.55 var(--font-body); }
main { max-width: 820px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; display: grid; gap: 28px; }
h1, h2 { font-family: var(--font-head); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: 2.1rem; } h2 { font-size: 1.45rem; }
p, li { max-width: 68ch; margin: 0; } ul, ol { margin: 0; padding-left: 1.3em; display: grid; gap: 4px; }
section { display: grid; gap: 12px; min-width: 0; }
.eyebrow { font: 600 0.78rem/1 var(--font-mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.muted { color: var(--muted); }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
figure .chart { overflow-x: auto; } figure svg { display: block; width: 100%; min-width: 560px; height: auto; }
figcaption { font-size: 0.92rem; color: var(--muted); max-width: 72ch; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { font: 600 0.8rem var(--font-mono); text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
td.num, th.num { text-align: right; }
tr.total td { font-weight: 600; border-bottom: none; }
.b { color: var(--bridge); font-weight: 600; } .r { color: var(--rdsp); font-weight: 600; }
.hotword { color: var(--hot); font-weight: 600; }
pre { margin: 0; overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; font: 0.84rem/1.5 var(--font-mono); }
pre .hl { color: var(--hot); font-weight: 600; }
code { font-family: var(--font-mono); font-size: 0.9em; }
.headline { font-size: 1.15rem; }
svg text { fill: var(--fg); font-family: var(--font-body); }
svg .axis { fill: var(--muted); font-size: 12px; font-family: var(--font-mono); }
svg .rowlabel { font-size: 13px; font-weight: 600; }
svg .legend { font-size: 12.5px; }
svg .grid { stroke: var(--line); stroke-width: 1; }
svg .line { fill: none; stroke-width: 2; } svg .line.bridge { stroke: var(--bridge); } svg .line.rdsp { stroke: var(--rdsp); }
svg .median { stroke-width: 1; stroke-dasharray: 4 4; } svg .median.bridge { stroke: var(--bridge); } svg .median.rdsp { stroke: var(--rdsp); }
svg .label { font-size: 13px; font-weight: 600; } svg .label.bridge { fill: var(--bridge); } svg .label.rdsp { fill: var(--rdsp); }
svg .p0 { fill: var(--p0); } svg .p1 { fill: var(--p1); } svg .p2 { fill: var(--p2); } svg .p3 { fill: var(--p3); } svg .p4 { fill: var(--p4); } svg .p5 { fill: var(--p5); }
svg .inbar { fill: #fff; font-size: 12px; font-weight: 600; font-family: var(--font-mono); }
.shot { display: block; border: 1px solid var(--line); border-radius: 4px; overflow: hidden; }
.shot img { display: block; width: 100%; height: auto; }
.nums { display: flex; flex-wrap: wrap; gap: 12px 32px; }
.nums div { display: grid; gap: 2px; }
.big { font: 600 2rem/1 var(--font-head); font-variant-numeric: tabular-nums; }
ol.answer { gap: 10px; } ol.answer li { padding-left: 4px; }
h3 { font-family: var(--font-head); font-weight: 600; font-size: 1.15rem; margin: 8px 0 0; }
@media (max-width: 480px) { h1 { font-size: 1.7rem; } }
"""


def main() -> None:
    b, r = step_times()
    med = {"bridge": statistics.median(v for s, v in b.items() if s >= 6),
           "rdsp": statistics.median(v for s, v in r.items() if s >= 6)}
    splits = {s: stage0_split(s, med[s])[0] for s in ("bridge", "rdsp")}
    vparts = vision_parts()
    wait = {"bridge": vparts[0][1], "rdsp": vparts[0][2]}
    vtot = {"bridge": sum(r[1] for r in vparts), "rdsp": sum(r[2] for r in vparts)}
    bwd = splits["bridge"]["Backward pass"] - splits["rdsp"]["Backward pass"]

    keys = list(splits["bridge"])
    rows = "".join(
        f'<tr><td>{esc(k)}</td><td class="num">{splits["bridge"][k]:.2f}</td>'
        f'<td class="num">{splits["rdsp"][k]:.2f}</td>'
        f'<td class="num">{splits["bridge"][k] - splits["rdsp"][k]:+.2f}</td></tr>' for k in keys)
    rows += (f'<tr class="total"><td>Step</td><td class="num">{med["bridge"]:.2f}</td>'
             f'<td class="num">{med["rdsp"]:.2f}</td><td class="num">{med["bridge"] - med["rdsp"]:+.2f}</td></tr>')

    part_names = {"attention: per-layer wait": "Line that reads a GPU value (Bridge: TE check; rdsp: lengths.tolist())",
                  "attention: rotary embedding": "Rotary position embedding",
                  "attention: the rest": "Rest of attention", "MLP": "MLP",
                  "outside the layers": "Outside the layers (patch embedding, merger)"}
    vrows = "".join(f'<tr><td>{esc(part_names.get(p, p))}</td><td class="num">{b_:.2f}</td>'
                    f'<td class="num">{r_:.2f}</td><td class="num">{b_ - r_:+.2f}</td></tr>' for p, b_, r_ in vparts)
    vrows += (f'<tr class="total"><td>Total</td><td class="num">{vtot["bridge"]:.2f}</td>'
              f'<td class="num">{vtot["rdsp"]:.2f}</td><td class="num">{vtot["bridge"] - vtot["rdsp"]:+.2f}</td></tr>')
    gap, dv = med["bridge"] - med["rdsp"], vtot["bridge"] - vtot["rdsp"]
    lo = {k: min(v for st, v in d.items() if st >= 6) for k, d in (("bridge", b), ("rdsp", r))}
    hi = {k: max(v for st, v in d.items() if st >= 6) for k, d in (("bridge", b), ("rdsp", r))}

    def shot(name, alt, caption):
        return (f'<figure><a class="shot" href="screens/{name}"><img src="screens/{name}" alt="{esc(alt)}"></a>'
                f'<figcaption>{caption}</figcaption></figure>')

    page = f"""<title>9B Bridge vs rdsp</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<main>
<header style="display:grid;gap:14px">
  <p class="muted"><a href="https://claude.ai/artifact/DwiJpTgif8NV8Y756vQupZ">← Back to the summary of all runs</a></p>
  <p class="eyebrow">Qwen3.5-9B · TP4 × PP2 × DP1 · 8× H100 · 2026-10-11</p>
  <h1>Why rdsp is faster than Megatron-Bridge on 9B</h1>
  <div class="nums">
    <div><span class="big r">{med["rdsp"]:.2f} s</span><span class="muted">rdsp, per step</span></div>
    <div><span class="big b">{med["bridge"]:.2f} s</span><span class="muted">Megatron-Bridge, per step</span></div>
    <div><span class="big">{gap:.1f} s</span><span class="muted">rdsp is faster ({100 * gap / med["bridge"]:.0f}%)</span></div>
  </div>
  <p class="muted">Same model, data, batch, layer split and loss on both. Both pad every row to 2,048 tokens.</p>
</header>

<section id="answer">
  <h2>The answer</h2>
  <ol class="answer">
    <li><b>The gap is in one place: the vision encoder.</b> On stage 0 it takes {vtot["bridge"]:.1f} s per step on Bridge and {vtot["rdsp"]:.1f} s on rdsp: {1000 * vtot["bridge"] / 64:.0f} ms vs {1000 * vtot["rdsp"] / 64:.0f} ms for each of the 64 microbatches.</li>
    <li><b>Bridge's vision encoder makes the CPU wait for the GPU 4 times per layer.</b> A wait empties the GPU's queue of work. rdsp's encoder also waits, but in the Nsight trace its waits are near zero.</li>
    <li><b>Bridge's vision code also spends about twice the CPU time per layer.</b> The CPU then cannot refill the GPU fast enough, so the GPU sits idle.</li>
  </ol>
</section>

<section id="nsight">
  <p class="eyebrow">Evidence 1 · Nsight Systems</p>
  <h2>One microbatch on each system</h2>
  <p>These pictures show <b>one microbatch</b>. A training step has 64. They come from the traced run, which is slower: Bridge's vision encoder takes 124 ms here vs {1000 * vtot["bridge"] / 64:.0f} ms per microbatch in the normal run, rdsp's 52 ms vs {1000 * vtot["rdsp"] / 64:.0f} ms.</p>
  <p>Time runs left to right. Three rows matter:</p>
  <div class="tablewrap"><table>
    <tr><th>Row</th><th>What it shows</th><th>Look for</th></tr>
    <tr><td><code>CUDA HW</code></td><td>The GPU</td><td>Blue = running. White = idle.</td></tr>
    <tr><td><code>NVTX</code></td><td>Which part of the model is running</td><td>Gray bars with names</td></tr>
    <tr><td><code>CUDA API</code></td><td>The CPU sending work to the GPU</td><td>Green = CPU waiting for the GPU</td></tr>
  </table></div>
  {shot("n1_bridge_one_microbatch.png", "Nsight timeline of one Bridge microbatch with labels",
        '<span class="b">Bridge</span>, stage 0, 205 ms shown. The vision encoder takes 124 ms. In it the CPU waits for the GPU 141 times (4 per layer), 28 ms in total. Only the longer waits are wide enough to see.')}
  {shot("n2_rdsp_one_microbatch.png", "Nsight timeline of one rdsp microbatch with labels",
        '<span class="r">rdsp</span>, stage 0, the same 205 ms width. Vision encoder 52 ms, language layers 52 ms. rdsp\'s encoder waits 91 times, but each wait is under 0.01 ms (0.4 ms in total), so none show.')}
  {shot("n3_bridge_one_wait.png", "One green box hovered: cudaStreamSynchronize, 2.2 ms",
        'One of Bridge\'s green boxes, hovered. The CPU waited 2.2 ms.')}
  <h3>Zoom in: what one wait costs</h3>
  {shot("n4_bridge_15ms.png", "Bridge, 15 ms at the start of the vision encoder, GPU idle for 3 ms",
        '<span class="b">Bridge</span>, 15 ms shown, at the start of the vision encoder. After the first kernel (the conv) the GPU does nothing for 3 ms. In the 7 ms after the conv, the GPU is busy 4% of the time.')}
  {shot("n5_rdsp_15ms.png", "rdsp, 15 ms at the start of the vision encoder, GPU busy",
        '<span class="r">rdsp</span>, 15 ms shown, the same point in the model. In the 7 ms after the conv, the GPU is busy 67% of the time: 24% computing, the rest in NCCL all-reduces (the encoder is split over 4 GPUs).')}
  <p class="muted">Nsight slows the CPU, rdsp more than Bridge (Bridge 18.2 to 22.6 s per step, rdsp 14.6 to 24.6 s). Use these pictures for the pattern, not for exact times.</p>
</section>

<section id="split">
  <p class="eyebrow">Evidence 2 · py-spy</p>
  <h2>Where stage 0's time goes</h2>
  <p>py-spy reads the Python stack 25 times a second and does not slow the run. Stage 0 sets the step time on both systems; stage 1 spends about half of each step waiting for it.</p>
  <figure><div class="chart">{svg_split(splits, med)}</div>
  <figcaption>Seconds per step on stage 0. Bridge: 2 of its 4 stage-0 GPUs sampled. rdsp: all 4.</figcaption></figure>
  <div class="tablewrap"><table>
    <tr><th>Stage 0, seconds per step</th><th class="num">Bridge</th><th class="num">rdsp</th><th class="num">Bridge − rdsp</th></tr>
    {rows}
  </table></div>
  <ul>
    <li>Vision encoder: Bridge is {splits["bridge"][keys[0]] - splits["rdsp"][keys[0]]:.1f} s slower. That is the whole gap.</li>
    <li>Backward pass: rdsp is {-bwd:.1f} s slower. rdsp launches 1.6× more GPU kernels per step (Nsight).</li>
  </ul>
  <h3>Inside the vision encoder</h3>
  <div class="tablewrap"><table>
    <tr><th>Vision encoder, seconds per step</th><th class="num">Bridge</th><th class="num">rdsp</th><th class="num">Bridge − rdsp</th></tr>
    {vrows}
  </table></div>
  <p>The first row is the line in each encoder that reads a value back from the GPU. Its time includes the Python work on that line. It explains {wait["bridge"] - wait["rdsp"]:.1f} s of the {dv:.1f} s. The rest is Bridge taking about twice as long in every part.</p>
</section>

<section id="flame">
  <p class="eyebrow">Evidence 3 · py-spy flame graphs</p>
  <h2>The same thing as flame graphs</h2>
  <p>Each box is a function. Wider = more time. The bottom row is the training step; each row up is one call deeper. Purple = the search match. Click a picture for full size.</p>
  {shot("1_bridge_vision.png", "py-spy flame graph, Bridge stage 0, vision encoder in purple",
        '<span class="b">Bridge</span>: the vision encoder (purple) is 38% of this GPU\'s time.')}
  {shot("2_rdsp_vision.png", "py-spy flame graph, rdsp stage 0, vision encoder in purple",
        '<span class="r">rdsp</span>: the vision encoder (purple) is 23%.')}
  {shot("3_bridge_vision_sync.png", "Bridge vision encoder, zoomed, wait line in purple",
        '<span class="b">Bridge</span>, zoomed into the vision encoder. Purple = the line that waits for the GPU (Transformer Engine, <code>utils.py:320</code>).')}
  {shot("4_rdsp_vision_sync.png", "rdsp vision encoder, zoomed, wait line in purple",
        '<span class="r">rdsp</span>, zoomed into the vision encoder. Purple = its wait line (Hugging Face, <code>modeling_qwen3_5.py:1070</code>).')}
  <p class="muted">Interactive versions (search, click to zoom): <a href="flame_bridge.svg">Bridge</a>, <a href="flame_rdsp.svg">rdsp</a>.</p>
</section>

<section id="code">
  <p class="eyebrow">The code</p>
  <h2>Why Bridge's encoder waits</h2>
  <ol>
    <li>Bridge keeps the longest image length on the GPU:
  <pre>megatron/bridge/models/qwen_vl/modelling_qwen3_vl/vision_model.py
386  <span class="hl">max_seqlen_q = seqlens.max()</span>          # stays on the GPU</pre></li>
    <li>Before every attention call, Transformer Engine compares it in an <code>if</code>. Reading a GPU value in Python makes the CPU wait for the GPU:
  <pre>transformer_engine/pytorch/attention/dot_product_attention/utils.py
320  <span class="hl">if sf != of:</span>                         # GPU value to Python bool: CPU waits</pre></li>
    <li>The check runs once per vision layer: 27 × 64 = 1,728 times per step.</li>
  </ol>
  <p>py-spy (untraced run) finds Bridge's thread on this line for {wait["bridge"]:.1f} s per step. Nsight (traced run) counts 4 waits per vision layer in Bridge. It does not record which line each wait comes from.</p>
  <p>rdsp's encoder (Hugging Face) also reads a GPU value once per layer, at <code>lengths.tolist()</code> (<code>modeling_qwen3_5.py:1070</code>). In Nsight each of those waits is under 0.01 ms.</p>
</section>

<section id="checks">
  <p class="eyebrow">How sure are we</p>
  <h2>Checks and limits</h2>
  <figure><div class="chart">{svg_steps(b, r)}</div>
  <figcaption>Every step from 6 to 50. Bridge {lo["bridge"]:.2f} to {hi["bridge"]:.2f} s, rdsp {lo["rdsp"]:.2f} to {hi["rdsp"]:.2f} s. The two never overlap.</figcaption></figure>
  <ul>
    <li>One run per system. No repeats.</li>
    <li>py-spy read 2 of Bridge's 4 stage-0 GPUs (it failed on the other 2).</li>
    <li>py-spy sees Python only. It cannot split Python work from time stuck inside a C call.</li>
    <li>The 16 | 16 layer split is not balanced for either system. Not tested.</li>
    <li>Fixed on 2026-10-11: the first version counted forward code that runs through <code>torch.autograd.Function</code> as backward. That shrank the vision gap to 2.2 s and made up a 1.2 s backward gap. The numbers here are corrected.</li>
  </ul>
  <p class="muted">Files: <code>rdsp/bench_oct10/bridge_vs_rdsp/9b_tp4pp2dp1/</code>. Logs in <code>logs/</code>, raw profiles in <code>profiles/</code>, tables and screenshots in <code>results/</code>, scripts in <code>scripts/</code> (this page: <code>make_report.py</code>).</p>
</section>
</main>"""
    open(f"{RES}/report.html", "w").write(page)
    print("report.html", f"{len(page) / 1e6:.2f} MB",
          {k: round(v, 2) for k, v in splits["bridge"].items()}, {k: round(v, 2) for k, v in splits["rdsp"].items()})


if __name__ == "__main__":
    main()
