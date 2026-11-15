# ruff: noqa: E501  (inline HTML strings)
"""Build report.html: MegatronMIMO vs rdsp on Qwen3.5-9B, layouts a and b.

    python scripts/make_report.py      (from mimo_vs_rdsp/, after annotate_screens.py)

Numbers come from: <layout>/logs (step times), <layout>/results/pyspy_roles.txt (py-spy),
nsys_facts.txt (Nsight, scripts/nsys_facts.py), sweep/*.log (vision sweep). Same page
design as the Bridge report (its CSS is reused).
"""

import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "../bridge_vs_rdsp/9b_tp4pp2dp1/scripts"))
from make_report import CSS, esc  # noqa: E402

A, B = "9b_v1_tp2pp2dp1", "9b_v1_tp2pp1dp2"
SUMMARY_URL = "https://claude.ai/artifact/DwiJpTgif8NV8Y756vQupZ"  # the summary of all 9B runs


def median_steps(layout):
    t = open(f"{HERE}/{layout}/logs/mimo.log").read()
    m = [float(x[2]) / 1e3 for x in re.finditer(r"iteration +(\d+)/ +\d+ .*?per iteration \(ms\): ([\d.]+)", t) if int(x[1]) >= 6]
    t = open(f"{HERE}/{layout}/logs/rdsp.log").read()
    r = [float(x[2]) / 1e3 for x in re.finditer(r"^step (\d+) loss [\d.]+ (\d+) ms", t, re.M) if int(x[1]) >= 5]
    return statistics.median(m), statistics.median(r)


def roles(layout):
    """{role: {row: (mimo, rdsp)}} from results/pyspy_roles.txt."""
    out, role = {}, None
    for line in open(f"{HERE}/{layout}/results/pyspy_roles.txt"):
        m = re.match(r"== (\w+) GPU process", line)
        if m:
            role = out.setdefault(m[1], {})
            continue
        m = re.match(r"  (.+?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+[+-][\d.]+$", line)
        if m and role is not None:
            role[m[1].strip()] = (float(m[2]), float(m[3]))
    return out


def sweep():
    rows = []
    for lang in ("tp2pp2dp1", "tp2pp1dp2"):
        for line in open(f"{HERE}/sweep/best_{lang}.txt"):
            name, sec = line.split()
            rows.append((lang, name.split("_")[1], float(sec)))
    return rows


def shot(path, alt, caption):
    return (f'<figure><a class="shot" href="{path}"><img src="{path}" alt="{esc(alt)}"></a>'
            f'<figcaption>{caption}</figcaption></figure>')


def main() -> None:
    (ma, ra), (mb, rb) = median_steps(A), median_steps(B)
    va, la, lb = roles(A)["vision"], roles(A), roles(B)["language"]
    fwd = {s: sum(v[i] for k, v in va.items() if k.startswith("forward")) for i, s in enumerate(("mimo", "rdsp"))}
    first = la["first"]
    vis_names = {"1GPU": "1 GPU", "v1": "1 GPU", "vdp2": "DP2 (2 GPUs)", "vtp2": "TP2 (2 GPUs)",
                 "vtp4": "TP4 (4 GPUs)", "vtp2dp2": "TP2 × DP2 (4 GPUs)"}
    lang_names = {"tp2pp2dp1": "a: TP2 × PP2", "tp2pp1dp2": "b: TP2 × DP2"}
    srows = "".join(f'<tr><td>{lang_names[g]}</td><td>{vis_names.get(v, v)}</td><td class="num">{s:.2f} s</td></tr>'
                    for g, v, s in sweep())
    brows = "".join(f'<tr><td>{k}</td><td class="num">{lb[key][0]:.1f}</td><td class="num">{lb[key][1]:.1f}</td></tr>'
                    for k, key in (("Forward pass", "forward"), ("Backward pass", "backward"),
                                   ("Busy, total", "busy (all but waiting)"), ("Waiting (sync, send/receive)", "waiting")))
    back = f'<p class="muted"><a href="{SUMMARY_URL}">← Back to the summary of all runs</a></p>' if SUMMARY_URL else ""
    S = f"{A}/results/screens"

    page = f"""<title>9B MIMO vs rdsp</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}
.nums {{ display: flex; flex-wrap: wrap; gap: 12px 32px; }}
.nums div {{ display: grid; gap: 2px; }}
.big {{ font: 600 2rem/1 var(--font-head); font-variant-numeric: tabular-nums; }}
ol.answer {{ gap: 10px; }}
h3 {{ font-family: var(--font-head); font-weight: 600; font-size: 1.15rem; margin: 8px 0 0; }}
</style>
<main>
<header style="display:grid;gap:14px">
  {back}
  <p class="eyebrow">Qwen3.5-9B · 5 H100 GPUs: vision on 1, language on 4 · 2026-10-11</p>
  <h1>MegatronMIMO vs rdsp on 9B</h1>
  <div class="nums">
    <div><span class="big r">{ra:.2f} s</span><span class="muted">a: rdsp, per step</span></div>
    <div><span class="big b">{ma:.2f} s</span><span class="muted">a: MIMO, per step</span></div>
    <div><span class="big">{100 * (ma - ra) / ma:.0f}%</span><span class="muted">a: rdsp faster</span></div>
  </div>
  <div class="nums">
    <div><span class="big r">{rb:.2f} s</span><span class="muted">b: rdsp, per step</span></div>
    <div><span class="big b">{mb:.2f} s</span><span class="muted">b: MIMO, per step</span></div>
    <div><span class="big">{100 * abs(mb - rb) / mb:.0f}%</span><span class="muted">b: MIMO faster (about even)</span></div>
  </div>
  <div class="tablewrap"><table>
    <tr><th>Layout</th><th>Vision encoder</th><th>Language model</th><th>Microbatches per step</th></tr>
    <tr><td>a</td><td>1 GPU</td><td>TP2 × PP2: 2 stages of 2 GPUs</td><td>64, 1 row each</td></tr>
    <tr><td>b</td><td>1 GPU</td><td>TP2 × DP2: 1 stage of 4 GPUs</td><td>32, 2 rows each</td></tr>
  </table></div>
  <p class="muted">Same model, data and loss on both. 64 rows per step, every row padded to 2,048 tokens on both.</p>
</header>

<section id="answer">
  <h2>The answer</h2>
  <ol class="answer">
    <li><b>Layout a: MIMO is held back by its vision GPU.</b> MIMO's vision forward pass takes {fwd["mimo"]:.1f} s per step ({1000 * fwd["mimo"] / 64:.0f} ms for each of the 64 microbatches), rdsp's {fwd["rdsp"]:.1f} s ({1000 * fwd["rdsp"] / 64:.0f} ms each). MIMO uses Megatron-Bridge's vision encoder, the same one that made Bridge slow.</li>
    <li><b>Layout a: MIMO's hand-off between vision and language blocks the CPU on every microbatch.</b> Its first language GPUs spend about 14% of each step in that hand-off code.</li>
    <li><b>Layout b: both are held back by their 4 language GPUs, which run at about the same speed.</b> MIMO's forward pass is {round(lb["forward"][0], 1) - round(lb["forward"][1], 1):.1f} s slower, rdsp's backward pass {round(lb["backward"][1], 1) - round(lb["backward"][0], 1):.1f} s slower.</li>
    <li><b>More vision GPUs make MIMO slower.</b> Its vision GPU computes only about 3.4 s of each step. The limit is the CPU launching about 158,000 kernels per step. Splitting the encoder most likely does not reduce that (the multi-GPU runs were not profiled).</li>
  </ol>
</section>

<section id="nsight">
  <p class="eyebrow">Evidence 1 · Nsight Systems, layout a</p>
  <h2>One microbatch cycle on each system</h2>
  <p>These pictures show <b>one microbatch</b>. A training step has 64, so a step takes about 64 times as long. They come from the traced run, which is slower than the normal run (below, the per-microbatch column compares them).</p>
  <p>Time runs left to right. Each picture shows two GPUs: the vision GPU and the first language GPU. The rows:</p>
  <div class="tablewrap"><table>
    <tr><th>Row</th><th>What it shows</th><th>Look for</th></tr>
    <tr><td><code>CUDA HW</code></td><td>The GPU</td><td>Blue = running. White = idle.</td></tr>
    <tr><td><code>NVTX</code></td><td>Named code ranges</td><td>rdsp: forward and backward bars. MIMO: only small Transformer Engine ranges</td></tr>
    <tr><td><code>CUDA API</code></td><td>The CPU sending work to the GPU</td><td>Long red or green bar = the CPU stuck waiting</td></tr>
  </table></div>
  {shot(f"{S}/n1_mimo_one_cycle.png", "Nsight, MIMO, one microbatch cycle, labeled",
        '<span class="b">MIMO</span>, 186 ms shown. One microbatch: the vision GPU runs its forward pass (94 ms), then its backward pass (80 ms). The first language GPU waits 48 ms (long red bar: a copy back to the CPU) until the vision GPU sends its output.')}
  {shot(f"{S}/n2_rdsp_one_cycle.png", "Nsight, rdsp, one microbatch cycle, labeled",
        '<span class="r">rdsp</span>, 184 ms shown. One microbatch: the vision forward pass takes 40 ms. In its backward pass the vision GPU then waits 65 ms (long green bar) for the gradient, until the first language GPU finishes its own backward pass (112 ms) and sends it.')}
  <p><b>What this shows:</b> in MIMO the language GPUs wait for the vision GPU. In rdsp the vision GPU waits for the language GPUs. The GPU everyone waits for sets the step time.</p>
  <p class="muted">Nsight slows the CPU, rdsp more than MIMO (traced steps 11.9 s and 12.8 s; untraced 9.5 s and 9.1 s). Use these pictures for the pattern, not for exact times. All numbers on the labels: <code>nsys_facts.txt</code>.</p>
</section>

<section id="pyspy">
  <p class="eyebrow">Evidence 2 · py-spy, layout a, untraced</p>
  <h2>The vision GPU's CPU thread, per step</h2>
  <div class="tablewrap"><table>
    <tr><th rowspan="2">Vision GPU</th><th colspan="2" class="num">Per step (64 microbatches), s</th><th colspan="2" class="num">Per microbatch, ms</th><th colspan="2" class="num">In the picture (traced), ms</th></tr>
    <tr><th class="num">MIMO</th><th class="num">rdsp</th><th class="num">MIMO</th><th class="num">rdsp</th><th class="num">MIMO</th><th class="num">rdsp</th></tr>
    <tr><td>Forward pass</td><td class="num">{fwd["mimo"]:.1f}</td><td class="num">{fwd["rdsp"]:.1f}</td><td class="num">{1000 * fwd["mimo"] / 64:.0f}</td><td class="num">{1000 * fwd["rdsp"] / 64:.0f}</td><td class="num">94</td><td class="num">40</td></tr>
    <tr><td>Backward pass</td><td class="num">{va["backward"][0]:.1f}</td><td class="num">{va["backward"][1]:.1f}*</td><td class="num">{1000 * va["backward"][0] / 64:.0f}</td><td class="num">{1000 * va["backward"][1] / 64:.0f}*</td><td class="num">80</td><td class="num">138*</td></tr>
    <tr><td>Waiting in send/receive code</td><td class="num">{va["waiting"][0]:.1f}</td><td class="num">{va["waiting"][1]:.2f}</td><td class="num">{1000 * va["waiting"][0] / 64:.0f}</td><td class="num">{1000 * va["waiting"][1] / 64:.0f}</td><td class="num">–</td><td class="num">–</td></tr>
  </table></div>
  <p class="muted">The traced numbers are higher because Nsight slows the CPU (MIMO's whole step: 11.9 s traced vs 9.5 s normal). Per microbatch = per step ÷ 64.</p>
  <p class="muted">* Includes rdsp's waits for the gradient. They happen inside the backward pass, where py-spy cannot see them (Nsight above shows them).</p>
  <p>MIMO's first language GPUs spend {100 * first["waiting"][0] / ma:.0f}% of each step waiting (py-spy), more than half of it in the hand-off code below.</p>
  <h3>The hand-off code in MIMO</h3>
  <pre>megatron/core/pipeline_parallel/bridge_communicator.py
973  shape = <span class="hl">forward_shape_tensor.tolist()</span>   # receiver: CPU waits until the sender's shape arrives
977  shape = <span class="hl">grad_shape_tensor.tolist()</span>      # sender: CPU waits until the gradient's shape arrives
690  logging.debug(f"... sum {{<span class="hl">aggregated_gradient.sum()</span>}}")   # runs even with debug logging off</pre>
  <p>Before each send between the vision and language GPUs, MIMO first exchanges the tensor's shape and reads it back to the CPU. That read waits for the other GPU. py-spy finds MIMO's first language GPU in this code for 14% of each step (12% on line 973), and the vision GPU on line 977 for 9%. The debug line on 690 also makes the CPU wait for the GPU (0.9%).</p>
</section>

<section id="layout-b">
  <p class="eyebrow">Layout b · vision 1 GPU + language TP2 × DP2</p>
  <h2>Both are limited by the language GPUs</h2>
  <p>On both systems the vision GPU waits for the language GPUs (Nsight, per traced step):</p>
  <div class="tablewrap"><table>
    <tr><th>Vision GPU, per traced step</th><th class="num">MIMO</th><th class="num">rdsp</th></tr>
    <tr><td>CPU stuck waiting on the language GPUs</td><td class="num">3.0 s</td><td class="num">4.4 s</td></tr>
    <tr><td>Where</td><td class="num">hand-off copy</td><td class="num">inside backward</td></tr>
    <tr><td>GPU computing</td><td class="num">3.5 s</td><td class="num">3.0 s</td></tr>
  </table></div>
  <p>So the 4 language GPUs set the pace. py-spy, per step, mean of the 4:</p>
  <div class="tablewrap"><table>
    <tr><th>Language GPU, seconds per step</th><th class="num">MIMO</th><th class="num">rdsp</th></tr>
    {brows}
    <tr class="total"><td>Step</td><td class="num">{mb:.2f}</td><td class="num">{rb:.2f}</td></tr>
  </table></div>
  <p>MIMO's forward pass is slower, rdsp's backward pass is slower, and rdsp waits less. They cancel out. These numbers are sound for py-spy: the language GPUs are the last pipeline stage, so their backward pass has no later stage to wait for.</p>
</section>

<section id="sweep">
  <p class="eyebrow">Vision layouts</p>
  <h2>More vision GPUs made MIMO slower</h2>
  <div class="tablewrap"><table>
    <tr><th>Language layout</th><th>Vision layout</th><th class="num">MIMO step (steps 6–15)</th></tr>
    {srows}
  </table></div>
  <p>Measured on the 1-GPU runs (Nsight, layout a): MIMO's vision GPU computes 3.4 s of an 11.9 s traced step and launches about 158,000 kernels per step. The CPU launching those kernels is the limit, not the GPU.</p>
  <p class="muted">Not measured: the multi-GPU vision runs were not profiled. The likely reason they are slower: each vision GPU still runs every layer's Python and launches about as many kernels, and splitting adds communication.</p>
</section>

<section id="checks">
  <p class="eyebrow">How sure are we</p>
  <h2>Checks and limits</h2>
  <ul>
    <li>One run per system. No repeats. Losses match (step 1: 0.841 on both).</li>
    <li>py-spy cannot see rdsp's waits for the gradient: they happen inside the backward pass. Nsight can. An earlier version of this analysis missed that and was corrected.</li>
    <li>Nsight slows the CPU, rdsp more than MIMO. Traced steps: a 11.9 / 12.8 s, b 9.6 / 12.0 s (MIMO / rdsp).</li>
    <li>In layout b, Nsight lost kernel records on 2 of the 4 language GPUs, on both systems. Those GPUs' Nsight numbers are not used.</li>
  </ul>
  <p class="muted">Files: <code>rdsp/bench_oct10/mimo_vs_rdsp/</code>. Per layout: <code>logs/</code>, <code>profiles/</code>, <code>results/</code> (conclusion, py-spy and Nsight tables, screenshots). Scripts in <code>scripts/</code>; this page: <code>make_report.py</code>.</p>
</section>
</main>"""
    open(f"{HERE}/report.html", "w").write(page)
    print("report.html", f"{len(page) / 1e3:.0f} KB", {"a": (ma, ra), "b": (mb, rb)})


if __name__ == "__main__":
    main()
