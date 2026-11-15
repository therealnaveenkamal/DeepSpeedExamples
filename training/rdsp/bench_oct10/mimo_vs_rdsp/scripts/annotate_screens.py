"""Draw plain-language labels on the layout-a Nsight screenshots.

    python scripts/annotate_screens.py      (from mimo_vs_rdsp/)

Reads 9b_v1_tp2pp2dp1/results/screens/{1,2}_nsight_*.jpg, writes n1_/n2_ PNGs next to
them. Every number on a label comes from the trace (mimo.sqlite / rdsp.sqlite);
x positions come from the timeline axis of each screenshot.
"""

import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../bridge_vs_rdsp/9b_tp4pp2dp1/scripts"
    ),
)
from annotate_screens import SMALL, arrow, box, bracket, label, pad_top  # noqa: E402

D = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "9b_v1_tp2pp2dp1",
    "results",
    "screens",
)


def mimo():
    # measured on the source image's time axis: labels +770 ms ... +950 ms are 6.656 px per ms
    # apart, +770 ms at x=274 (ms after 30 s). Checks: the vision autograd thread's last call
    # (+952.3 ms) lands at x=1487, seen at 1488; the red cudaMemcpyAsync is drawn at 626-939
    # (its rounded ends trim about 3 px on each side of 623-942)
    x = lambda ms: round(274 + (ms - 770) * 6.656)  # noqa: E731
    img = pad_top(Image.open(f"{D}/1_nsight_mimo_vision_cycle.jpg").convert("RGB"), 70, 56)
    d, o = ImageDraw.Draw(img), 70
    # vision main thread busy 30.7768-30.8710 s; vision autograd thread 30.8719-30.9523 s
    bracket(d, x(776.8), x(871.0), 58, "Vision forward: 94 ms")
    bracket(d, x(871.9), x(952.3), 58, "Vision backward: 80 ms")
    # language GPU 1084: cudaMemcpyAsync 30.8225-30.8703 s; vision sends at 30.8700 s
    box(d, (x(822.5), o + 735, x(870.3), o + 759))
    arrow(d, (x(870.0), o + 350), (x(870.0), o + 730))
    label(d, (x(822.5), o + 800), "Language GPU waits 48 ms for the vision output.", font=SMALL)
    label(d, (10, o + 800), "1 microbatch (a step has 64)", font=SMALL)
    label(d, (x(874), o + 520), "Vision sends here: the wait ends.", font=SMALL)
    label(d, (250, o + 40), "VISION GPU", font=SMALL)
    label(d, (250, o + 413), "FIRST LANGUAGE GPU", font=SMALL)
    img.save(f"{D}/n1_mimo_one_cycle.png")


def rdsp():
    # measured on the source image: vision NVTX forward starts at x=278 (+786.8 ms), backward
    # ends at x=1478 (+965.4 ms): 6.72 px/ms. Checks: language backward ends at x=934 (+884.4 ms),
    # the green wait spans x 561-994 (+828.4 to +893.7 ms)
    x = lambda ms: round(278 + (ms - 786.8) * 6.72)  # noqa: E731
    img = pad_top(Image.open(f"{D}/2_nsight_rdsp_vision_cycle.jpg").convert("RGB"), 70, 56)
    d, o = ImageDraw.Draw(img), 70
    # vision NVTX: forward 30.7868-30.8267 s, backward 30.8276-30.9654 s
    bracket(d, x(786.8), x(826.7), 58, "Vision fwd: 40 ms")
    bracket(d, x(827.6), x(965.4), 58, "Vision backward: 138 ms")
    # vision autograd thread: cudaStreamSynchronize 30.82839-30.89370 s
    box(d, (x(828.4), o + 590, x(893.7), o + 616))
    label(
        d,
        (x(828.4) + 40, o + 800),
        "Vision waits 65 ms for the gradient from the language GPU.",
        font=SMALL,
    )
    label(d, (10, o + 800), "1 microbatch (a step has 64)", font=SMALL)
    # language NVTX backward ends 30.8844 s; it sends the gradient at 30.8934 s
    box(d, (240, o + 258, x(884.4), o + 283))
    label(d, (250, o + 196), "Language backward: 112 ms", font=SMALL)
    label(d, (250, o + 40), "FIRST LANGUAGE GPU", font=SMALL)
    label(d, (250, o + 489), "VISION GPU", font=SMALL)
    arrow(d, (x(893.4), o + 300), (x(893.4), o + 588))
    label(d, (x(897), o + 470), "Gradient sent here: the wait ends.", font=SMALL)
    img.save(f"{D}/n2_rdsp_one_cycle.png")


if __name__ == "__main__":
    mimo()
    rdsp()
    print(sorted(p for p in os.listdir(D) if p.startswith("n")))
