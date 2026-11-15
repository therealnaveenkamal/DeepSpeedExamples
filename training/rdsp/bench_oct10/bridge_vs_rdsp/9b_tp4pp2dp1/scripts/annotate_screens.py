"""Draw plain-language labels on the Nsight screenshots in results/screens/.

    python scripts/annotate_screens.py      (from the layout directory)

Writes results/screens/n*_*.png. Positions are pixels in each source screenshot,
read off the timeline axis (e.g. Bridge 205 ms view: +610 ms at x=360, 5.5 px/ms).
"""

import os

from PIL import Image, ImageDraw, ImageFont

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "screens")
RED, INK, PAPER = (214, 40, 90), (25, 25, 25), (255, 255, 255)
FONT = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 30)
SMALL = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 26)


def label(d, xy, text, font=FONT, anchor="la"):
    """Text on a white box with a red border, so it reads on any background."""
    l, t, r, b = d.textbbox(xy, text, font=font, anchor=anchor)
    d.rectangle((l - 6, t - 4, r + 6, b + 4), fill=PAPER, outline=RED, width=2)
    d.text(xy, text, font=font, fill=INK, anchor=anchor)
    return l - 6, t - 4, r + 6, b + 4


def box(d, xyxy, width=3):
    d.rectangle(xyxy, outline=RED, width=width)


def bracket(d, x0, x1, y, text, up=True):
    """A horizontal span with end ticks and a label above (or below) it."""
    s = -1 if up else 1
    d.line((x0, y, x1, y), fill=RED, width=3)
    d.line((x0, y, x0, y - s * 10), fill=RED, width=3)
    d.line((x1, y, x1, y - s * 10), fill=RED, width=3)
    label(d, ((x0 + x1) / 2, y + s * 8), text, anchor="mb" if up else "mt")


def arrow(d, start, end):
    d.line((*start, *end), fill=RED, width=3)
    x, y = end
    d.polygon([(x, y), (x - 7, y - 12), (x + 7, y - 12)] if start[1] < y else
              [(x, y), (x - 7, y + 12), (x + 7, y + 12)], fill=RED)


def pad_top(img, h, bottom=0):
    """Extra white space above (and below) the screenshot for labels."""
    out = Image.new("RGB", (img.width, img.height + h + bottom), PAPER)
    out.paste(img, (0, h))
    return out


def bridge_205():
    img = pad_top(Image.open(f"{D}/7_nsight_bridge_stage0_205ms.jpg").convert("RGB"), 80, 50)
    d = ImageDraw.Draw(img)
    o = 80  # rows moved down by the padding
    bracket(d, 325, 1102, 62, "Vision encoder: 124 ms")
    bracket(d, 1105, 1468, 62, "16 language layers: 58 ms")
    box(d, (322, o + 333, 1104, o + 362))
    label(d, (10, o + 412), "1 microbatch (a step has 64)", font=SMALL)
    label(d, (700, o + 412), "Green = CPU waiting for the GPU: 4 waits per layer, 28 ms in total.", font=SMALL)
    box(d, (240, o + 52, 1540, o + 80), width=2)
    label(d, (950, o + 104), "GPU kernels (blue). White gaps = GPU idle.", font=SMALL)
    img.save(f"{D}/n1_bridge_one_microbatch.png")


def rdsp_205():
    img = pad_top(Image.open(f"{D}/5_nsight_rdsp_stage0_205ms.jpg").convert("RGB"), 80, 50)
    d = ImageDraw.Draw(img)
    o = 80
    # NVTX forward bar x 281-960 = 112.7 ms from 30.1065 s (6.03 px/ms); conv at 30.1150 s,
    # first language kernel at 30.1673 s; the one visible wait (2.7 ms) at 30.1093 s
    bracket(d, 332, 647, 62, "Vision: 52 ms")
    bracket(d, 649, 960, 62, "16 language layers: 52 ms")
    box(d, (332, o + 313, 960, o + 340))
    box(d, (296, o + 311, 318, o + 342), width=2)
    label(d, (250, o + 215), "2.7 ms wait, before the encoder", font=SMALL)
    label(d, (10, o + 412), "1 microbatch (a step has 64)", font=SMALL)
    label(d, (560, o + 412), "Encoder waits 91 times, each under 0.01 ms: too short to see.", font=SMALL)
    img.save(f"{D}/n2_rdsp_one_microbatch.png")


def tooltip():
    img = Image.open(f"{D}/8_nsight_bridge_sync_tooltip.jpg").convert("RGB").crop((200, 360, 1100, 520))
    img = pad_top(img, 56)
    d = ImageDraw.Draw(img)
    label(d, (10, 12), "Hovering one green box: the CPU sat in cudaStreamSynchronize for 2.2 ms.", font=SMALL)
    box(d, (380, 56 + 44, 410, 56 + 62))
    img.save(f"{D}/n3_bridge_one_wait.png")


def bridge_15():
    img = pad_top(Image.open(f"{D}/9_nsight_bridge_stage0_15ms.jpg").convert("RGB"), 64)
    d = ImageDraw.Draw(img)
    o = 64
    # axis: +613 ms at x=280, 82.5 px/ms; conv ends at +618.4 ms, next kernels at +621.3 ms
    bracket(d, 637, 722, 50, "Conv")
    bracket(d, 726, 965, 50, "GPU idle for 3 ms")
    box(d, (726, o + 52, 965, o + 150))
    box(d, (726, o + 462, 965, o + 492))
    label(d, (700, o + 520), "CPU sends almost no work to the GPU here: 5 CUDA calls in 3 ms.", font=SMALL)
    img.save(f"{D}/n4_bridge_15ms.png")


def rdsp_15():
    img = pad_top(Image.open(f"{D}/6_nsight_rdsp_stage0_15ms.jpg").convert("RGB"), 64)
    d = ImageDraw.Draw(img)
    o = 64
    # axis: +114 ms at x=750, 85 px/ms; vision conv +115.0 to +116.0 ms
    bracket(d, 835, 922, 50, "Conv")
    bracket(d, 925, 1535, 50, "Next 7 ms: GPU 67% busy (24% computing)")
    box(d, (925, o + 312, 1535, o + 340))
    label(d, (930, o + 372), "CPU sends CUDA calls nonstop: 406 in 7 ms.", font=SMALL)
    img.save(f"{D}/n5_rdsp_15ms.png")


if __name__ == "__main__":
    for f in (bridge_205, rdsp_205, tooltip, bridge_15, rdsp_15):
        f()
    print(sorted(p for p in os.listdir(D) if p.startswith("n")))
