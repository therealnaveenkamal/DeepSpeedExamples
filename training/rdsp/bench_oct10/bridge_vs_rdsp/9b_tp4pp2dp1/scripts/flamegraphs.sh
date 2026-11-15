#!/bin/bash
# flamegraphs.sh: py-spy flamegraphs of one stage-0 GPU process per system, and PNG
# screenshots of them, from the raw stacks in ../profiles. Runs on the Mac (needs
# inferno-flamegraph: `cargo install inferno`, the renderer py-spy uses, and Chrome).
#
# Each flamegraph is the training thread of one stage-0 process over the same 40 s of
# steady training (py-spy, 25 Hz, idle included), only samples inside the training step.
# Width = share of wall-clock time. Purple = frames matching the search in the file name.
#   results/flame_<system>.svg               interactive (open in a browser, click to zoom)
#   results/screens/<n>_<system>_<what>.png  the screenshots
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
P=$HERE/profiles R=$HERE/results S=$HERE/results/screens
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
mkdir -p "$S"
STEP='forward_backward_pipelining|_run_step \(.*stage_worker'

flame() {  # flame <system> <pid> <title>
  grep -E "$STEP" "$P/pyspy-$1-$2.txt" \
    | inferno-flamegraph --title "$3" --subtitle "9B TP4xPP2xDP1, 8x H100. py-spy, 40 s of the training thread of one stage-0 GPU (pid $2). Width = share of wall-clock time." \
      --colors hot --minwidth 0.05 | sed -E 's/<!DOCTYPE[^>]*>//' > "$R/flame_$1.svg"  # no DTD: artifact hosting refuses it
}

frame_at() {  # frame_at <svg> <regex>: "x y" of the widest frame whose title matches
  python3 - "$1" "$2" <<'PY'
import re, sys
svg, pat = open(sys.argv[1]).read(), re.compile(sys.argv[2])
best = None
for m in re.finditer(r'<title>([^<]*)</title><rect x="([\d.]+%)" y="(\d+)" width="([\d.]+)%"', svg):
    if pat.search(m[1]) and (best is None or float(m[4]) > best[0]):
        best = (float(m[4]), m[2], m[3])
print(best[1], best[2])
PY
}

shot() {  # shot <svg> <png> <query>: full-height screenshot, 1800 px wide
  local h; h=$(grep -o -m1 'height="[0-9]*"' "$1" | tr -dc 0-9)
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1800,"$h" \
    --virtual-time-budget=3000 --screenshot="$2" "file://$1?$3" > /dev/null 2>&1
}

enc() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

flame bridge 254456 "Megatron-Bridge, stage 0"
flame rdsp 266531 "rdsp, stage 0"

# 1-2: the whole step, the vision encoder's forward pass in purple
shot "$R/flame_bridge.svg" "$S/1_bridge_vision.png" "s=$(enc 'vision_model\.py')"
shot "$R/flame_rdsp.svg" "$S/2_rdsp_vision.png" "s=$(enc 'get_image_features')"
# 3-4: zoomed into the vision encoder, its wait-for-the-GPU line in purple
read -r x y < <(frame_at "$R/flame_bridge.svg" '^forward \(.*vision_model\.py')
shot "$R/flame_bridge.svg" "$S/3_bridge_vision_sync.png" "x=$(enc "$x")&y=$y&s=$(enc '__eq__ \(transformer_engine')"
read -r x y < <(frame_at "$R/flame_rdsp.svg" '^get_image_features')
shot "$R/flame_rdsp.svg" "$S/4_rdsp_vision_sync.png" "x=$(enc "$x")&y=$y&s=$(enc 'modeling_qwen3_5\.py:1070')"
ls -la "$S"
