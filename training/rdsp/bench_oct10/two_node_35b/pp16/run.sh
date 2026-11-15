#!/bin/bash
# run.sh rdsp: Qwen3.5-35B-A3B as a 16-stage pipeline, one GPU per stage (plan.md run 9): stages
# 0-7 on node 1, 8-15 on node 2, the pipeline crossing the network once. 128 microbatches of one
# row. --cuts measured. On node 1, after ../scripts/prep.sh, with the Ray cluster up.
# Stage 0's vision encoder is recomputed (scripts/common.sh vision_first; VISION_RC=0: not).
# STEPS (default 50), RECOMPUTE=1 (every block), NAME (log suffix). Logs in logs/.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../scripts/common.sh"
export LOGS=$HERE/logs$LOGS_SUFFIX; mkdir -p "$LOGS"
TAG=""; [ "$RECOMPUTE" = 1 ] && TAG=-rc
case "${1:-}" in
  rdsp)
    rdsp "rdsp$TAG${NAME:-}" --stages 16 --cuts measured $(vision_first 1) $(stage_flags 1 15 1) \
      --rows 1 --microbatches 128 ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac
