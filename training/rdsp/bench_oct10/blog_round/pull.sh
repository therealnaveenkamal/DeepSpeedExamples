#!/bin/bash
# pull.sh, on the Mac: copy this session's logs and profiles from the node into the local tree.
# Never deletes or replaces newer local files. TRACES=1 also copies the Nsight exports (.sqlite).
set -e
cd "$(dirname "$0")"
O=../../../ops/h100; . "$O/env.sh"
SSH="ssh -i $O/key.pem -o StrictHostKeyChecking=no"
EX=(--exclude "*.nsys-rep"); [ "${TRACES:-0}" = 1 ] || EX+=(--exclude "*.sqlite")
mkdir -p logs
for d in bridge_vs_rdsp/27b_tp1pp8dp1/logs bridge_vs_rdsp/27b_tp1pp8dp1/profiles mimo_vs_rdsp/9b_v1_tp2pp1dp2/logs mimo_vs_rdsp/9b_v1_tp2pp2dp1/logs blog_round/prep_logs; do
  mkdir -p "../$d"
  rsync -az --update -e "$SSH" "${EX[@]}" "ubuntu@$IP:rdsp/bench_oct10/$d/" "../$d/" || true
done
for d in $($SSH "ubuntu@$IP" 'ls -d rdsp/bench_oct10/mimo_vs_rdsp/9b_v1_tp2pp1dp2/profiles-* 2>/dev/null'); do
  mkdir -p "../${d#rdsp/bench_oct10/}"
  rsync -az --update -e "$SSH" "${EX[@]}" "ubuntu@$IP:$d/" "../${d#rdsp/bench_oct10/}/" || true
done
for f in prep.out queue_profile.out setup.log; do
  rsync -az -e "$SSH" "ubuntu@$IP:$f" logs/ 2>/dev/null || true
done
grep -h "^=== " logs/queue_profile.out 2>/dev/null | tail -30
