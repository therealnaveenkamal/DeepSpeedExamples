#!/bin/bash
# pull2.sh, on the Mac: copy the pair's 27B logs (node 1) into the local tree. Never deletes or
# replaces newer local files.
set -e
cd "$(dirname "$0")"
O=../../../ops/h100; . "$O/env2.sh"
SSH="ssh -i $O/key2.pem -o StrictHostKeyChecking=no"
mkdir -p logs27b prep2_logs logs
rsync -az --update -e "$SSH" "ubuntu@$IP1:rdsp/bench_oct10/blog_round/logs27b/" logs27b/ || true
rsync -az --update -e "$SSH" "ubuntu@$IP1:rdsp/bench_oct10/blog_round/prep2_logs/" prep2_logs/ || true
for f in prep2.out queue_27b.out; do rsync -az -e "$SSH" "ubuntu@$IP1:$f" logs/ 2>/dev/null || true; done
grep -h "^=== " logs/queue_27b.out 2>/dev/null | tail -20
