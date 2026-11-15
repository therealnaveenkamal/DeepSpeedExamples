#!/bin/bash
# queue_mimo_pyspy.sh, on the node: waits for queue_check.sh, then repeats the 9B py-spy step under
# a new tag. Why: on 2026-11-11 a code push from the Mac during the first rerun copied the earlier
# attempt's log over the live one, so MIMO was sampled during its startup, not in training.
# (Never push while a queue runs.) Last marker: "=== ALL DONE MIMOPYSPY".
set -uo pipefail
cd ~/rdsp
M=~/rdsp/bench_oct10/mimo_vs_rdsp
L9=9b_v1_tp2pp1dp2
mark() { echo "=== $* $(date -u +%H:%M:%S)"; }
until grep -qE "^=== ALL DONE CHECK" ~/queue_check.out 2>/dev/null; do sleep 10; done
mark "START pyspy-9b-b"
timeout 840 env TAG=-nov11b bash "$M/scripts/run.sh" pyspy $L9; rc=$?
sudo docker ps -q | xargs -r sudo docker kill > /dev/null 2>&1
pkill -f "[t]rain_vl.py"; sudo pkill -f "[p]y-spy record"; bash -c ". ~/venv/bin/activate; ray stop --force" > /dev/null 2>&1
mark "END pyspy-9b-b exit $rc; stack files $(ls "$M/$L9/profiles-nov11b"/pyspy-*.txt 2>/dev/null | wc -l)"
for f in "$M/$L9/profiles-nov11b"/pyspy-mimo-*.txt; do echo "$(basename "$f") training-step samples: $(grep -c forward_backward "$f")"; done
mark "ALL DONE MIMOPYSPY"
