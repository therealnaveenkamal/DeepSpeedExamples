"""Inside the vision encoder's forward: top inclusive and innermost frames, as a share
of all of the process's training-step samples."""
import collections
import re
import sys

pat = re.compile(sys.argv[1])
STEP = re.compile(r"forward_backward_pipelining|_run_step \(.*stage_worker")
BWD = re.compile(r"backward \(|run_backward|_engine_run_backward")  # not "autograd": forward runs through autograd.Function too
inc, leaf, total, vis = collections.Counter(), collections.Counter(), 0, 0
for path in sys.argv[2:]:
    for line in open(path):
        stack, _, n = line.rstrip("\n").rpartition(" ")
        if not stack or not n.isdigit() or not STEP.search(stack):
            continue
        n = int(n)
        total += n
        if not pat.search(stack) or BWD.search(stack):
            continue
        vis += n
        parts = [re.sub(r" \(.*/", " (", f) for f in stack.split(";")]
        i = max(i for i, f in enumerate(parts) if pat.search(f))
        for f in set(parts[i:]):
            inc[f] += n
        leaf[parts[-1]] += n
print(f"vision forward = {100 * vis / total:.1f}% of training-step samples")
print("inclusive, from the encoder down:")
for f, n in inc.most_common(28):
    print(f"  {100 * n / total:5.1f}%  {f[:115]}")
print("innermost:")
for f, n in leaf.most_common(14):
    print(f"  {100 * n / total:5.1f}%  {f[:115]}")
