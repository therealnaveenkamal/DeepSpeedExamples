"""Where each GPU process spends its wall-clock time, from py-spy raw (folded) stacks.

    python analyze_pyspy.py <system> profiles/pyspy-<system>-*.txt > results/pyspy_<system>.txt

Only samples inside the training step count: stacks through Megatron's pipeline
schedule (forward_backward_pipelining...) or rdsp's stage worker (_run_step).
Each such sample is one 40 ms slice (25 Hz) of the training thread, labelled by the
first rule that matches its stack:
  sync         blocked on the GPU: a CUDA synchronize, or a Python call that waits
               for a GPU result (.item(), .tolist(), .cpu(), nonzero, ...)
  p2p          pipeline send/receive between the stages
  backward     inside autograd's backward (C++ engine and its hooks). Not any frame
               under torch/autograd: fused kernels and TE layers run their FORWARD
               through torch/autograd/function.py (an earlier version counted those
               as backward)
  forward      the rest of the forward pass (Python + kernel launches)
  other        anything else in the step (optimizer, schedule bookkeeping)
Stage: a process whose stacks reach the vision encoder is the first stage.
"""

import collections
import re
import sys

STEP = re.compile(r"forward_backward_pipelining|_run_step \(.*stage_worker")
VISION = re.compile(r"vision_model\.py|get_image_features|VisionModel|vision_utils")
RULES = (
    (
        "sync",
        re.compile(
            r"Synchronize|synchronize|_local_scalar_dense|\bitem \(|tolist \("
            r"|\bcpu \(|nonzero|is_nonzero|__bool__|__int__|__index__"
        ),
    ),
    (
        "p2p",
        re.compile(
            r"p2p_communication|_communicate|batch_isend_irecv|send_forward"
            r"|recv_forward|send_backward|recv_backward|p2p\.py|recv_boundary"
            r"|send_boundary|_receive_forward"
        ),
    ),
    ("backward", re.compile(r"backward \(|run_backward|_engine_run_backward")),
    ("forward", re.compile(r"forward \(|forward_step|_forward \(")),
)
ORDER = ("sync", "p2p", "backward", "forward", "other")
# where the forward samples go
PARTS = (
    ("vision", VISION),
    ("data", re.compile(r"get_batch|_to_device|prepare|iterator")),
    ("checks", re.compile(r"rerun_state_machine|validate_result|check_for_nan")),
    (
        "language",
        re.compile(
            r"transformer_block|transformer_layer|modeling_layers|text_model"
            r"|DecoderLayer|layers\.py"
        ),
    ),
)


def frame_name(f: str) -> str:
    return re.sub(r" \(.*/", " (", f)


def main() -> None:
    system, paths = sys.argv[1], sys.argv[2:]
    print(
        f"{system}: wall-clock share of the training thread per process "
        f"(py-spy, 25 Hz, 40 s while steady)\n"
    )
    stage_totals = collections.defaultdict(collections.Counter)
    stage_frames = collections.defaultdict(collections.Counter)
    stage_leaf = collections.defaultdict(collections.Counter)
    for path in paths:
        cats, frames, leaves = collections.Counter(), collections.Counter(), collections.Counter()
        vision = False
        for line in open(path):
            stack, _, n = line.rstrip("\n").rpartition(" ")
            if not stack or not n.isdigit() or not STEP.search(stack):
                continue
            n = int(n)
            vision |= bool(VISION.search(stack))
            label = next((c for c, rx in RULES if rx.search(stack)), "other")
            cats[label] += n
            if label == "forward":
                cats["fwd:" + next((c for c, rx in PARTS if rx.search(stack)), "rest")] += n
            parts = stack.split(";")
            for f in set(frame_name(x) for x in parts):
                frames[f] += n
            leaves[frame_name(parts[-1])] += n
        total = sum(cats[c] for c in ORDER)
        if not total:
            print(f"{path}: no training-step samples")
            continue
        stage = "stage 0" if vision else "stage 1"
        stage_totals[stage].update(cats)
        stage_frames[stage].update(frames)
        stage_leaf[stage].update(leaves)
        print(
            f"{path.rsplit('/', 1)[-1]:28} {stage}  "
            + "  ".join(f"{c} {100 * cats[c] / total:4.1f}%" for c in ORDER)
        )
        print(
            " " * 37
            + "forward = "
            + "  ".join(
                f"{c} {100 * cats['fwd:' + c] / total:4.1f}%"
                for c in [c for c, _ in PARTS] + ["rest"]
            )
        )
    for stage in sorted(stage_totals):
        cats = stage_totals[stage]
        total = sum(cats[c] for c in ORDER)
        print(
            f"\n== {stage}, all its processes: "
            + "  ".join(f"{c} {100 * cats[c] / total:.1f}%" for c in ORDER)
        )
        print(
            "  forward = "
            + "  ".join(
                f"{c} {100 * cats['fwd:' + c] / total:.1f}%"
                for c in [c for c, _ in PARTS] + ["rest"]
            )
        )
        print("  top functions on the stack (inclusive share):")
        for f, n in stage_frames[stage].most_common(30):
            print(f"    {100 * n / total:5.1f}%  {f[:120]}")
        print("  top innermost frames (where the thread actually was):")
        for f, n in stage_leaf[stage].most_common(15):
            print(f"    {100 * n / total:5.1f}%  {f[:120]}")


if __name__ == "__main__":
    main()
