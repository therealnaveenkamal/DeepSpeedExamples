"""Pick a partition's cuts by timing them.

A cost model only estimates where a pipeline balances: it cannot see
attention cost growing with sequence length, an encoder's long patch
sequences, or the first stage holding more microbatches in flight. So
pick_cuts() starts from BalancedTransformerBlocks' estimate and walks to the
fastest cut: it times a few training steps for each neighbouring cut (one cut
moved by one block), moves to the fastest, and stops when no cut one or two
blocks away is faster. A cut that fails (out of memory) counts as infinitely slow.

Each trial is a throwaway engine on the first batch, repeated; the model and
the data stream are untouched, so the real run starts from the same weights.
"""

import dataclasses
import math
import statistics
import time
from collections.abc import Callable

from ray_deepspeed_pipeline.config import ExplicitCuts


def search_cuts(measure: Callable[[tuple], float], start: tuple, n_blocks: int,
                min_first: int = 0) -> tuple[tuple, dict]:
    """Greedy walk from `start` to a cut that no cut one or two blocks away
    beats. measure(cuts) returns seconds per step (math.inf if the cut
    cannot run). Cuts stay strictly increasing, the first at least
    `min_first`, the last below n_blocks. Returns (best cuts, {cuts: seconds}
    for every cut measured)."""
    trials = {}

    def cost(cuts):
        if cuts not in trials:
            trials[cuts] = measure(cuts)
        return trials[cuts]

    def valid(cuts):
        return (cuts[0] >= min_first and cuts[-1] < n_blocks
                and all(a < b for a, b in zip(cuts, cuts[1:])))

    def neighbours(cuts, step):
        moves = [tuple(c + (d if j == i else 0) for j, c in enumerate(cuts))
                 for i in range(len(cuts)) for d in (-step, step)]
        return [m for m in moves if valid(m)]

    current = tuple(start)
    while True:
        # one block away first; if none is faster, two blocks away, so that
        # one noisy reading next to the best cut does not end the walk
        for step in (1, 2):
            best = min(neighbours(current, step), key=cost, default=current)
            if cost(best) < cost(current):
                current = best
                break
        else:
            return current, trials


def pick_cuts(*, model, config, loss_fn, pipeline_config, sample: list, weights=None,
              steps: int = 5, log: Callable = print) -> ExplicitCuts:
    """The fastest cuts for this model, layout and batch, measured. sample:
    one step's (inputs, labels) entries, replayed for every trial; steps:
    training steps per trial, timed after the first two (warm-up)."""
    from ray_deepspeed_pipeline.api import initialize
    from ray_deepspeed_pipeline.errors import StepFailed, ValidationError
    from ray_deepspeed_pipeline.partition import find_block_list, partition_parameters

    stages = pipeline_config.stages
    start = tuple(p.block_start for p in partition_parameters(
        model, pipeline_config.partition, stages)[1:])
    n_blocks = len(find_block_list(model)[1])
    min_first = 0 if pipeline_config.colocated_vision is None else 1

    def measure(cuts):
        trial = dataclasses.replace(pipeline_config, partition=ExplicitCuts(cuts),
                                    prefetch=False)
        engine = None
        try:
            engine, _, _, _ = initialize(model=model, config=config, loss_fn=loss_fn,
                                         pipeline_config=trial, weights=weights)
            times = []
            for _ in range(steps):
                started = time.perf_counter()
                engine.train_batch(data_iter=iter(sample))
                times.append(time.perf_counter() - started)
            seconds = statistics.median(times[2:]) if steps > 2 else times[-1]
        except (StepFailed, ValidationError, RuntimeError) as e:
            log(f"cuts {cuts}: cannot run ({type(e).__name__})")
            seconds = math.inf
        finally:
            if engine is not None:
                engine.shutdown()
        if seconds < math.inf:
            log(f"cuts {cuts}: {seconds:.2f} s per step")
        return seconds

    best, trials = search_cuts(measure, start, n_blocks, min_first)
    log(f"picked cuts {best} after {len(trials)} trials, starting from {start}")
    return ExplicitCuts(best)
