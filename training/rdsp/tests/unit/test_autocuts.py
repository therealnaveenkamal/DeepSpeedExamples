"""Picking cuts by measuring them: a greedy walk over neighbouring cuts from
an estimate, keeping the fastest, each cut measured once."""

import math

from ray_deepspeed_pipeline.autocuts import search_cuts


def test_walks_from_the_estimate_to_the_fastest_cut():
    seen = []

    def measure(cuts):
        seen.append(cuts)
        return 5.0 + abs(cuts[0] - 10)

    best, trials = search_cuts(measure, start=(14,), n_blocks=24)
    assert best == (10,)
    assert len(seen) == len(set(seen))  # each cut measured once
    assert trials[(10,)] == 5.0 and (9,) in trials and (11,) in trials


def test_a_cut_that_runs_out_of_memory_counts_as_infinitely_slow():
    def measure(cuts):
        return math.inf if cuts[0] >= 13 else 5.0 + abs(cuts[0] - 12)

    best, _ = search_cuts(measure, start=(13,), n_blocks=24)
    assert best == (12,)


def test_more_stages_move_one_cut_at_a_time():
    def measure(cuts):
        return 5.0 + abs(cuts[0] - 6) + abs(cuts[1] - 13)

    best, _ = search_cuts(measure, start=(8, 16), n_blocks=24)
    assert best == (6, 13)


def test_cuts_stay_in_order_and_inside_the_model():
    measured = []

    def measure(cuts):
        measured.append(cuts)
        return float(sum(cuts))  # pulls every cut toward the low edge

    best, _ = search_cuts(measure, start=(2, 3), n_blocks=6, min_first=1)
    assert best == (1, 2)
    assert all(0 < a < b < 6 for a, b in measured)


def test_one_noisy_reading_does_not_stop_the_walk_early():
    """Cut 11 reads slow once (noise): from 13 the walk stalls at 12 by one
    block, so it also looks two blocks away and still reaches 10."""
    def measure(cuts):
        return 9.0 if cuts == (11,) else 5.0 + abs(cuts[0] - 10)

    best, _ = search_cuts(measure, start=(13,), n_blocks=24)
    assert best == (10,)
