"""train_vl.py --cuts: each spelling maps to one partition policy."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import ray_deepspeed_pipeline as rdsp  # noqa: E402
import train_vl  # noqa: E402


def test_balanced_measures_the_vision_ratio_from_the_first_step():
    policy = train_vl.partition_policy("balanced", ratio=lambda: 2.5)
    assert policy == rdsp.BalancedTransformerBlocks(vision_token_ratio=2.5)


def test_zero_then_balanced_gives_the_encoder_its_own_stage():
    policy = train_vl.partition_policy("0,balanced", ratio=lambda: 2.5)
    assert policy == rdsp.BalancedTransformerBlocks(vision_stage=True)


@pytest.mark.parametrize("cuts,policy", [
    ("even", rdsp.UniformTransformerBlocks()),
    ("0,16", rdsp.ExplicitCuts((0, 16))),
    ("13", rdsp.ExplicitCuts((13,))),
])
def test_even_and_explicit_cuts(cuts, policy):
    assert train_vl.partition_policy(cuts, ratio=lambda: 1 / 0) == policy
