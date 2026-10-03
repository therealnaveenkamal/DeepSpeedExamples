"""Colocated vision: who encodes which image, who receives its features, and
the per-rank vision engine, in one process."""

import pytest
import torch

from ray_deepspeed_pipeline.boundary import Grid
from ray_deepspeed_pipeline.vision import ColocatedVisionEngine, VisionLayout, route_images


def step_inputs(has_image):
    """Per microbatch, per row: one 4-patch image or none."""
    out = []
    for row_flags in has_image:
        out.append({
            "input_ids": torch.zeros(len(row_flags), 3, dtype=torch.long),
            "pixel_values": [torch.ones(4 if f else 0, 2) for f in row_flags],
            "image_grid_thw": [torch.tensor([[1, 2, 2]]) if f
                               else torch.zeros(0, 3, dtype=torch.long) for f in row_flags],
        })
    return out


def test_images_spread_over_every_rank_and_reach_their_rows_cell():
    inputs = step_inputs([[1, 1], [1, 1]])  # images 0..3 = (mb, row) in order
    layout = VisionLayout(first=Grid(dp=2), world=3)
    routes = [route_images(inputs, layout, rank) for rank in range(3)]
    owned = [[image for image, *_ in r["own"]] for r in routes]
    assert owned == [[0, 1], [2], [3]]
    # image 1 is row 1 of microbatch 0: data-parallel cell 1 of stage 0
    assert [(image, dests, rep) for image, _, _, dests, rep in routes[0]["own"]] == \
        [(0, [0], 0), (1, [1], 1)]
    assert routes[0]["need"] == [(0, 0, 0), (1, 2, 1)]  # (microbatch, image, owner)
    assert routes[1]["need"] == [(0, 1, 0), (1, 3, 2)]
    assert routes[2]["need"] == []


def test_rows_without_images_are_skipped_and_tp_peers_all_receive():
    inputs = step_inputs([[1, 0], [0, 1]])
    layout = VisionLayout(first=Grid(dp=1, tp=2), world=4)
    routes = [route_images(inputs, layout, rank) for rank in range(4)]
    assert [[i for i, *_ in r["own"]] for r in routes] == [[0], [], [3], []]
    assert routes[0]["own"][0][3:] == ([0, 1], 0)  # both TP ranks; TP rank 0 returns grads
    assert routes[1]["need"] == [(0, 0, 0), (1, 3, 2)]


def test_per_row_pixels_required():
    inputs = step_inputs([[1, 1]])
    inputs[0]["pixel_values"] = torch.cat(inputs[0]["pixel_values"])
    with pytest.raises(Exception, match="per row"):
        route_images(inputs, VisionLayout(first=Grid(), world=1), 0)


def tiny_vision():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5VisionModel"):
        pytest.skip("transformers without Qwen3.5")
    cfg = transformers.Qwen3_5VisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=16,
        patch_size=4, spatial_merge_size=2, temporal_patch_size=1, num_position_embeddings=16)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return transformers.Qwen3_5VisionModel(cfg).float()


def test_engine_features_and_gradients_match_the_encoder():
    """The engine encodes several images in one call and splits the output
    per image; its gradients are the encoder's, divided as asked."""
    import copy

    from ray_deepspeed_pipeline.vision import VisionTower

    encoder = tiny_vision()
    reference = copy.deepcopy(encoder)
    engine = ColocatedVisionEngine(VisionTower(encoder, "visual"),
                                   {"optimizer": {"type": "AdamW", "params": {"lr": 0.01}}},
                                   torch.device("cpu"))
    g = torch.Generator().manual_seed(1)
    grid = torch.tensor([[1, 4, 4]])
    images = [(i, torch.randn(16, 48, generator=g), grid) for i in (3, 7)]
    features = engine.forward(images, train=True)
    assert list(features) == [3, 7] and features[3].shape == (4, 16)

    full = reference(torch.cat([p for _, p, _ in images]), grid_thw=grid.repeat(2, 1)).pooler_output
    assert torch.allclose(torch.cat([features[3], features[7]]), full, atol=1e-6)

    grads = {3: torch.randn(4, 16, generator=g), 7: torch.randn(4, 16, generator=g)}
    engine.backward(grads)
    engine.reduce_gradients(group=None, divide_by=2)
    full.backward(torch.cat([grads[3], grads[7]]) / 2)
    for (name, p), q in zip(engine.tower.module.named_parameters(), reference.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6), name
