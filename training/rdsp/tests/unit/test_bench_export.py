"""Samples exported from Megatron-Bridge's SFT pipeline reach rdsp unchanged:
the same tokens, images and supervised positions, in the same order and
grouping into microbatches."""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "bench", "mimo"), os.path.join(HERE, "..", "..")]

from export_bridge_batches import save_step, to_rdsp_sample  # noqa: E402


def bridge_sample(tokens, supervised):
    """One collated row as Megatron-Bridge's Qwen-VL collate leaves it:
    labels already shifted left by one, a float loss mask on the same
    (shifted) positions, -100 where unsupervised."""
    ids = torch.tensor([tokens])
    mask = torch.zeros(1, len(tokens))
    for t in supervised:
        mask[0, t - 1] = 1.0  # predicting token t happens at position t - 1
    labels = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], dim=1)
    labels = labels.masked_fill(mask == 0, -100)
    return {
        "input_ids": ids, "labels": labels, "loss_mask": mask,
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, len(tokens), dtype=torch.long),
        "position_ids": torch.arange(len(tokens)).unsqueeze(0),
        "visual_inputs": type("V", (), {"pixel_values": torch.randn(16, 6),
                                        "image_grid_thw": torch.tensor([[1, 4, 4]])})(),
    }


def test_labels_come_back_unshifted_on_the_same_supervised_tokens():
    sample = to_rdsp_sample(bridge_sample([5, 6, 7, 8, 9, 0], supervised=[3, 4]))
    assert sample["labels"].tolist() == [[-100, -100, -100, 8, 9, -100]]
    assert sample["input_ids"].tolist() == [[5, 6, 7, 8, 9, 0]]
    assert sample["pixel_values"].dtype == torch.bfloat16
    assert sample["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert sample["tokens"] == 2  # the loss mask's count


def test_exported_steps_group_rows_into_microbatches_in_order(tmp_path):
    import train_vl

    rows = [to_rdsp_sample(bridge_sample([i, i + 1, i + 2, 0], supervised=[2])) for i in range(8)]
    save_step(str(tmp_path), 0, rows[:4])
    save_step(str(tmp_path), 1, rows[4:])
    steps = list(train_vl.exported_microbatches(str(tmp_path), rows=2))
    assert len(steps) == 4  # 2 steps x 2 microbatches of 2 rows
    inputs, labels = steps[1]
    assert inputs["input_ids"][:, 0].tolist() == [2, 3]  # samples 2 and 3
    assert [p.shape for p in inputs["pixel_values"]] == [(16, 6), (16, 6)]
    assert labels.tolist() == [[-100, -100, 4, -100], [-100, -100, 5, -100]]
    assert set(inputs) == {"input_ids", "attention_mask", "mm_token_type_ids",
                           "pixel_values", "image_grid_thw"}
    assert train_vl.token_count(labels) == 2


def test_padding_trimmed_to_each_steps_longest_row(tmp_path):
    """Like a collate padding to a multiple of 128, but one length per step
    (a step's microbatches share a shape)."""
    import train_vl

    rows = []
    for real in (3, 5, 2, 4):
        sample = to_rdsp_sample(bridge_sample(list(range(1, 11)), supervised=[2]))
        sample["attention_mask"][:, real:] = 0
        rows.append(sample)
    save_step(str(tmp_path), 0, rows)
    shapes = [x["input_ids"].shape for x, _ in train_vl.exported_microbatches(
        str(tmp_path), rows=2, pad_multiple=4)]
    assert shapes == [(2, 8), (2, 8)]  # longest row 5 -> 8


def test_padding_trimmed_to_each_microbatchs_longest_row(tmp_path):
    """Per microbatch, as Megatron's collate pads each microbatch: rows 3 and
    5 -> 8, rows 2 and 4 -> 4."""
    import train_vl

    rows = []
    for real in (3, 5, 2, 4):
        sample = to_rdsp_sample(bridge_sample(list(range(1, 11)), supervised=[2]))
        sample["attention_mask"][:, real:] = 0
        rows.append(sample)
    save_step(str(tmp_path), 0, rows)
    got = list(train_vl.exported_microbatches(str(tmp_path), rows=2, pad_multiple=4,
                                              per_microbatch=True))
    assert [x["input_ids"].shape for x, _ in got] == [(2, 8), (2, 4)]
    assert [y.shape for _, y in got] == [(2, 8), (2, 4)]
