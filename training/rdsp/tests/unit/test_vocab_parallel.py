"""Vocab-parallel embedding and output head: two processes stand in for two
tensor-parallel ranks; each holds half of the vocabulary rows. Outputs and
gradients must equal the unsplit modules'."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from ray_deepspeed_pipeline.vocab_parallel import shard_vocab

VOCAB, HIDDEN, TP = 12, 4, 2


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.pos = nn.Embedding(5, HIDDEN)          # not vocab-sized: left alone
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, ids):
        return self.lm_head(self.embed_tokens(ids) + self.pos.weight[: ids.shape[1]])


def _rank(rank, init_file, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=TP)
    torch.manual_seed(0)
    full = Tiny()
    sharded = Tiny()
    sharded.load_state_dict(full.state_dict())
    shard_vocab(sharded, VOCAB, rank=rank, world=TP, group=lambda: dist.group.WORLD)
    ids = torch.tensor([[0, 5, 6, 11]])
    target = torch.randn(1, 4, VOCAB, generator=torch.Generator().manual_seed(1))
    (full(ids) * target).sum().backward()
    out = sharded(ids)
    (out * target).sum().backward()
    rows = slice(rank * VOCAB // TP, (rank + 1) * VOCAB // TP)
    torch.save({
        "out_close": torch.allclose(out, full(ids), atol=1e-6),
        "embed_grad": torch.allclose(sharded.embed_tokens.weight.grad,
                                     full.embed_tokens.weight.grad[rows], atol=1e-6),
        "head_grad": torch.allclose(sharded.lm_head.weight.grad,
                                    full.lm_head.weight.grad[rows], atol=1e-6),
        "pos_whole": sharded.pos.weight.shape == (5, HIDDEN),
        "shapes": (tuple(sharded.embed_tokens.weight.shape), tuple(sharded.lm_head.weight.shape)),
    }, os.path.join(out_dir, f"r{rank}.pt"))
    dist.destroy_process_group()


def test_sharded_embedding_and_head_match_the_unsplit_modules():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_rank, args=(os.path.join(d, "init"), d), nprocs=TP)
        for rank in range(TP):
            r = torch.load(os.path.join(d, f"r{rank}.pt"))
            assert r["shapes"] == ((VOCAB // TP, HIDDEN), (VOCAB // TP, HIDDEN))
            assert r["out_close"] and r["embed_grad"] and r["head_grad"] and r["pos_whole"]
