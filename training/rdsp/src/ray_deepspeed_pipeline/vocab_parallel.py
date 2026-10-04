"""Vocab-parallel input embedding and output head for tensor-parallel stages.

A large vocabulary makes the embedding and the output head the biggest single
matrices in a stage (Qwen3.5: 248k x hidden each). AutoTP leaves them whole
on every tensor-parallel rank, so each rank carries their full weights,
gradients and optimizer state. Here each rank keeps 1/tp of the vocabulary
rows instead, as Megatron does:

- embedding: a rank looks up the token ids in its row range, zeros the rest,
  and the ranks' outputs are summed (each id is in exactly one range);
- head: a rank computes the logits of its rows, and they are gathered along
  the vocabulary, so the loss sees ordinary full logits.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class _SumOverRanks(torch.autograd.Function):
    """Sum over the group forward; backward passes the gradient on as is
    (every rank's output feeds the same downstream computation)."""

    @staticmethod
    def forward(ctx, x, group):
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _SumGradOverRanks(torch.autograd.Function):
    """Identity forward; backward sums the gradient over the group: each
    rank's head shard sees only its vocabulary rows' share of the input
    gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _GatherLastDim(torch.autograd.Function):
    """Concatenate the ranks' tensors along the last dim; backward keeps this
    rank's slice of the gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group, ctx.width = group, x.shape[-1]
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        rank = dist.get_rank(ctx.group)
        return grad[..., rank * ctx.width:(rank + 1) * ctx.width].contiguous(), None


class VocabShardedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, rank: int, world: int, group):
        super().__init__()
        rows = embedding.num_embeddings // world
        self.start, self.stop = rank * rows, (rank + 1) * rows
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach()[self.start:self.stop].clone(),
                                   requires_grad=embedding.weight.requires_grad)
        self._group = group

    def forward(self, ids):
        outside = (ids < self.start) | (ids >= self.stop)
        out = F.embedding((ids - self.start).masked_fill(outside, 0), self.weight)
        out = out.masked_fill(outside.unsqueeze(-1), 0)
        return _SumOverRanks.apply(out, self._group())


class VocabShardedHead(nn.Module):
    def __init__(self, head: nn.Linear, rank: int, world: int, group):
        super().__init__()
        rows = head.out_features // world
        self.in_features, self.out_features = head.in_features, head.out_features
        self.weight = nn.Parameter(head.weight.detach()[rank * rows:(rank + 1) * rows].clone(),
                                   requires_grad=head.weight.requires_grad)
        self._group = group

    def forward(self, x):
        group = self._group()
        x = _SumGradOverRanks.apply(x, group)
        return _GatherLastDim.apply(F.linear(x, self.weight), group)


def shard_vocab(module: nn.Module, vocab_size: int, rank: int, world: int, group) -> int:
    """Replace, in place, every bias-free nn.Embedding / nn.Linear whose
    vocabulary dimension is vocab_size by its rank's shard. group: a
    callable returning the tensor-parallel process group (resolved at call
    time, so it can be created after this). Returns how many were replaced."""
    if vocab_size % world:
        return 0
    replaced = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Embedding) and child.num_embeddings == vocab_size \
                    and child.padding_idx is None and not child.weight.is_meta:
                setattr(parent, name, VocabShardedEmbedding(child, rank, world, group))
                replaced += 1
            elif isinstance(child, nn.Linear) and child.out_features == vocab_size \
                    and child.bias is None and not child.weight.is_meta:
                setattr(parent, name, VocabShardedHead(child, rank, world, group))
                replaced += 1
    return replaced
