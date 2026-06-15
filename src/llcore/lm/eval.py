# SPDX-License-Identifier: Apache-2.0
"""Evaluation: char unigram baseline + held-out perplexity for the char-LM.

The P0 acceptance gate compares the model's held-out perplexity against an order-0
(memoryless) char unigram baseline, on the *same* held-out split, same vocab, same
base (natural log). A context-aware LM that has actually learned structure must beat
the unigram baseline by a clear margin (see :func:`passes_gate`).
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from llcore.lm.data import get_batch
from llcore.lm.model import CharGPT


def unigram_nll(
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    vocab_size: int,
    alpha: float = 1.0,
) -> float:
    """Mean per-char negative log-likelihood (nats) of the train unigram on val.

    Char probabilities are the add-``alpha``-smoothed empirical frequencies on the
    train split, so a val char unseen in train still gets finite probability.
    """
    counts = torch.bincount(train_ids, minlength=vocab_size).double()
    probs = (counts + alpha) / (train_ids.numel() + alpha * vocab_size)
    logp = torch.log(probs)
    return float(-logp[val_ids].mean().item())


def unigram_perplexity(
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    vocab_size: int,
    alpha: float = 1.0,
) -> float:
    """``exp`` of :func:`unigram_nll`."""
    return math.exp(unigram_nll(train_ids, val_ids, vocab_size, alpha))


@torch.no_grad()
def held_out_nll(
    model: CharGPT,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
) -> float:
    """Mean per-token cross-entropy (nats) over non-overlapping windows of val.

    Deterministic full pass: val is tiled into contiguous windows of ``block_size``;
    every predicted position contributes. This is the model-side number compared
    against :func:`unigram_nll`.
    """
    was_training = model.training
    model.eval()
    n = val_ids.size(0)
    starts = list(range(0, n - block_size, block_size))
    if not starts:
        raise ValueError(f"val length {n} too small for block_size {block_size}")
    total_nll = 0.0
    total_tok = 0
    for s in range(0, len(starts), batch_size):
        idxs = starts[s : s + batch_size]
        x = torch.stack([val_ids[i : i + block_size] for i in idxs])
        y = torch.stack([val_ids[i + 1 : i + 1 + block_size] for i in idxs])
        logits, _ = model(x)
        loss_sum = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        total_nll += float(loss_sum.item())
        total_tok += int(y.numel())
    if was_training:
        model.train()
    return total_nll / total_tok


def held_out_perplexity(
    model: CharGPT,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
) -> float:
    """``exp`` of :func:`held_out_nll`."""
    return math.exp(held_out_nll(model, val_ids, block_size, batch_size))


@torch.no_grad()
def estimate_loss(
    model: CharGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    generator: torch.Generator | None = None,
) -> float:
    """Mean loss over ``eval_iters`` random batches (training-time monitor)."""
    was_training = model.training
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, generator)
        _, loss = model(x, y)
        assert loss is not None
        losses[k] = loss.item()
    if was_training:
        model.train()
    return float(losses.mean().item())


def passes_gate(model_ppl: float, unigram_ppl: float, margin: float = 0.85) -> bool:
    """P0 perplexity gate: model PPL must be ``<= margin * unigram_ppl``.

    ``margin=0.85`` is a deliberately conservative floor (≥15% below unigram); a
    genuinely-learned char-LM typically beats unigram by 2x+.
    """
    return model_ppl <= margin * unigram_ppl
