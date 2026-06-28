# SPDX-License-Identifier: Apache-2.0
"""Evaluation: char unigram baseline + held-out perplexity for the char-LM.

The P0 acceptance gate compares the model's held-out perplexity against an order-0
(memoryless) char unigram baseline, on the *same* held-out split, same vocab, same
base (natural log). A context-aware LM that has actually learned structure must beat
the unigram baseline by a clear margin (see :func:`passes_gate`).
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Protocol

import torch
from torch.nn import functional as F

from llcore.lm.data import get_batch
from llcore.lm.device import model_device


class SupportsForwardLogits(Protocol):
    training: bool

    def eval(self) -> SupportsForwardLogits: ...

    def train(self, mode: bool = True) -> SupportsForwardLogits: ...

    def forward_logits(self, idx: torch.Tensor) -> torch.Tensor: ...

    def parameters(self) -> Iterator[torch.Tensor]: ...


class TrainableLM(SupportsForwardLogits, Protocol):
    def __call__(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


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
    model: TrainableLM,
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
    model: TrainableLM,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
) -> float:
    """``exp`` of :func:`held_out_nll`."""
    return math.exp(held_out_nll(model, val_ids, block_size, batch_size))


@torch.no_grad()
def estimate_loss(
    model: TrainableLM,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    generator: torch.Generator | None = None,
) -> float:
    """Mean loss over ``eval_iters`` random batches (training-time monitor)."""
    was_training = model.training
    model.eval()
    dev = model_device(model)
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, generator)
        x, y = x.to(dev), y.to(dev)
        _, loss = model(x, y)
        assert loss is not None
        losses[k] = loss.item()
    if was_training:
        model.train()
    return float(losses.mean().item())


@torch.no_grad()
def held_out_report(
    model: SupportsForwardLogits,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    vocab_size: int,
    block_size: int,
    batch_size: int = 32,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Backward-compatible alias of :func:`held_out_report_any`."""
    return held_out_report_any(model, train_ids, val_ids, vocab_size, block_size, batch_size, alpha)


@torch.no_grad()
def held_out_report_any(
    model: SupportsForwardLogits,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    vocab_size: int,
    block_size: int,
    batch_size: int = 32,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Score a model and the unigram baseline on the exact same held-out tokens."""
    was_training = model.training
    model.eval()
    counts = torch.bincount(train_ids, minlength=vocab_size).double()
    probs = (counts + alpha) / (train_ids.numel() + alpha * vocab_size)
    logp = torch.log(probs)
    n = val_ids.size(0)
    starts = list(range(0, n - block_size, block_size))
    if not starts:
        raise ValueError(f"val length {n} too small for block_size {block_size}")
    total_model = 0.0
    total_unigram = 0.0
    total_tok = 0
    for s in range(0, len(starts), batch_size):
        idxs = starts[s : s + batch_size]
        x = torch.stack([val_ids[i : i + block_size] for i in idxs])
        y = torch.stack([val_ids[i + 1 : i + 1 + block_size] for i in idxs])
        logits = model.forward_logits(x)
        flat_y = y.reshape(-1)
        total_model += float(
            F.cross_entropy(logits.view(-1, logits.size(-1)), flat_y, reduction="sum").item()
        )
        total_unigram += float(-logp[flat_y].sum().item())
        total_tok += int(flat_y.numel())
    model.train(was_training)
    model_nll = total_model / total_tok
    unigram_nll_aligned = total_unigram / total_tok
    return {
        "model_nll": model_nll,
        "unigram_nll": unigram_nll_aligned,
        "model_ppl": math.exp(model_nll),
        "unigram_ppl": math.exp(unigram_nll_aligned),
        "n_tokens": float(total_tok),
    }


@torch.no_grad()
def held_out_top1_report(
    model: SupportsForwardLogits,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
) -> dict[str, float]:
    """Teacher-forced next-token top-1 / top-5 accuracy over held-out windows.

    A *hard*-capability proxy that complements perplexity: top-1 asks "did the model
    rank the true next char first?", which can expose capability loss (e.g. from
    aggressive quantization) that a soft likelihood metric understates — the bit-width
    sweep (`scripts/quant_bitwidth_sweep.py`) showed a PPL-only gate passing a model
    whose top-1 had collapsed. Uses the same non-overlapping windowing as
    :func:`held_out_report_any` so the token set is identical.
    """
    was_training = model.training
    model.eval()
    n = val_ids.size(0)
    starts = list(range(0, n - block_size, block_size))
    if not starts:
        raise ValueError(f"val length {n} too small for block_size {block_size}")
    top1 = 0
    top5 = 0
    total = 0
    for s in range(0, len(starts), batch_size):
        idxs = starts[s : s + batch_size]
        x = torch.stack([val_ids[i : i + block_size] for i in idxs])
        y = torch.stack([val_ids[i + 1 : i + 1 + block_size] for i in idxs])
        logits = model.forward_logits(x)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_y = y.reshape(-1)
        top1 += int((flat_logits.argmax(dim=-1) == flat_y).sum().item())
        # top-5: is the true token among the 5 highest-scoring candidates?
        k = min(5, flat_logits.size(-1))
        top5_idx = flat_logits.topk(k, dim=-1).indices
        top5 += int((top5_idx == flat_y.unsqueeze(-1)).any(dim=-1).sum().item())
        total += int(flat_y.numel())
    model.train(was_training)
    return {"top1_acc": top1 / total, "top5_acc": top5 / total, "n_tokens": float(total)}


def passes_gate(model_ppl: float, unigram_ppl: float, margin: float = 0.85) -> bool:
    """P0 perplexity gate: model PPL must be ``<= margin * unigram_ppl``.

    ``margin=0.85`` is a deliberately conservative floor (≥15% below unigram); a
    genuinely-learned char-LM typically beats unigram by 2x+.
    """
    return model_ppl <= margin * unigram_ppl


def passes_capability_gate(
    model_top1: float, reference_top1: float, min_retention: float = 0.97
) -> bool:
    """Capability-retention gate: model must keep ``>= min_retention`` of a reference top-1.

    Complements :func:`passes_gate`, which only checks PPL against the unigram baseline.
    The bit-width sweep showed a perplexity-only gate can PASS a quantized model that has
    lost a large fraction of its exact next-token accuracy (e.g. a 2-bit model with top-1
    nearly halved still cleared the unigram PPL gate). This gate catches that by requiring
    the model to retain most of a reference (e.g. fp32) top-1 accuracy. A non-positive
    reference is treated as "no constraint" (any non-negative top-1 passes).
    """
    if reference_top1 <= 0.0:
        return model_top1 >= 0.0
    return model_top1 >= min_retention * reference_top1
