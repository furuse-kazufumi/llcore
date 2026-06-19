# SPDX-License-Identifier: Apache-2.0
"""Confound-controlled long-context evaluation for constant-state recurrent LMs.

The structural memory win (constant O(1) state vs GPT's O(T) KV / O(T^2) attention) is
already settled with random models. The question these helpers answer is about a *trained*
model's QUALITY at long context, while neutralizing the obvious artifact traps:

- :func:`context_length_curve` is the headline, confound-free metric. For a *fixed* set of
  target positions it scores each target conditioning on exactly ``c`` preceding tokens
  (fresh zero-state window), sweeping ``c``. Holding the target positions fixed controls
  for text difficulty perfectly, and because every measurement starts cold there is no
  warmup asymmetry: a drop-then-plateau in NLL is a clean read of the model's *effective*
  context length, and it can probe ``c`` far beyond ``block_size`` (the helpers drive the
  model via ``step``, which has no context cap — only the batched ``forward`` does).

- :func:`block_reset_nll` is the bounded-context analogue of ``streaming_nll`` (reset the
  state every ``reset_every`` predictions). It mirrors ``streaming_nll``'s exact chunk
  arithmetic, so with ``reset_every`` past the sequence length it equals ``streaming_nll``.
  NOTE: resetting pays a cold-start penalty at each boundary, so ``streaming_nll`` <
  ``block_reset_nll`` conflates "long context helps" with "avoided repeated warmups" — that
  comparison is reported for context, but the load-bearing claim rests on
  :func:`context_length_curve`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch.nn import functional as F

from llcore.lm.eval import passes_gate
from llcore.lm.model import CharGPT
from llcore.lm.recurrent import RecurrentLM
from llcore.lm.rwkv import RWKVLM

ConstantStateLM: TypeAlias = RecurrentLM | RWKVLM


@torch.no_grad()
def run_steps(model: ConstantStateLM, x: torch.Tensor) -> torch.Tensor:
    """Logits ``[B, T, V]`` from the per-token ``step`` loop — NOT capped by ``block_size``.

    Identical computation to the models' ``forward`` (zero initial state, step per position),
    but bypasses the ``block_size`` guard so ``T`` may exceed the training window.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D [B,T], got shape {tuple(x.shape)}")
    state: object = None
    outs: list[torch.Tensor] = []
    for pos in range(x.size(1)):
        logits, state = model.step(x[:, pos], state)  # type: ignore[arg-type]
        outs.append(logits.unsqueeze(1))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def nll_at_positions_with_context(
    model: ConstantStateLM,
    ids: torch.Tensor,
    positions: Sequence[int],
    context_len: int,
    batch_size: int = 64,
) -> tuple[float, int]:
    """Mean NLL (nats) predicting ``ids[p]`` from a fresh window ``ids[p-context_len:p]``.

    Each target sees exactly ``context_len`` preceding tokens from a cold state, so results
    at different ``context_len`` on the *same* positions differ only in available context.
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
    if context_len <= 0:
        raise ValueError(f"context_len must be > 0, got {context_len}")
    n = int(ids.size(0))
    pos_list = [int(p) for p in positions]
    for p in pos_list:
        if p - context_len < 0 or p >= n:
            raise ValueError(
                f"position {p} invalid for context_len {context_len} and length {n} "
                f"(need 0 <= p-context_len and p < n)"
            )
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for s in range(0, len(pos_list), batch_size):
        batch_pos = pos_list[s : s + batch_size]
        x = torch.stack([ids[p - context_len : p] for p in batch_pos])
        y = torch.tensor([int(ids[p]) for p in batch_pos], dtype=torch.long)
        logits = run_steps(model, x)
        last = logits[:, -1, :]
        total += float(F.cross_entropy(last, y, reduction="sum").item())
        count += len(batch_pos)
    if was_training:
        model.train()
    return total / count, count


@torch.no_grad()
def context_length_curve(
    model: ConstantStateLM,
    ids: torch.Tensor,
    context_lens: Sequence[int],
    n_positions: int,
    seed: int,
    batch_size: int = 64,
) -> dict[str, object]:
    """Headline metric: mean NLL vs available context length on a *fixed* position sample.

    Samples ``n_positions`` target positions valid for the largest ``context_len`` (so the
    same positions are reused across every ``c``), then sweeps ``c``. The resulting
    ``nll_by_context`` curve shows how much context the trained model actually exploits —
    confound-free (text difficulty held fixed, every measurement cold-started).
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
    if n_positions <= 0:
        raise ValueError(f"n_positions must be > 0, got {n_positions}")
    lens = sorted({int(c) for c in context_lens})
    if not lens:
        raise ValueError("context_lens must be non-empty")
    max_c = lens[-1]
    n = int(ids.size(0))
    lo, hi = max_c, n - 1
    if hi < lo:
        raise ValueError(f"sequence length {n} too short for max context_len {max_c}")
    pool = hi - lo + 1
    if pool < n_positions:
        raise ValueError(
            f"only {pool} valid positions for context_len {max_c}, need {n_positions}"
        )
    gen = torch.Generator().manual_seed(seed)
    picked = torch.randperm(pool, generator=gen)[:n_positions] + lo
    positions = sorted(int(p) for p in picked.tolist())
    nll_by_context: dict[int, float] = {}
    for c in lens:
        nll, _ = nll_at_positions_with_context(model, ids, positions, c, batch_size=batch_size)
        nll_by_context[c] = nll
    return {
        "nll_by_context": nll_by_context,
        "ppl_by_context": {c: float(torch.tensor(v).exp().item()) for c, v in nll_by_context.items()},
        "n_positions": len(positions),
        "context_lens": lens,
        "positions_range": [lo, hi],
        "seed": seed,
    }


@torch.no_grad()
def block_reset_nll(
    model: ConstantStateLM,
    ids: torch.Tensor,
    reset_every: int,
    chunk_size: int = 256,
) -> tuple[float, int]:
    """Mean next-token NLL over the sequence, resetting the state every ``reset_every`` steps.

    Mirrors ``streaming_nll``'s exact chunk arithmetic, so ``reset_every`` past the sequence
    length reproduces ``streaming_nll`` to the bit. Smaller ``reset_every`` caps the context
    each prediction can use (the bounded-context analogue). See the module docstring caveat
    about the cold-start confound before reading any streaming-vs-reset gap as "context helps".
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
    n = int(ids.size(0))
    if n < 2:
        raise ValueError(f"block_reset_nll needs >= 2 tokens, got {n}")
    if reset_every <= 0:
        raise ValueError(f"reset_every must be > 0, got {reset_every}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    was_training = model.training
    model.eval()
    inputs, targets = ids[:-1], ids[1:]
    n_pred = n - 1
    total = 0.0
    for w_start in range(0, n_pred, reset_every):
        w_stop = min(w_start + reset_every, n_pred)
        state: object = None
        for start in range(w_start, w_stop, chunk_size):
            stop = min(start + chunk_size, w_stop)
            logits_chunk: list[torch.Tensor] = []
            for i in range(start, stop):
                logits, state = model.step(inputs[i : i + 1], state)  # type: ignore[arg-type]
                logits_chunk.append(logits)
            chunk = torch.cat(logits_chunk, dim=0)
            total += float(F.cross_entropy(chunk, targets[start:stop], reduction="sum").item())
    if was_training:
        model.train()
    return total / n_pred, n_pred
