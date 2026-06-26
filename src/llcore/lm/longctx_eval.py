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
from llcore.lm.ttt import TTTLinearLM

ConstantStateLM: TypeAlias = RecurrentLM | RWKVLM | TTTLinearLM


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


@torch.no_grad()
def streaming_metrics_by_band(
    model: ConstantStateLM,
    ids: torch.Tensor,
    band_edges: Sequence[int],
    *,
    unigram_logp: torch.Tensor | None = None,
    tail_start: int | None = None,
    chunk_size: int = 256,
) -> dict[str, object]:
    """Per-position-band NLL / top-k (and optional unigram floor) in ONE carried-state pass.

    This is the load-bearing long-context-quality metric. It scores every target position
    ``1..n-1`` once while carrying the O(1) state, then buckets each per-token loss by the
    target's *absolute position* into ``band_edges`` (e.g. ``[0,128,256,512,1024,2048]``).
    A flat curve from the first band out to far beyond ``block_size`` is the honest
    "constant-memory non-degradation" result; ``tail_mean`` (positions ``>= tail_start``)
    isolates the steady-state regime from the one-time cold-start warmup. The scalar
    ``streaming_nll`` is exactly the token-weighted mean of these bands and must never be
    reported on its own (it hides the warmup + the OOD-extrapolation structure).
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
    n = int(ids.size(0))
    if n < 2:
        raise ValueError(f"streaming_metrics_by_band needs >= 2 tokens, got {n}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    edges = sorted({int(e) for e in band_edges})
    if not edges:
        raise ValueError("band_edges must be non-empty")
    if edges[0] != 0:
        edges = [0, *edges]
    nb = len(edges)
    if tail_start is None:
        tail_start = edges[1] if nb > 1 else 0
    edges_t = torch.tensor(edges, dtype=torch.long)
    inputs, targets = ids[:-1], ids[1:]
    n_pred = n - 1
    sum_nll = [0.0] * nb
    cnt = [0] * nb
    top1 = [0] * nb
    top5 = [0] * nb
    sum_uni = [0.0] * nb
    tail_sum = 0.0
    tail_cnt = 0
    was_training = model.training
    model.eval()
    state: object = None
    for start in range(0, n_pred, chunk_size):
        stop = min(start + chunk_size, n_pred)
        logits_chunk: list[torch.Tensor] = []
        for i in range(start, stop):
            logits, state = model.step(inputs[i : i + 1], state)  # type: ignore[arg-type]
            logits_chunk.append(logits)
        chunk = torch.cat(logits_chunk, dim=0)
        tgt = targets[start:stop]
        ce = F.cross_entropy(chunk, tgt, reduction="none")
        pred1 = chunk.argmax(dim=-1)
        k = min(5, chunk.size(-1))
        top5_hit = (chunk.topk(k, dim=-1).indices == tgt.unsqueeze(-1)).any(dim=-1)
        positions = torch.arange(start + 1, stop + 1)
        bands = torch.bucketize(positions, edges_t, right=True) - 1
        uni = (-unigram_logp[tgt]) if unigram_logp is not None else None
        tmask = positions >= tail_start
        if bool(tmask.any()):
            tail_sum += float(ce[tmask].sum().item())
            tail_cnt += int(tmask.sum().item())
        for b in range(nb):
            m = bands == b
            c = int(m.sum().item())
            if c == 0:
                continue
            sum_nll[b] += float(ce[m].sum().item())
            cnt[b] += c
            top1[b] += int((pred1[m] == tgt[m]).sum().item())
            top5[b] += int(top5_hit[m].sum().item())
            if uni is not None:
                sum_uni[b] += float(uni[m].sum().item())
    if was_training:
        model.train()
    bands_out: list[dict[str, object]] = []
    for b in range(nb):
        if cnt[b] == 0:
            continue
        mean = sum_nll[b] / cnt[b]
        rep: dict[str, object] = {
            "lo": edges[b],
            "hi": (edges[b + 1] if b + 1 < nb else None),
            "n_tok": cnt[b],
            "mean_nll": mean,
            "ppl": math.exp(mean),
            "bpc": mean / math.log(2),
            "top1": top1[b] / cnt[b],
            "top5": top5[b] / cnt[b],
        }
        if unigram_logp is not None:
            u = sum_uni[b] / cnt[b]
            rep["unigram_nll"] = u
            rep["unigram_ppl"] = math.exp(u)
            rep["beats_unigram"] = bool(passes_gate(math.exp(mean), math.exp(u)))
        bands_out.append(rep)
    total_nll = sum(sum_nll)
    total_tok = sum(cnt)
    full_mean = total_nll / total_tok
    return {
        "bands": bands_out,
        "band_edges": edges,
        "full_mean_nll": full_mean,
        "full_mean_ppl": math.exp(full_mean),
        "full_mean_bpc": full_mean / math.log(2),
        "n_tok": total_tok,
        "tail_start": tail_start,
        "tail_mean_nll": (tail_sum / tail_cnt) if tail_cnt else None,
        "tail_mean_ppl": (math.exp(tail_sum / tail_cnt) if tail_cnt else None),
        "tail_n_tok": tail_cnt,
    }


@torch.no_grad()
def gpt_sliding_window_nll(gpt: CharGPT, ids: torch.Tensor, stride: int) -> tuple[float, int]:
    """Strided sliding-window NLL for a block_size-bounded GPT (HF fixed-length convention).

    A GPT cannot consume ``T > block_size`` in one window, so to score it on a long held-out
    sequence we slide a ``block_size`` window with step ``stride`` and, for each target,
    take the logit from the window where that target sits at the last (max-left-context)
    position. ``stride == block_size`` is the cheap non-overlapping baseline (each target sees
    ``~block_size/2`` left context on average); ``stride == 1`` is the gold "full context"
    baseline (each target sees a full ``block_size`` of left context) at O(T*block_size) cost.
    Every target ``1..n-1`` is scored exactly once under either stride.
    """
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1-D [T], got shape {tuple(ids.shape)}")
    n = int(ids.size(0))
    if n < 2:
        raise ValueError(f"gpt_sliding_window_nll needs >= 2 tokens, got {n}")
    block = gpt.config.block_size
    if stride <= 0 or stride > block:
        raise ValueError(f"stride must be in [1, block_size={block}], got {stride}")
    was_training = gpt.training
    gpt.eval()
    total = 0.0
    ntok = 0
    prev_end = 0  # highest target index already scored
    begin = 0
    while True:
        end = min(begin + block, n)
        last_target = min(end, n - 1)  # logits predict ids[begin+1..end]; valid targets <= n-1
        new_lo = prev_end + 1
        new_hi = last_target
        if new_hi >= new_lo:
            logits = gpt.forward_logits(ids[begin:end].unsqueeze(0))[0]  # [L, V]
            sel = logits[new_lo - 1 - begin : new_hi - begin]  # logits[j] predicts ids[begin+j+1]
            tgt = ids[new_lo : new_hi + 1]
            total += float(F.cross_entropy(sel, tgt, reduction="sum").item())
            ntok += int(new_hi - new_lo + 1)
            prev_end = new_hi
        if end >= n:
            break
        begin += stride
    if was_training:
        gpt.train()
    return total / ntok, ntok
