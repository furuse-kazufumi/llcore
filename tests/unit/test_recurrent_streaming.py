# SPDX-License-Identifier: Apache-2.0
"""Tests for constant-memory long-context scoring on the recurrent LMs.

The structural memory win of a constant-state recurrent model is that it can score a
context of *any* length with O(1)-in-T state and O(1)-in-T activation memory — there is no
architectural reason for a ``block_size`` cap (that cap only exists on the batched
``forward``). :meth:`streaming_nll` exposes that: it scores a sequence longer than
``block_size`` without materializing O(T) logits, and matches ``forward``'s loss on short
sequences. (A GPT cannot do this: attention is O(T²) and ``block_size``-bounded.)
"""
from __future__ import annotations

import pytest
import torch

from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM


def _recurrent() -> RecurrentLM:
    torch.manual_seed(0)
    m = RecurrentLM(RecurrentConfig(vocab_size=32, block_size=16, n_layer=2, n_embd=24, state_size=24))
    m.eval()
    return m


def _rwkv() -> RWKVLM:
    torch.manual_seed(0)
    m = RWKVLM(RWKVConfig(vocab_size=32, block_size=16, n_layer=2, n_embd=24))
    m.eval()
    return m


# --- correctness: streaming matches the batched forward on a short sequence ---


def test_recurrent_streaming_nll_matches_forward() -> None:
    model = _recurrent()
    torch.manual_seed(3)
    ids = torch.randint(0, 32, (12,))  # within block_size
    nll, count = model.streaming_nll(ids)
    _, loss = model(ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0))
    assert loss is not None
    assert count == 11
    assert nll == pytest.approx(float(loss.item()), abs=1e-4)


def test_rwkv_streaming_nll_matches_forward() -> None:
    model = _rwkv()
    torch.manual_seed(3)
    ids = torch.randint(0, 32, (12,))
    nll, count = model.streaming_nll(ids)
    _, loss = model(ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0))
    assert loss is not None
    assert count == 11
    assert nll == pytest.approx(float(loss.item()), abs=1e-4)


# --- structural win: scores context far longer than block_size (forward would reject) ---


def test_recurrent_streaming_nll_handles_context_longer_than_block_size() -> None:
    model = _recurrent()  # block_size=16
    ids = torch.randint(0, 32, (200,))
    # forward refuses T > block_size ...
    with pytest.raises(ValueError):
        model(ids.unsqueeze(0))
    # ... streaming has no such cap (constant state = no context limit).
    nll, count = model.streaming_nll(ids)
    assert count == 199
    assert nll > 0.0 and torch.isfinite(torch.tensor(nll))


def test_rwkv_streaming_nll_handles_context_longer_than_block_size() -> None:
    model = _rwkv()
    ids = torch.randint(0, 32, (200,))
    with pytest.raises(ValueError):
        model(ids.unsqueeze(0))
    nll, count = model.streaming_nll(ids)
    assert count == 199
    assert nll > 0.0 and torch.isfinite(torch.tensor(nll))


# --- input validation ---


@pytest.mark.parametrize("factory", [_recurrent, _rwkv])
def test_streaming_nll_rejects_bad_input(factory) -> None:  # type: ignore[no-untyped-def]
    model = factory()
    with pytest.raises(ValueError):
        model.streaming_nll(torch.randint(0, 32, (1,)))  # need >= 2 tokens
    with pytest.raises(ValueError):
        model.streaming_nll(torch.randint(0, 32, (2, 4)))  # must be 1-D
