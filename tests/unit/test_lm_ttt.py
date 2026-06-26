# SPDX-License-Identifier: Apache-2.0
"""Tests for the TTT-Linear constant-state char LM (``llcore.lm.ttt``).

Pins the public interface the trainer / long-context evaluators rely on (``step`` / ``forward`` /
``streaming_nll`` / ``init_state``), the constant-size fast-weight state (O(1) in sequence length),
and that one training step reduces the loss — so the L1 plateau experiment runs on a real model.
"""
from __future__ import annotations

import pytest
import torch

from llcore.lm.ttt import TTTLinearConfig, TTTLinearCore, TTTLinearLM


def _cfg(**kw: object) -> TTTLinearConfig:
    base = {"vocab_size": 23, "block_size": 16, "n_layer": 2, "n_embd": 32, "state_dim": 16}
    base.update(kw)
    return TTTLinearConfig(**base)  # type: ignore[arg-type]


def test_config_rejects_bad_values() -> None:
    for bad in ({"vocab_size": 0}, {"block_size": 0}, {"n_layer": 0}, {"n_embd": 0}, {"state_dim": 0}):
        with pytest.raises(ValueError):
            _cfg(**bad)


def test_forward_shapes_and_loss() -> None:
    torch.manual_seed(0)
    m = TTTLinearLM(_cfg())
    idx = torch.randint(0, 23, (4, 16))
    tgt = torch.randint(0, 23, (4, 16))
    logits, loss = m(idx, tgt)
    assert logits.shape == (4, 16, 23)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)


def test_forward_rejects_overlong_sequence() -> None:
    m = TTTLinearLM(_cfg(block_size=8))
    with pytest.raises(ValueError):
        m(torch.randint(0, 23, (1, 9)))


def test_state_is_constant_size_in_sequence_length() -> None:
    m = TTTLinearLM(_cfg())
    state = m.init_state(2)
    assert len(state) == 2  # one per layer
    assert all(tuple(s.shape) == (2, 16, 16) for s in state)
    bytes0 = m.state_bytes(state)
    # step many times -> state shape (and byte size) must not grow
    idx = torch.zeros(2, dtype=torch.long)
    for _ in range(50):
        _, state = m.step(idx, state)
    assert all(tuple(s.shape) == (2, 16, 16) for s in state)
    assert m.state_bytes(state) == bytes0


def test_step_matches_forward_logits_first_token() -> None:
    torch.manual_seed(1)
    m = TTTLinearLM(_cfg())
    m.eval()
    idx = torch.randint(0, 23, (3, 16))
    with torch.no_grad():
        full = m.forward_logits(idx)
        step0, _ = m.step(idx[:, 0])
    assert torch.allclose(full[:, 0, :], step0, atol=1e-5)


def test_streaming_nll_runs_beyond_block_size() -> None:
    torch.manual_seed(2)
    m = TTTLinearLM(_cfg(block_size=8))
    ids = torch.randint(0, 23, (200,))  # >> block_size: constant state has no context cap
    nll, n = m.streaming_nll(ids, chunk_size=32)
    assert n == 199 and nll > 0 and torch.isfinite(torch.tensor(nll))


def test_generate_appends_tokens() -> None:
    torch.manual_seed(3)
    m = TTTLinearLM(_cfg())
    out = m.generate(torch.randint(0, 23, (1, 4)), max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 9)


def test_one_optimizer_step_reduces_loss() -> None:
    torch.manual_seed(4)
    m = TTTLinearLM(_cfg())
    idx = torch.randint(0, 23, (8, 16))
    tgt = torch.randint(0, 23, (8, 16))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    _, loss0 = m(idx, tgt)
    assert loss0 is not None
    for _ in range(20):
        opt.zero_grad()
        _, loss = m(idx, tgt)
        assert loss is not None
        loss.backward()
        opt.step()
    _, loss1 = m(idx, tgt)
    assert loss1 is not None and float(loss1) < float(loss0)


def test_core_step_state_shape() -> None:
    torch.manual_seed(5)
    core = TTTLinearCore(_cfg(state_dim=16, n_embd=32))
    h = torch.randn(4, 32)
    s = torch.zeros(4, 16, 16)
    next_h, next_s = core.step(h, s)
    assert next_h.shape == (4, 32)
    assert next_s.shape == (4, 16, 16)
    assert torch.isfinite(next_s).all()
