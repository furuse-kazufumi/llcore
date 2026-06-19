# SPDX-License-Identifier: Apache-2.0
"""Tests for truncated-BPTT long-context training of constant-state LMs (pure-add).

The baseline ``Trainer`` re-inits the state to zeros every ``forward`` call, so its effective
BPTT window is ``block_size`` — the model never receives a gradient for dependencies longer
than ``block_size``. The TBPTT trainer instead streams long contiguous segments through the
``step`` API, carrying a DETACHED state across ``chunk_size`` sub-windows, so gradients still
truncate at ``chunk_size`` but the *forward* state reflects the whole segment so far. This is
the only training path that can teach the model to use context beyond ``block_size`` — the
prerequisite for any honest "uses long context" claim. It must be a pure-add module that does
not touch ``Trainer.train`` or ``forward``.
"""
from __future__ import annotations

import torch

from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM


def _recurrent() -> RecurrentLM:
    torch.manual_seed(0)
    return RecurrentLM(RecurrentConfig(vocab_size=8, block_size=16, n_layer=2, n_embd=32, state_size=32))


def _rwkv() -> RWKVLM:
    torch.manual_seed(0)
    return RWKVLM(RWKVConfig(vocab_size=8, block_size=16, n_layer=2, n_embd=32))


# --- detach_state: cuts the graph but preserves values, for both state shapes ---


def test_detach_state_recurrent() -> None:
    from llcore.lm.tbptt import detach_state

    model = _recurrent()
    idx = torch.zeros(2, dtype=torch.long)
    logits, state = model.step(idx)
    # make the state part of a graph
    loss = logits.sum()
    loss.backward()
    detached = detach_state(state)
    assert len(detached) == len(state)
    for d, s in zip(detached, state, strict=True):
        assert not d.requires_grad
        assert torch.equal(d, s.detach())


def test_detach_state_rwkv_namedtuple_fields() -> None:
    from llcore.lm.tbptt import detach_state

    model = _rwkv()
    idx = torch.zeros(2, dtype=torch.long)
    _, state = model.step(idx)
    detached = detach_state(state)
    assert len(detached) == len(state)
    for d, s in zip(detached, state, strict=True):
        assert type(d) is type(s)  # still an RWKVLayerState
        for df, sf in zip(d, s, strict=True):
            assert not df.requires_grad
            assert torch.equal(df, sf.detach())


# --- reset_state_slots: only the masked batch rows are reset to a fresh init ---


def test_reset_state_slots_recurrent() -> None:
    from llcore.lm.tbptt import reset_state_slots

    model = _recurrent()
    state = model.init_state(3)
    # perturb all rows so reset is observable
    state = [s + 1.0 for s in state]
    mask = torch.tensor([True, False, True])
    out = reset_state_slots(model, state, mask)
    for o, s in zip(out, state, strict=True):
        assert torch.equal(o[1], s[1])  # untouched row preserved
        assert torch.count_nonzero(o[0]) == 0  # reset row zeroed
        assert torch.count_nonzero(o[2]) == 0


# --- the trainer actually learns: train loss drops on a learnable repeating pattern ---


def test_tbptt_learns_repeating_pattern() -> None:
    from llcore.lm.tbptt import TBPTTConfig, TBPTTTrainer

    # period-4 repeating sequence over vocab 8: trivially learnable
    pattern = torch.tensor([1, 2, 3, 4] * 600, dtype=torch.long)
    train_ids = pattern[:1600]
    val_ids = pattern[1600:]
    model = _recurrent()
    cfg = TBPTTConfig(
        seg_len=64, chunk_size=16, batch_size=4, max_updates=300,
        eval_interval=50, eval_iters=10, seed=0,
    )
    trainer = TBPTTTrainer(model, cfg)
    result = trainer.train(train_ids, val_ids)
    history = result["history"]
    assert len(history) >= 2
    first = history[0]["train_loss"]
    last = history[-1]["train_loss"]
    assert last < first - 0.3  # clearly learned the pattern
    assert all(k in history[0] for k in ("update", "train_loss", "val_loss"))


def test_tbptt_runs_on_rwkv() -> None:
    from llcore.lm.tbptt import TBPTTConfig, TBPTTTrainer

    pattern = torch.tensor([1, 2, 3, 4, 5] * 400, dtype=torch.long)
    model = _rwkv()
    cfg = TBPTTConfig(seg_len=48, chunk_size=12, batch_size=4, max_updates=120, eval_interval=40, eval_iters=8, seed=0)
    result = TBPTTTrainer(model, cfg).train(pattern[:1500], pattern[1500:])
    assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]
