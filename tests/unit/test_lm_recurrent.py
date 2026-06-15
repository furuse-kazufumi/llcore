# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.recurrent`."""
from __future__ import annotations

import math

import pytest
import torch

from llcore.lm.eval import held_out_report_any
from llcore.lm.generation import generate_text
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer


def test_recurrent_forward_shape_and_loss() -> None:
    cfg = RecurrentConfig(vocab_size=16, block_size=8, n_layer=2, n_embd=16, state_size=12)
    model = RecurrentLM(cfg)
    idx = torch.randint(0, 16, (3, 8))
    targets = torch.randint(0, 16, (3, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 8, 16)
    assert loss is not None
    assert loss.ndim == 0
    assert math.isfinite(loss.item())


def test_recurrent_forward_logits_matches_forward() -> None:
    cfg = RecurrentConfig(vocab_size=8, block_size=6, n_layer=1, n_embd=12, state_size=10)
    model = RecurrentLM(cfg)
    idx = torch.randint(0, 8, (2, 6))
    logits, _ = model(idx)
    assert torch.allclose(logits, model.forward_logits(idx))


def test_recurrent_is_causal() -> None:
    cfg = RecurrentConfig(vocab_size=8, block_size=6, n_layer=1, n_embd=12, state_size=10)
    model = RecurrentLM(cfg).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 0]])
    b = a.clone()
    b[0, -1] = 7
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[0, :5], lb[0, :5], atol=1e-6)
    assert not torch.allclose(la[0, 5], lb[0, 5], atol=1e-6)


def test_recurrent_generate_uses_constant_state_shape() -> None:
    cfg = RecurrentConfig(vocab_size=8, block_size=4, n_layer=3, n_embd=12, state_size=10)
    model = RecurrentLM(cfg)
    state = model.init_state(batch_size=2)
    base_bytes = model.state_bytes(state)
    _, next_state = model.step(torch.tensor([0, 1]), state)
    assert len(next_state) == cfg.n_layer
    assert model.state_bytes(next_state) == base_bytes
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10)
    assert out.shape == (1, 11)


def test_recurrent_generate_text_works_with_shared_harness() -> None:
    text = "abcabcabcabcabcabc"
    tok = CharTokenizer.from_text(text)
    model = RecurrentLM(
        RecurrentConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_embd=16, state_size=12)
    )
    sample = generate_text(model, tok, prompt="a", max_new_tokens=8, temperature=0.8, seed=0)
    assert sample.startswith("a")
    assert len(sample) == 9


def test_recurrent_report_any_and_trainer_integration() -> None:
    torch.manual_seed(0)
    text = ("0123456789" * 150) + "\n"
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids = ids[: int(ids.numel() * 0.9)]
    val_ids = ids[int(ids.numel() * 0.9) :]
    model = RecurrentLM(
        RecurrentConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_embd=24, state_size=24)
    )
    Trainer(
        model,
        TrainConfig(
            max_iters=120,
            warmup_iters=12,
            lr_decay_iters=120,
            batch_size=16,
            eval_interval=120,
            eval_iters=4,
            seed=0,
        ),
    ).train(train_ids, val_ids)
    report = held_out_report_any(model, train_ids, val_ids, tok.vocab_size, block_size=16)
    assert report["n_tokens"] > 0
    assert report["model_ppl"] < report["unigram_ppl"]


def test_recurrent_config_validation() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        RecurrentConfig(vocab_size=0, block_size=4)
    with pytest.raises(ValueError, match="block_size"):
        RecurrentConfig(vocab_size=8, block_size=0)
    with pytest.raises(ValueError, match="n_layer"):
        RecurrentConfig(vocab_size=8, block_size=4, n_layer=0)
