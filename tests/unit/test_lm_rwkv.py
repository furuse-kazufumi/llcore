# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.rwkv`."""
from __future__ import annotations

import math

import pytest
import torch

from llcore.lm.eval import held_out_report_any
from llcore.lm.generation import generate_text
from llcore.lm.rwkv import RWKVConfig, RWKVLM
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer


def test_rwkv_forward_shape_and_loss() -> None:
    cfg = RWKVConfig(vocab_size=16, block_size=8, n_layer=2, n_embd=16)
    model = RWKVLM(cfg)
    idx = torch.randint(0, 16, (3, 8))
    targets = torch.randint(0, 16, (3, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 8, 16)
    assert loss is not None
    assert loss.ndim == 0
    assert math.isfinite(loss.item())


def test_rwkv_is_causal() -> None:
    cfg = RWKVConfig(vocab_size=8, block_size=6, n_layer=1, n_embd=12)
    model = RWKVLM(cfg).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 0]])
    b = a.clone()
    b[0, -1] = 7
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[0, :5], lb[0, :5], atol=1e-6)
    assert not torch.allclose(la[0, 5], lb[0, 5], atol=1e-6)


def test_rwkv_forward_matches_step_scan() -> None:
    cfg = RWKVConfig(vocab_size=8, block_size=6, n_layer=2, n_embd=12)
    model = RWKVLM(cfg).eval()
    idx = torch.randint(0, 8, (2, 6))
    with torch.no_grad():
        logits, _ = model(idx)
        state = model.init_state(batch_size=2)
        steps = []
        for pos in range(idx.size(1)):
            logits_t, state = model.step(idx[:, pos], state)
            steps.append(logits_t.unsqueeze(1))
        stepped = torch.cat(steps, dim=1)
    assert torch.allclose(logits, stepped, atol=1e-6)


def test_rwkv_state_bytes_are_constant() -> None:
    cfg = RWKVConfig(vocab_size=8, block_size=4, n_layer=3, n_embd=10)
    model = RWKVLM(cfg)
    state = model.init_state(batch_size=2)
    base_bytes = model.state_bytes(state)
    _, next_state = model.step(torch.tensor([0, 1]), state)
    assert model.state_bytes(next_state) == base_bytes
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10)
    assert out.shape == (1, 11)


def test_rwkv_generate_text_works_with_shared_harness() -> None:
    text = "abcabcabcabcabcabc"
    tok = CharTokenizer.from_text(text)
    model = RWKVLM(RWKVConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_embd=16))
    sample = generate_text(model, tok, prompt="a", max_new_tokens=8, temperature=0.8, seed=0)
    assert sample.startswith("a")
    assert len(sample) == 9


def test_rwkv_trainer_and_report_any_integration() -> None:
    torch.manual_seed(0)
    text = ("0123456789" * 120) + "\n"
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids = ids[: int(ids.numel() * 0.9)]
    val_ids = ids[int(ids.numel() * 0.9) :]
    model = RWKVLM(RWKVConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_embd=24))
    Trainer(
        model,
        TrainConfig(
            max_iters=80,
            warmup_iters=8,
            lr_decay_iters=80,
            batch_size=12,
            eval_interval=80,
            eval_iters=4,
            seed=0,
        ),
    ).train(train_ids, val_ids)
    report = held_out_report_any(model, train_ids, val_ids, tok.vocab_size, block_size=16)
    assert report["n_tokens"] > 0
    assert math.isfinite(report["model_ppl"])


def test_rwkv_config_validation() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        RWKVConfig(vocab_size=0, block_size=4)
    with pytest.raises(ValueError, match="block_size"):
        RWKVConfig(vocab_size=8, block_size=0)
    with pytest.raises(ValueError, match="n_layer"):
        RWKVConfig(vocab_size=8, block_size=4, n_layer=0)
