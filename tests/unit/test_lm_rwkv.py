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


def test_rwkv_wkv_matches_reference_formula_without_extra_decay() -> None:
    cfg = RWKVConfig(vocab_size=8, block_size=4, n_layer=1, n_embd=3, bias=True)
    model = RWKVLM(cfg).eval()
    block = model.blocks[0]
    with torch.no_grad():
        block.time_mixer.mix_k.fill_(1.0)
        block.time_mixer.mix_v.fill_(1.0)
        block.time_mixer.mix_r.fill_(1.0)
        block.time_mixer.time_decay.fill_(0.0)
        block.time_mixer.time_first.fill_(0.0)
        for layer in (
            block.time_mixer.key,
            block.time_mixer.value,
            block.time_mixer.receptance,
            block.time_mixer.output,
        ):
            layer.weight.zero_()
            layer.bias.zero_()
            for i in range(cfg.n_embd):
                layer.weight[i, i] = 1.0

    x = torch.tensor([[0.1, -0.2, 0.3]])
    prev_x = torch.zeros_like(x)
    a = torch.tensor([[0.8, 0.4, -0.3]])
    b = torch.tensor([[1.5, 1.2, 0.9]])
    p = torch.tensor([[0.7, -0.1, 0.2]])
    out, next_a, next_b, next_p = block.time_mixer.step(x, prev_x, a, b, p)

    k = x
    v = x
    r = torch.sigmoid(x)
    u = torch.zeros_like(x)
    decay = -torch.exp(torch.zeros_like(x))
    q = torch.maximum(p, u + k)
    e1 = torch.exp(p - q)
    e2 = torch.exp(u + k - q)
    expected_wkv = (e1 * a + e2 * v) / (e1 * b + e2)
    expected_out = r * expected_wkv

    q2 = torch.maximum(p + decay, k)
    e1n = torch.exp(p + decay - q2)
    e2n = torch.exp(k - q2)
    expected_a = e1n * a + e2n * v
    expected_b = e1n * b + e2n

    assert torch.allclose(out, expected_out, atol=1e-6)
    assert torch.allclose(next_a, expected_a, atol=1e-6)
    assert torch.allclose(next_b, expected_b, atol=1e-6)
    assert torch.allclose(next_p, q2, atol=1e-6)


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


def test_rwkv_generate_rejects_empty_prompt() -> None:
    model = RWKVLM(RWKVConfig(vocab_size=8, block_size=8, n_layer=1, n_embd=16))
    with pytest.raises(ValueError, match="at least one prompt token"):
        model.generate(torch.zeros((1, 0), dtype=torch.long), max_new_tokens=1)


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
