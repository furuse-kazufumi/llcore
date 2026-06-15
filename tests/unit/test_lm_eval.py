# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.eval` — unigram baseline + held-out perplexity."""
from __future__ import annotations

import math

import torch

from llcore.lm.eval import (
    held_out_nll,
    held_out_perplexity,
    passes_gate,
    unigram_nll,
    unigram_perplexity,
)
from llcore.lm.model import CharGPT, GPTConfig


def test_unigram_nll_matches_hand_computation() -> None:
    # train counts: 0 -> 3, 1 -> 1; vocab 2; alpha=1
    # probs = ([3,1]+1)/(4+2) = [4/6, 2/6]
    train = torch.tensor([0, 0, 0, 1])
    val = torch.tensor([0, 1])
    expected = -(math.log(4 / 6) + math.log(2 / 6)) / 2
    assert math.isclose(unigram_nll(train, val, vocab_size=2), expected, rel_tol=1e-9)


def test_unigram_perplexity_is_exp_of_nll() -> None:
    train = torch.tensor([0, 0, 1, 2])
    val = torch.tensor([0, 1, 2])
    nll = unigram_nll(train, val, vocab_size=3)
    assert math.isclose(unigram_perplexity(train, val, vocab_size=3), math.exp(nll), rel_tol=1e-9)


def test_unigram_uniform_corpus_ppl_near_vocab() -> None:
    # uniform corpus over 4 chars -> unigram ppl ~ 4
    train = torch.tensor([0, 1, 2, 3] * 50)
    val = torch.tensor([0, 1, 2, 3] * 5)
    ppl = unigram_perplexity(train, val, vocab_size=4)
    assert 3.5 < ppl < 4.5


def test_held_out_nll_finite_and_near_random_init() -> None:
    cfg = GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    val = torch.randint(0, 8, (200,))
    nll = held_out_nll(model, val, block_size=8, batch_size=8)
    # random-init model over 8 classes ~ ln(8) = 2.08 nats
    assert 1.6 < nll < 2.6
    assert math.isclose(held_out_perplexity(model, val, 8, 8), math.exp(nll), rel_tol=1e-9)


def test_held_out_eval_does_not_leave_model_in_eval_mode() -> None:
    cfg = GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    model.train()
    held_out_nll(model, torch.randint(0, 8, (200,)), block_size=8)
    assert model.training is True


def test_passes_gate() -> None:
    assert passes_gate(model_ppl=10.0, unigram_ppl=40.0) is True
    assert passes_gate(model_ppl=39.0, unigram_ppl=40.0) is False
    assert passes_gate(model_ppl=34.0, unigram_ppl=40.0, margin=0.85) is True
