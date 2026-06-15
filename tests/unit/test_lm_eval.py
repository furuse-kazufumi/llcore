# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.eval` — unigram baseline + held-out perplexity."""
from __future__ import annotations

import math

import torch

from llcore.lm.eval import (
    held_out_nll,
    held_out_perplexity,
    held_out_report,
    held_out_report_any,
    passes_gate,
    unigram_nll,
    unigram_perplexity,
)
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM


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


def test_held_out_report_scores_same_token_count() -> None:
    cfg = GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    train = torch.randint(0, 8, (400,))
    val = torch.randint(0, 8, (200,))
    r = held_out_report(model, train, val, vocab_size=8, block_size=8)
    # model and unigram are scored on the *same* number of tokens
    n_windows = len(range(0, 200 - 8, 8))
    assert r["n_tokens"] == n_windows * 8
    assert math.isclose(r["model_ppl"], math.exp(r["model_nll"]), rel_tol=1e-9)
    assert math.isclose(r["unigram_ppl"], math.exp(r["unigram_nll"]), rel_tol=1e-9)
    # aligned unigram ~ full-set unigram (unigram is position-independent / i.i.d.)
    full = unigram_nll(train, val, vocab_size=8)
    assert abs(r["unigram_nll"] - full) < 0.15


def test_held_out_report_any_matches_gpt_specific_report() -> None:
    cfg = GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    train = torch.randint(0, 8, (400,))
    val = torch.randint(0, 8, (200,))
    specific = held_out_report(model, train, val, vocab_size=8, block_size=8)
    generic = held_out_report_any(model, train, val, vocab_size=8, block_size=8)
    assert specific == generic


def test_held_out_report_any_supports_recurrent_lm() -> None:
    cfg = RecurrentConfig(vocab_size=8, block_size=8, n_layer=1, n_embd=16, state_size=12)
    model = RecurrentLM(cfg)
    train = torch.randint(0, 8, (400,))
    val = torch.randint(0, 8, (200,))
    report = held_out_report_any(model, train, val, vocab_size=8, block_size=8)
    assert report["n_tokens"] > 0
    assert math.isfinite(report["model_nll"])


def test_held_out_report_model_beats_unigram_on_pattern() -> None:
    from llcore.lm.data import encode_corpus, train_val_split
    from llcore.lm.tokenizer import CharTokenizer
    from llcore.lm.trainer import Trainer, TrainConfig

    torch.manual_seed(0)
    text = ("0123456789" * 300) + "\n"
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32)
    )
    Trainer(
        model,
        TrainConfig(max_iters=250, warmup_iters=25, lr_decay_iters=250, batch_size=16,
                    eval_interval=250, eval_iters=5),
    ).train(train_ids, val_ids)
    r = held_out_report(model, train_ids, val_ids, tok.vocab_size, 16)
    assert r["model_ppl"] < r["unigram_ppl"]
    assert passes_gate(r["model_ppl"], r["unigram_ppl"])


def test_passes_gate() -> None:
    assert passes_gate(model_ppl=10.0, unigram_ppl=40.0) is True
    assert passes_gate(model_ppl=39.0, unigram_ppl=40.0) is False
    assert passes_gate(model_ppl=34.0, unigram_ppl=40.0, margin=0.85) is True
