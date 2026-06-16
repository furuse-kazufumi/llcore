# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.trainer` — LR schedule + end-to-end learning."""
from __future__ import annotations

from copy import deepcopy

import torch

from llcore.lm.data import encode_corpus, train_val_split
from llcore.lm.eval import held_out_perplexity, passes_gate, unigram_perplexity
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import Trainer, TrainConfig


def test_lr_schedule_warmup_and_decay() -> None:
    model = CharGPT(GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16))
    cfg = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
    trainer = Trainer(model, cfg)
    assert trainer.get_lr(0) < trainer.get_lr(50) < trainer.get_lr(99)  # warming up
    assert abs(trainer.get_lr(99) - 1e-3) < 2e-4  # ~peak at end of warmup
    assert trainer.get_lr(2000) == 1e-4  # clamped to min after decay window
    # cosine decay: mid-decay lr is between min and peak
    mid = trainer.get_lr(550)
    assert 1e-4 < mid < 1e-3


def test_optimizer_param_grouping() -> None:
    model = CharGPT(GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16))
    trainer = Trainer(model, TrainConfig())
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["weight_decay"] > 0  # 2-D matmul weights decayed
    assert groups[1]["weight_decay"] == 0  # biases / LayerNorm not decayed


def test_end_to_end_beats_unigram_on_learnable_pattern() -> None:
    """The whole stack: a periodic corpus is easily learned -> model PPL << unigram."""
    torch.manual_seed(0)
    # A learnable pattern (period 10) repeated; an n-gram model crushes unigram here.
    text = ("0123456789" * 400) + "\n"
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=0.1)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )
    model = CharGPT(cfg)
    train_cfg = TrainConfig(
        max_iters=300, warmup_iters=30, lr_decay_iters=300, batch_size=16,
        eval_interval=150, eval_iters=10,
    )
    Trainer(model, train_cfg).train(train_ids, val_ids)

    unigram_ppl = unigram_perplexity(train_ids, val_ids, tok.vocab_size)
    model_ppl = held_out_perplexity(model, val_ids, cfg.block_size, batch_size=8)
    assert model_ppl < unigram_ppl
    assert passes_gate(model_ppl, unigram_ppl)


def test_train_returns_history_and_loss_decreases() -> None:
    torch.manual_seed(0)
    text = ("abcde" * 300) + "\n"
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=12, n_layer=1, n_head=2, n_embd=32)
    )
    cfg = TrainConfig(
        max_iters=200, warmup_iters=20, lr_decay_iters=200, batch_size=16,
        eval_interval=100, eval_iters=10,
    )
    result = Trainer(model, cfg).train(train_ids, val_ids)
    history = result["history"]
    assert isinstance(history, list) and len(history) >= 2
    # training loss should fall from first to last eval
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def _assert_resume_matches_continuous_training(
    *, seed: int, dropout: float, eval_interval: int
) -> None:
    torch.manual_seed(seed)
    text = ("abcdefg" * 200) + "\n"
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=0.2)
    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=dropout
    )
    base_model = CharGPT(model_cfg)
    initial_state = deepcopy(base_model.state_dict())
    initial_rng_state = torch.get_rng_state()

    continuous_model = CharGPT(model_cfg)
    continuous_model.load_state_dict(initial_state)
    continuous_cfg = TrainConfig(
        max_iters=6,
        warmup_iters=2,
        lr_decay_iters=6,
        batch_size=8,
        eval_interval=eval_interval,
        eval_iters=1,
        seed=seed + 1,
    )
    torch.set_rng_state(initial_rng_state)
    continuous_trainer = Trainer(continuous_model, continuous_cfg)
    continuous_trainer.train(train_ids, val_ids)

    partial_model = CharGPT(model_cfg)
    partial_model.load_state_dict(initial_state)
    partial_cfg = TrainConfig(
        max_iters=3,
        warmup_iters=2,
        lr_decay_iters=6,
        batch_size=8,
        eval_interval=eval_interval,
        eval_iters=1,
        seed=seed + 1,
    )
    torch.set_rng_state(initial_rng_state)
    partial_trainer = Trainer(partial_model, partial_cfg)
    partial_trainer.train(train_ids, val_ids)
    partial_state = partial_trainer.state_dict()

    resumed_model = CharGPT(model_cfg)
    resumed_model.load_state_dict(partial_model.state_dict())
    resumed_trainer = Trainer(resumed_model, continuous_cfg)
    resumed_trainer.load_state_dict(partial_state)
    resumed_trainer.train(train_ids, val_ids)

    for name, param in continuous_model.state_dict().items():
        torch.testing.assert_close(param, resumed_model.state_dict()[name], rtol=0.0, atol=0.0)
    assert resumed_trainer.iter_num == continuous_trainer.iter_num == 6
    assert resumed_trainer.history == continuous_trainer.history


def test_trainer_resume_matches_continuous_training() -> None:
    _assert_resume_matches_continuous_training(seed=123, dropout=0.0, eval_interval=1)


def test_trainer_resume_matches_continuous_training_with_dropout() -> None:
    _assert_resume_matches_continuous_training(seed=456, dropout=0.2, eval_interval=1)


def test_trainer_resume_matches_continuous_training_with_sparse_evals() -> None:
    _assert_resume_matches_continuous_training(seed=789, dropout=0.2, eval_interval=2)
