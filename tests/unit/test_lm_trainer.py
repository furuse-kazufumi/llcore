# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.trainer` — LR schedule + end-to-end learning."""
from __future__ import annotations

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
