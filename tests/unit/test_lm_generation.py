# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.generation` — sampling + degeneracy gate."""
from __future__ import annotations

from llcore.lm.generation import generate_text, is_degenerate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.tokenizer import CharTokenizer


def test_generate_text_length_and_prompt_prefix() -> None:
    text = "abcdefgh\n"
    tok = CharTokenizer.from_text(text)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    out = generate_text(model, tok, prompt="ab", max_new_tokens=20, seed=0)
    assert out.startswith("ab")
    assert len(out) == 2 + 20
    # every char is in-vocab (decoded from valid ids)
    assert all(c in tok.stoi for c in out)


def test_generate_text_deterministic_with_seed() -> None:
    text = "abcdefgh\n"
    tok = CharTokenizer.from_text(text)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = CharGPT(cfg)
    a = generate_text(model, tok, "a", 15, seed=7)
    b = generate_text(model, tok, "a", 15, seed=7)
    assert a == b


def test_is_degenerate_flags_repetition() -> None:
    # a back-to-back repeated block of length >= max_repeat_run
    looped = "abcdefghij" * 6  # period-10 loop
    assert is_degenerate(looped, min_distinct=5, max_repeat_run=10) is True


def test_is_degenerate_flags_low_diversity() -> None:
    assert is_degenerate("aaaaaaaaaaaaaaaa", min_distinct=15) is True


def test_is_degenerate_passes_varied_text() -> None:
    varied = (
        "The quick brown fox jumps over the lazy dog, "
        "and then wandered off into the misty wood."
    )
    assert is_degenerate(varied, min_distinct=15, max_repeat_run=10) is False
