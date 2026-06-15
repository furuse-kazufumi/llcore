# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.tokenizer`."""
from __future__ import annotations

import pytest

from llcore.lm.tokenizer import CharTokenizer


def test_from_text_is_sorted_and_deterministic() -> None:
    tok = CharTokenizer.from_text("cba abc")
    # vocab = sorted(set("cba abc")) = [' ', 'a', 'b', 'c']
    assert tok.itos == [" ", "a", "b", "c"]
    assert tok.vocab_size == 4
    # deterministic: same text -> same mapping
    assert CharTokenizer.from_text("cba abc") == tok


def test_encode_decode_roundtrip() -> None:
    text = "hello, world!\nこんにちは"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text


def test_encode_raises_on_oov() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.encode("z")


def test_encode_safe_maps_oov_to_default() -> None:
    tok = CharTokenizer.from_text("abc")
    assert tok.encode_safe("az", default=0) == [tok.stoi["a"], 0]


def test_empty_text_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        CharTokenizer.from_text("")


def test_duplicate_and_multichar_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CharTokenizer(["a", "a"])
    with pytest.raises(ValueError, match="single chars"):
        CharTokenizer(["ab"])


def test_save_load_roundtrip(tmp_path) -> None:
    tok = CharTokenizer.from_text("日本語 abc\n")
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded == tok
    assert loaded.vocab_size == tok.vocab_size
