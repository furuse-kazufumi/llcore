# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.data` — Aozora cleaning, split, batching."""
from __future__ import annotations

import pytest
import torch

from llcore.lm.data import (
    clean_aozora,
    encode_corpus,
    extract_aozora_text_from_zip_bytes,
    get_batch,
    train_val_split,
)
from llcore.lm.tokenizer import CharTokenizer

# A miniature Aozora-formatted document (header / legend / body / colophon).
AOZORA_SAMPLE = (
    "吾輩は猫である\n"
    "夏目漱石\n"
    "\n"
    "-------------------------------------------------------\n"
    "【テキスト中に現れる記号について】\n"
    "《》：ルビ\n"
    "-------------------------------------------------------\n"
    "吾輩《わがはい》は猫である。｜名前《なまえ》はまだ無い。\n"
    "　どこで生れたか［＃「生れた」に傍点］見当がつかぬ。\n"
    "\n"
    "底本：「吾輩は猫である」岩波書店\n"
    "入力：青空文庫\n"
)


def test_clean_aozora_strips_markup() -> None:
    body = clean_aozora(AOZORA_SAMPLE)
    assert "吾輩は猫である。名前はまだ無い。" in body
    assert "どこで生れたか見当がつかぬ。" in body
    # ruby readings, anchors, editor notes, headers, colophon all gone
    assert "《" not in body and "》" not in body
    assert "｜" not in body
    assert "［＃" not in body
    assert "底本" not in body
    assert "テキスト中に現れる記号" not in body
    assert "　" not in body  # full-width indent removed


def test_clean_aozora_decodes_shift_jis_bytes() -> None:
    raw = AOZORA_SAMPLE.encode("cp932")
    body = clean_aozora(raw)
    assert "名前はまだ無い" in body


def test_extract_aozora_text_from_zip_bytes() -> None:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sample.txt", AOZORA_SAMPLE.encode("cp932"))
    body = extract_aozora_text_from_zip_bytes(buf.getvalue())
    assert "名前はまだ無い" in body
    assert "《" not in body


def test_train_val_split_contiguous() -> None:
    ids = torch.arange(100)
    train, val = train_val_split(ids, val_frac=0.1)
    assert train.size(0) == 90
    assert val.size(0) == 10
    # contiguous trailing held-out
    assert torch.equal(val, torch.arange(90, 100))


def test_train_val_split_rejects_bad_frac() -> None:
    with pytest.raises(ValueError, match="val_frac"):
        train_val_split(torch.arange(100), val_frac=1.5)


def test_get_batch_shapes_and_shift() -> None:
    data = torch.arange(50)
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=gen)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted by one position (next-token targets)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_deterministic_with_generator() -> None:
    data = torch.arange(50)
    x1, _ = get_batch(data, 8, 4, torch.Generator().manual_seed(42))
    x2, _ = get_batch(data, 8, 4, torch.Generator().manual_seed(42))
    assert torch.equal(x1, x2)


def test_get_batch_rejects_short_data() -> None:
    with pytest.raises(ValueError, match="block_size"):
        get_batch(torch.arange(4), block_size=8, batch_size=2)


def test_encode_corpus_roundtrips() -> None:
    text = "abcabc\n"
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    assert ids.dtype == torch.long
    assert tok.decode(ids.tolist()) == text
