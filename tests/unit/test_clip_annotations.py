# SPDX-License-Identifier: Apache-2.0
"""AnnotationStore のユニットテスト — 分割 / dedup / 符号化キャッシュの計算節約を検証。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from llcore.clip.annotations import AnnotationStore, split_annotations


class CountingEncoder:
    """encode_texts 呼び出し回数と渡されたテキストを記録する fake エンコーダ。"""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode_texts(self, texts: Sequence[str]) -> Any:
        self.calls.append(list(texts))
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**31))
        vecs = rng.normal(size=(len(texts), self.dim))
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    @property
    def total_encoded(self) -> int:
        return sum(len(c) for c in self.calls)


# -- 分割 ----------------------------------------------------------------------


def test_split_annotations_sentences_and_clauses() -> None:
    text = "My name is Kazufumi. I live in Japan, and I like pasta!"
    anns = split_annotations(text)
    assert "my name is kazufumi" in anns
    assert "i live in japan" in anns
    assert "and i like pasta" in anns


def test_split_annotations_normalizes_and_filters() -> None:
    assert split_annotations("  HELLO   World.  ") == ["hello world"]
    assert split_annotations("!!! ... ---") == []  # 記号のみは捨てる
    assert split_annotations("ab") == []  # 短すぎは捨てる


def test_split_annotations_japanese_punctuation() -> None:
    anns = split_annotations("私の名前はカズです。東京に住んでいます、よろしく。")
    assert "私の名前はカズです" in anns
    assert "東京に住んでいます" in anns


# -- ユニーク保持 + 符号化キャッシュ (計算節約の核) ------------------------------


def test_store_encodes_each_unique_annotation_once() -> None:
    enc = CountingEncoder()
    store = AnnotationStore(enc)
    ids1 = store.add_text("My name is Kazufumi. I like pasta.")
    ids2 = store.add_text("My name is Kazufumi. I like sushi.")  # 1 句は既出
    assert enc.total_encoded == 3  # ユニーク 3 句のみ符号化 (kazufumi/pasta/sushi)
    stats = store.stats()
    assert stats["unique_annotations"] == 3
    assert stats["total_instances"] == 4
    assert stats["encoder_calls_texts"] == 3
    assert stats["encode_saved_ratio"] == pytest.approx(1 - 3 / 4)
    # 同一アノテーションは同一 id
    assert ids1[0] == ids2[0]


def test_store_repeated_text_costs_zero_encodes() -> None:
    enc = CountingEncoder()
    store = AnnotationStore(enc)
    store.add_text("the sky is blue.")
    n_after_first = enc.total_encoded
    for _ in range(10):
        store.add_text("the sky is blue.")
    assert enc.total_encoded == n_after_first  # 再出現は符号化ゼロ
    # stats は 4 桁丸めのため abs 許容で比較
    assert store.stats()["encode_saved_ratio"] == pytest.approx(1 - 1 / 11, abs=1e-4)


def test_store_neighbors_and_query() -> None:
    enc = CountingEncoder()
    store = AnnotationStore(enc)
    store.add_text("alpha beta. gamma delta. epsilon zeta.")
    nb = store.neighbors(0, k=2)
    assert len(nb) == 2
    assert all(j != 0 for j, _ in nb)
    q = store.query("alpha beta", k=1)
    assert len(q) == 1


def test_store_empty_fail_closed() -> None:
    store = AnnotationStore(CountingEncoder())
    with pytest.raises(ValueError, match="empty"):
        store.embedding_matrix()
    with pytest.raises(ValueError, match="path"):
        store.save()


def test_store_save_load_roundtrip(tmp_path: Path) -> None:
    enc = CountingEncoder()
    path = tmp_path / "store.json"
    store = AnnotationStore(enc, path=path)
    store.add_text("alpha beta. gamma delta.")
    store.save()

    enc2 = CountingEncoder()
    store2 = AnnotationStore(enc2, path=path)
    assert store2.annotations == store.annotations
    assert store2.stats()["unique_annotations"] == 2
    # ロード後の再出現も符号化ゼロ (キャッシュが永続している)
    store2.add_text("alpha beta.")
    assert enc2.total_encoded == 0
    assert np.allclose(store2.embedding_matrix(), store.embedding_matrix())
