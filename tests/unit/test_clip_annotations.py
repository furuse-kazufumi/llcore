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


class VecEncoder:
    """指定ベクトル辞書をそのまま返す fake (多義性/リンク試験用)。"""

    def __init__(self, vecs: dict[str, list[float]]) -> None:
        self._vecs = vecs

    def encode_texts(self, texts: Sequence[str]) -> Any:
        arr = np.array([self._vecs[t] for t in texts], dtype=float)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_links_surface_connects_typos() -> None:
    """誤字は表層 (bigram Jaccard) エッジで接続される — 埋め込みが離れていても繋がる。"""
    vecs = {
        "what is my name": [1.0, 0.0, 0.0],
        "what is my nmae": [0.0, 1.0, 0.0],  # 故意に埋め込みを直交させる (誤字で意味が壊れた想定)
        "a bowl of soup": [0.0, 0.0, 1.0],
    }
    store = AnnotationStore(VecEncoder(vecs))
    store.add_text("what is my name. what is my nmae. a bowl of soup.")
    links = store.build_links(tau_sem=0.9, tau_surf=0.45)
    surf_pairs = {(e["a"], e["b"]) for e in links["surface"]}
    ann = store.annotations
    i, j = ann.index("what is my name"), ann.index("what is my nmae")
    assert (min(i, j), max(i, j)) in surf_pairs  # 誤字ペアが表層で接続
    # 無関係ペアは表層接続されない
    k = ann.index("a bowl of soup")
    assert not any(k in p for p in surf_pairs)


def test_links_semantic_keeps_multiple_edges() -> None:
    """意味エッジは argmax 1 本でなく閾値以上を複数保持 (多義性を潰さない)。"""
    vecs = {
        "mercury": [0.7, 0.7, 0.0],     # 惑星とも元素とも近い (多義)
        "venus and mars": [1.0, 0.1, 0.0],
        "thermometer metal": [0.1, 1.0, 0.0],
        "a bowl of soup": [0.0, 0.0, 1.0],
    }
    store = AnnotationStore(VecEncoder(vecs))
    store.add_text("mercury. venus and mars. thermometer metal. a bowl of soup.")
    links = store.build_links(k_sem=3, tau_sem=0.6, tau_surf=0.99)
    ann = store.annotations
    m = ann.index("mercury")
    sem_partners = {e["b"] if e["a"] == m else e["a"]
                    for e in links["semantic"] if m in (e["a"], e["b"])}
    assert ann.index("venus and mars") in sem_partners
    assert ann.index("thermometer metal") in sem_partners  # 両義のエッジが共存


def test_ambiguity_detects_bridging_node() -> None:
    """多義ノード (互いに似ていない近傍を持つ) はスコアが高い。"""
    vecs = {
        "mercury": [0.7, 0.7, 0.0],
        "venus and mars": [1.0, 0.05, 0.0],
        "thermometer metal": [0.05, 1.0, 0.0],
        "red planet mars": [0.99, 0.1, 0.0],
        "liquid metal element": [0.1, 0.99, 0.0],
    }
    store = AnnotationStore(VecEncoder(vecs))
    store.add_text("mercury. venus and mars. thermometer metal. red planet mars. liquid metal element.")
    ann = store.annotations
    amb_mercury = store.ambiguity(ann.index("mercury"), k=4, tau=0.6)
    amb_mars = store.ambiguity(ann.index("venus and mars"), k=4, tau=0.6)
    assert amb_mercury > amb_mars  # 橋渡しノードの方が多義的


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
