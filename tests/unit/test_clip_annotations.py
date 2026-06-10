# SPDX-License-Identifier: Apache-2.0
"""AnnotationStore のユニットテスト — 分割 / dedup / 符号化キャッシュの計算節約を検証。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from llcore.clip.annotations import (
    UINT64_MASK,
    AnnotationStore,
    annotation_id,
    id_cosine,
    id_to_unit_vector,
    is_question,
    split_annotations,
)


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
    # 同一アノテーションは同一 uint64 id (content-addressed) で、ulonglong 範囲に収まる
    assert ids1[0] == ids2[0]
    assert all(0 <= i <= UINT64_MASK for i in ids1 + ids2)


def test_annotation_id_stable_and_uint64() -> None:
    """アノテーション番号は content-addressed で安定、かつ符号なし 64bit 範囲。"""
    a = annotation_id("my name is kazufumi")
    assert a == annotation_id("my name is kazufumi")  # 安定 (プロセス間で再現)
    assert a != annotation_id("my name is alice")
    assert 0 <= a <= UINT64_MASK


def test_id_to_unit_vector_is_cosine_ready() -> None:
    """uint64 ID → 実数化: 単位ノルム + 値域有界 + 全ビット無損失 (精度落ちなし)。"""
    big = UINT64_MASK - 12345  # 2^53 を遥かに超える ID
    v = id_to_unit_vector(big, dim=64)
    assert v.shape == (64,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)  # cosine 標準形 (単位ノルム)
    assert np.all(np.abs(v) <= 1.0)  # 値域有界 (数値的に扱いやすい)
    # 1bit だけ違う ID は別ベクトルになる (2^53 超でも潰れない=精度保持)
    v2 = id_to_unit_vector(big ^ 1, dim=64)
    assert not np.allclose(v, v2)
    # 同一 ID は決定論的に同一
    assert np.allclose(v, id_to_unit_vector(big, dim=64))


def test_id_to_unit_vector_cosine_separates_ids() -> None:
    """異なる ID 間の cosine は ~0 付近 (識別空間で互いにほぼ直交)、同一は 1。"""
    a = annotation_id("alpha")
    b = annotation_id("beta")
    va, vb = id_to_unit_vector(a), id_to_unit_vector(b)
    assert np.isclose(float(va @ va), 1.0, atol=1e-6)
    assert abs(float(va @ vb)) < 0.6  # 別 ID は識別的に分離 (SimHash 的)


def test_id_to_unit_vector_extended_dim() -> None:
    v = id_to_unit_vector(annotation_id("gamma"), dim=128)
    assert v.shape == (128,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_id_cosine_exact_matches_vector_dot() -> None:
    """popcount 厳密コサインが ±1 ベクトルの内積と一致 (精度喪失なし)。"""
    big = UINT64_MASK - 999
    other = annotation_id("some other annotation")
    for a, b in [(big, other), (big, big), (big, big ^ 0xF0F0)]:
        exact = id_cosine(a, b, dim=64)
        va, vb = id_to_unit_vector(a, 64), id_to_unit_vector(b, 64)
        assert exact == pytest.approx(float(va @ vb), abs=1e-6)
    # 同一 ID は厳密に 1.0 (float 誤差なし — 整数演算 + 2^6 除算)
    assert id_cosine(big, big) == 1.0
    # 1bit 違いは厳密に (64-2)/64
    assert id_cosine(big, big ^ 1) == pytest.approx((64 - 2) / 64)


def test_id_cosine_no_float_precision_loss_above_2e53() -> None:
    """2^53 を超える 2 ID でも識別が潰れない (生 int→float の精度喪失を回避)。"""
    a = (1 << 63) | 12345
    b = (1 << 63) | 12344  # 下位 1bit のみ違う巨大 ID
    assert a != b
    assert id_cosine(a, b) == pytest.approx((64 - 2) / 64)  # 厳密に 1bit 差
    assert id_cosine(a, b) < 1.0  # 潰れていない


def test_store_id_neighbors_exact() -> None:
    enc = CountingEncoder()
    store = AnnotationStore(enc)
    ids = store.add_text("alpha beta. gamma delta. epsilon zeta.")
    nb = store.id_neighbors(ids[0], k=2)
    assert len(nb) == 2
    # スコアは全て厳密な popcount コサイン (i 番目の行 ID と突合)
    for row, score in nb:
        assert score == id_cosine(ids[0], store.id_of_row(row))


def test_store_id_row_roundtrip() -> None:
    enc = CountingEncoder()
    store = AnnotationStore(enc)
    ids = store.add_text("alpha beta. gamma delta.")
    for aid in ids:
        row = store.row_of_id(aid)
        assert store.id_of_row(row) == aid
        assert store.annotation_of_id(aid) == store.annotations[row]
    assert store.ids == [annotation_id(a) for a in store.annotations]


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


def test_embedding_matrix_is_contiguous_float32() -> None:
    """大規模 cosine 用に連続 float32 行列で保持 (query 毎の vstack なし)。"""
    enc = CountingEncoder(dim=16)
    store = AnnotationStore(enc)
    store.add_text("alpha beta. gamma delta. epsilon zeta.")
    M = store.embedding_matrix()
    assert M.dtype == np.float32
    assert M.flags["C_CONTIGUOUS"]
    assert M.shape == (3, 16)
    # 単位ノルム不変条件
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-5)


def test_capacity_doubling_keeps_rows_intact() -> None:
    """容量倍々で増えても既存行が壊れない (大量 add でも整合)。"""
    enc = CountingEncoder(dim=8)
    store = AnnotationStore(enc)
    first_ids = store.add_text("a1 word. a2 word. a3 word.")  # 3 unique 句
    M_before = store.embedding_matrix().copy()
    for n in range(20):
        store.add_text(f"filler{n} alpha. filler{n} beta.")  # 多数追加で再確保を誘発
    M_after = store.embedding_matrix()
    assert M_after.shape[0] > M_before.shape[0]
    # 最初の 3 行は不変
    assert np.allclose(M_after[:3], M_before)
    # id も不変
    assert store.ids[:3] == [store.id_of_row(store.row_of_id(i)) for i in first_ids[:3]]


def test_int8_quantized_query_matches_float_ranking() -> None:
    """int8 量子化 query の順位が float 経路とほぼ一致 (大規模・省メモリ近似)。"""
    rng = np.random.default_rng(0)
    vocab = {f"phrase number {i}": list(rng.normal(size=24)) for i in range(40)}

    class V:
        def encode_texts(self, texts: Sequence[str]) -> Any:
            arr = np.array([vocab[t] for t in texts], dtype=float)
            return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    store = AnnotationStore(V())
    store.add_text(". ".join(vocab) + ".")
    Q, scale = store.int8_matrix()
    assert Q.dtype == np.int8
    q = "phrase number 7"
    top_float = [j for j, _ in store.query(q, k=5)]
    top_int8 = [j for j, _ in store.query(q, k=5, quantized=True)]
    assert top_float[0] == top_int8[0]  # top1 一致
    assert len(set(top_float) & set(top_int8)) >= 4  # top5 ほぼ一致


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
