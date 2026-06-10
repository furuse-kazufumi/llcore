# SPDX-License-Identifier: Apache-2.0
"""llcore.clip.annotations — アノテーション分割 + ユニーク保持 + 符号化キャッシュ。

ユーザー設計 (2026-06-11):
  「テキストをアノテーションに分割、アノテーションはユニークな値として保持、
   CLIP のテキストエンコーダーをアノテーションエンコーダーとして実装、
   といったことで計算量規模を抑えられないか」

実装方針:
- **分割 (v0 = rule-based, $0)**: 文境界 + 接続的カンマ/セミコロンで短句に割り、正規化
  (小文字化・空白圧縮) してアノテーション化。LLM 分割は将来の opt-in (計算削減という
  目的に反するため default にしない)。
- **ユニーク保持**: 正規化文字列をキーに dedup。ユニーク値の集合 = 世界モデルのノード語彙。
- **符号化キャッシュ**: 新規ユニークのみ CLIP text encoder で符号化 (1 回限り)。
  計算量はテキスト総量でなく**ユニーク数**に比例 — 会話が長いほど節約率が上がる。
  節約率は実測値 (``stats()``) として報告する (主張でなく測定)。

honest 留保:
- CLIP text encoder は短句に強く長文に弱い (≤64-77 token 制約) — アノテーション単位とは
  相性が良いが、**属性の束縛 (compositional binding) に弱い**ことが知られる
  (例:「赤い四角と青い円」の区別が崩れやすい)。連結性グラフ用途では許容しつつ開示。
- rule-based 分割は粗い。分割粒度はトレードオフ (細かすぎ=構成が失われ、粗すぎ=dedup 率低下)。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# アノテーション番号 (annotation id) の型上限 = 符号なし 64bit (ulonglong)。
# 大量のユニーク語を扱うため content-addressed な 64bit ID を採用 (dense index ではない):
# 同一アノテーションは store/シャードを跨いで常に同じ ID になる (分散・マージ可能)。
UINT64_MASK = (1 << 64) - 1


def annotation_id(annotation_norm: str) -> int:
    """正規化済みアノテーション文字列 → 安定な符号なし 64bit ID (ulonglong 範囲)。

    blake2b の先頭 8 バイトを uint64 に。content-addressed なので別プロセス・別ストアでも
    同一文字列は同一 ID。誕生日衝突は ~2^32 (約 43 億) ユニークで ~50% — 実用域では無視できるが、
    AnnotationStore は登録時に**異なる文字列が同一 ID になったら fail-closed で拒否**する。
    """
    h = hashlib.blake2b(annotation_norm.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & UINT64_MASK


def _char_bigrams(s: str) -> frozenset[str]:
    """表層類似用の文字 bigram 集合 (日本語にも有効 — 単語境界に依存しない)。"""
    t = f" {s} "
    return frozenset(t[i : i + 2] for i in range(len(t) - 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# 文末 (. ! ? 。!?) + 接続的区切り (, ; : 、;) で短句に割る
_SENT_SPLIT = re.compile(r"[.!?。!?]+[\s]*")
_CLAUSE_SPLIT = re.compile(r"[,;:、;:]+[\s]*")
_WS = re.compile(r"\s+")
# 記号のみ・極端に短い断片はアノテーションにしない
_MIN_CHARS = 3


def split_annotations(text: str) -> list[str]:
    """テキストを正規化済みアノテーション (短句) のリストに分割する。

    v0 は rule-based (文境界 + 接続区切り)。順序は保持、重複は呼び出し側で dedup。
    """
    out: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        for clause in _CLAUSE_SPLIT.split(sent):
            norm = _WS.sub(" ", clause).strip().lower()
            if len(norm) >= _MIN_CHARS and any(c.isalnum() for c in norm):
                out.append(norm)
    return out


class AnnotationStore:
    """ユニークなアノテーション → 埋め込み のキャッシュつきストア。

    Args:
        encoder: ``encode_texts(Sequence[str]) -> (n, d)`` を持つもの
            (:class:`llcore.clip.ClipBackend` または互換 fake)。
        path: 永続化先 (JSON + 埋め込みは npz 同名)。None ならメモリのみ。

    計算量の要: ``add_text`` は**新規ユニークのみ**を符号化する。
    ``stats()['encode_saved_ratio']`` が実測の節約率。

    内部表現は二層:
    - **行 (row, 0..n-1)**: 埋め込み行列の dense 添字。近傍/グラフ等の行列演算が使う。
    - **アノテーション番号 (id, uint64)**: content-addressed の安定外部 ID。
      ``add_text`` が返すのはこの id (大量語を扱うため ulonglong 範囲)。
      行 ⇄ id ⇄ 文字列の相互変換は :meth:`row_of_id` / :meth:`id_of_row` /
      :meth:`annotation_of_id` で行う。
    """

    def __init__(self, encoder: Any, path: Path | None = None) -> None:
        self._encoder = encoder
        self._path = path
        self._ann2idx: dict[str, int] = {}        # 正規化文字列 -> 行
        self._ids: list[int] = []                 # 行 -> uint64 アノテーション番号
        self._id2idx: dict[int, int] = {}         # uint64 id -> 行
        self._embeddings: list[Any] = []          # 行 -> (d,) numpy 配列
        self._counts: list[int] = []              # 行 -> 出現回数
        self._n_instances = 0                     # 観測したアノテーション延べ数
        self._n_encoded = 0                       # 実際に encoder を呼んだユニーク数
        if path is not None and path.exists():
            self._load(path)

    # -- 取り込み -------------------------------------------------------------

    def add_text(self, text: str, *, source: str | None = None) -> list[int]:
        """テキストを分割し、新規ユニークのみ符号化して、**アノテーション番号 (uint64) 列**を返す。"""
        anns = split_annotations(text)
        new = [a for a in dict.fromkeys(anns) if a not in self._ann2idx]
        if new:
            vecs = self._encoder.encode_texts(new)
            for a, v in zip(new, vecs):
                aid = annotation_id(a)
                existing = self._id2idx.get(aid)
                if existing is not None:
                    # 異なる文字列が同一 uint64 に衝突 → fail-closed (黙って上書きしない)
                    raise ValueError(
                        f"annotation id collision (uint64) between "
                        f"{self._ids[existing]!r}={self.annotations[existing]!r} and {a!r}"
                    )
                row = len(self._embeddings)
                self._ann2idx[a] = row
                self._ids.append(aid)
                self._id2idx[aid] = row
                self._embeddings.append(v)
                self._counts.append(0)
            self._n_encoded += len(new)
        ids = []
        for a in anns:
            row = self._ann2idx[a]
            self._counts[row] += 1
            ids.append(self._ids[row])
        self._n_instances += len(anns)
        return ids

    # -- id ⇄ 行 ⇄ 文字列 -----------------------------------------------------

    def row_of_id(self, annotation_id_: int) -> int:
        """アノテーション番号 (uint64) → dense 行。未登録なら KeyError。"""
        return self._id2idx[annotation_id_]

    def id_of_row(self, row: int) -> int:
        """dense 行 → アノテーション番号 (uint64)。"""
        return self._ids[row]

    def annotation_of_id(self, annotation_id_: int) -> str:
        """アノテーション番号 (uint64) → 正規化文字列。"""
        return self.annotations[self._id2idx[annotation_id_]]

    # -- 参照 -----------------------------------------------------------------

    @property
    def ids(self) -> list[int]:
        """行順のアノテーション番号 (uint64) 一覧。"""
        return list(self._ids)

    @property
    def annotations(self) -> list[str]:
        """行順のユニークアノテーション一覧。"""
        inv = [""] * len(self._ann2idx)
        for a, i in self._ann2idx.items():
            inv[i] = a
        return inv

    def embedding_matrix(self) -> "np.ndarray":
        import numpy as np

        if not self._embeddings:
            raise ValueError("store is empty")
        return np.vstack(self._embeddings)

    def neighbors(self, idx: int, k: int = 5) -> list[tuple[int, float]]:
        """アノテーション idx の近傍 (cosine 降順, 自身を除く) — 連結性の最小クエリ。"""
        import numpy as np

        M = self.embedding_matrix()
        sims = M @ M[idx]
        order = np.argsort(sims)[::-1]
        return [(int(j), float(sims[int(j)])) for j in order if int(j) != idx][:k]

    def query(self, text: str, k: int = 5) -> list[tuple[int, float]]:
        """自由テキスト 1 件でストアを cosine 検索 (符号化 1 回; キャッシュには入れない)。"""
        import numpy as np

        q = self._encoder.encode_texts([text])[0]
        M = self.embedding_matrix()
        sims = M @ q
        order = np.argsort(sims)[::-1]
        return [(int(j), float(sims[int(j)])) for j in order][:k]

    def stats(self) -> dict[str, float | int]:
        """実測の計算節約: encode_saved_ratio = 1 - (符号化回数 / 延べ出現数)。"""
        saved = 1.0 - (self._n_encoded / self._n_instances) if self._n_instances else 0.0
        return {
            "unique_annotations": len(self._ann2idx),
            "total_instances": self._n_instances,
            "encoder_calls_texts": self._n_encoded,
            "encode_saved_ratio": round(saved, 4),
        }

    # -- 連結グラフ (二層リンク + 多義性保持) -----------------------------------

    def build_links(
        self,
        k_sem: int = 5,
        tau_sem: float = 0.75,
        tau_surf: float = 0.45,
    ) -> dict[str, list[dict[str, object]]]:
        """アノテーション間の typed エッジを構築する (ユーザー要件 2026-06-11)。

        - **semantic**: CLIP cosine ≥ tau_sem の近傍を **top-k 複数保持**
          (argmax 1 本に潰さない = 多義性のエッジを残す)。
        - **surface**: 文字 bigram Jaccard ≥ tau_surf (誤字・表記揺れの接続。
          埋め込みが離れていても表層で繋がる)。
        返り値: {"semantic": [...], "surface": [...]} 各 edge = {a, b, weight}。
        """
        import numpy as np

        ann = self.annotations
        n = len(ann)
        edges_sem: list[dict[str, object]] = []
        edges_surf: list[dict[str, object]] = []
        if n < 2:
            return {"semantic": edges_sem, "surface": edges_surf}
        M = self.embedding_matrix()
        S = M @ M.T
        for i in range(n):
            order = np.argsort(S[i])[::-1]
            picked = 0
            for j in order:
                j = int(j)
                if j == i:
                    continue
                if picked >= k_sem or S[i, j] < tau_sem:
                    break
                if j > i:  # 無向エッジの重複回避
                    edges_sem.append({"a": i, "b": j, "weight": round(float(S[i, j]), 4)})
                picked += 1
        grams = [_char_bigrams(a) for a in ann]
        for i in range(n):
            for j in range(i + 1, n):
                w = _jaccard(grams[i], grams[j])
                if w >= tau_surf:
                    edges_surf.append({"a": i, "b": j, "weight": round(w, 4)})
        return {"semantic": edges_sem, "surface": edges_surf}

    def ambiguity(self, idx: int, k: int = 5, tau: float = 0.75) -> float:
        """多義性スコア: 自分の意味近傍同士の平均相互類似度の低さ (0=単義的, 1=多義的)。

        近傍が互いに似ていない = このアノテーションが異なる意味クラスタを橋渡し
        している、の素朴な定量化 (研究 PoC レベルの指標)。近傍 2 未満なら 0。
        """
        import numpy as np

        nb = [j for j, s in self.neighbors(idx, k=k) if s >= tau]
        if len(nb) < 2:
            return 0.0
        M = self.embedding_matrix()
        sims = [float(M[a] @ M[b]) for ai, a in enumerate(nb) for b in nb[ai + 1:]]
        return round(1.0 - float(np.mean(sims)), 4)

    # -- 永続化 ---------------------------------------------------------------

    def save(self) -> None:
        """path 指定時のみ JSON (+ .npz) へ保存。未指定なら ValueError (fail-closed)。"""
        import numpy as np

        if self._path is None:
            raise ValueError("AnnotationStore was created without a path")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "annotations": self.annotations,
            # JSON の数値は安全に uint64 を表現できないため id は 10 進文字列で保存
            "ids": [str(i) for i in self._ids],
            "counts": self._counts,
            "n_instances": self._n_instances,
            "n_encoded": self._n_encoded,
        }
        self._path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(self._path.with_suffix(".npz"), embeddings=self.embedding_matrix())

    def _load(self, path: Path) -> None:
        import numpy as np

        meta = json.loads(path.read_text(encoding="utf-8"))
        emb = np.load(path.with_suffix(".npz"))["embeddings"]
        anns = meta["annotations"]
        if emb.shape[0] != len(anns):
            raise ValueError("annotation store is corrupt: embeddings/annotations mismatch")
        self._ann2idx = {a: i for i, a in enumerate(anns)}
        # 後方互換: 旧形式 (ids 無し) は文字列から再生成
        self._ids = [int(s) for s in meta["ids"]] if "ids" in meta else [
            annotation_id(a) for a in anns
        ]
        self._id2idx = {aid: i for i, aid in enumerate(self._ids)}
        self._embeddings = [emb[i] for i in range(emb.shape[0])]
        self._counts = list(meta["counts"])
        self._n_instances = int(meta["n_instances"])
        self._n_encoded = int(meta["n_encoded"])
