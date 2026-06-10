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

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

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
    """

    def __init__(self, encoder: Any, path: Path | None = None) -> None:
        self._encoder = encoder
        self._path = path
        self._ann2idx: dict[str, int] = {}
        self._embeddings: list[Any] = []          # idx -> (d,) numpy 配列
        self._counts: list[int] = []              # idx -> 出現回数
        self._n_instances = 0                     # 観測したアノテーション延べ数
        self._n_encoded = 0                       # 実際に encoder を呼んだユニーク数
        if path is not None and path.exists():
            self._load(path)

    # -- 取り込み -------------------------------------------------------------

    def add_text(self, text: str, *, source: str | None = None) -> list[int]:
        """テキストを分割し、新規ユニークのみ符号化して、アノテーション id 列を返す。"""
        anns = split_annotations(text)
        new = [a for a in dict.fromkeys(anns) if a not in self._ann2idx]
        if new:
            vecs = self._encoder.encode_texts(new)
            for a, v in zip(new, vecs):
                self._ann2idx[a] = len(self._embeddings)
                self._embeddings.append(v)
                self._counts.append(0)
            self._n_encoded += len(new)
        ids = []
        for a in anns:
            idx = self._ann2idx[a]
            self._counts[idx] += 1
            ids.append(idx)
        self._n_instances += len(anns)
        return ids

    # -- 参照 -----------------------------------------------------------------

    @property
    def annotations(self) -> list[str]:
        """idx 順のユニークアノテーション一覧。"""
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

    # -- 永続化 ---------------------------------------------------------------

    def save(self) -> None:
        """path 指定時のみ JSON (+ .npz) へ保存。未指定なら ValueError (fail-closed)。"""
        import numpy as np

        if self._path is None:
            raise ValueError("AnnotationStore was created without a path")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "annotations": self.annotations,
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
        if emb.shape[0] != len(meta["annotations"]):
            raise ValueError("annotation store is corrupt: embeddings/annotations mismatch")
        self._ann2idx = {a: i for i, a in enumerate(meta["annotations"])}
        self._embeddings = [emb[i] for i in range(emb.shape[0])]
        self._counts = list(meta["counts"])
        self._n_instances = int(meta["n_instances"])
        self._n_encoded = int(meta["n_encoded"])
