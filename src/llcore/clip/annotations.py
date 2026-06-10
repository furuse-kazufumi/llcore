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
from typing import Any

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


def id_to_unit_vector(annotation_id_: int, dim: int = 64) -> Any:
    """アノテーション番号 (uint64) → コサイン計算に適した単位ノルム実数ベクトル。

    なぜ単純キャストでないか (重要):
    - uint64 は最大 ~1.8e19。float64 の仮数は 52bit ＝ 2^53 を超える整数は**精度が落ち、
      異なる ID が同じ float に潰れうる**。生の int→float は禁止。
    - 単一スカラに正規化 (id/2^64 等) しても、コサインは 1 次元では符号しか見ず退化する。

    **採用 (適切な正規化)**: 64bit を各ビット {-1,+1} の 64 次元ベクトルへ**可逆展開**
    (全 64bit の情報を無損失で保持・値域有界) し、**L2 正規化** (cosine = 内積 の標準形)。
    dim<64 なら blake2b で dim*32bit を生成しビット展開 (任意長へ拡張可能)。
    こうして得たベクトル間 cosine は 2 ID のビット一致度 (SimHash 的) に対応する。

    注意: これは ID の**識別情報**を実数空間へ写すもので、意味的近さではない
    (意味的近さは CLIP 埋め込み側で測る)。用途に応じて使い分けること。
    """
    import numpy as np

    if dim <= 64:
        bits = [(annotation_id_ >> i) & 1 for i in range(dim)]
    else:
        # dim>64: blake2b で必要バイト数を生成しビット展開
        nbytes = (dim + 7) // 8
        raw = hashlib.blake2b(
            annotation_id_.to_bytes(8, "big"), digest_size=nbytes
        ).digest()
        bits = [(raw[b] >> (7 - (k % 8))) & 1 for k in range(dim) for b in [k // 8]][:dim]
    v = np.where(np.array(bits, dtype=np.float32) > 0, 1.0, -1.0)
    return v / np.float32(np.sqrt(dim))


def id_cosine(id_a: int, id_b: int, dim: int = 64) -> float:
    """2 つのアノテーション番号 (uint64) 間のコサインを **float を介さず厳密計算**する。

    :func:`id_to_unit_vector` の ±1 ビットベクトル表現に対するコサインは、
    整数の popcount だけで閉じる:

        cos(a, b) = (一致bit数 − 不一致bit数) / dim
                  = (dim − 2·popcount(a XOR b)) / dim

    - **精度を落とさない**: popcount は整数演算 (Python int は任意精度・誤差ゼロ)。
      最後の除算のみだが、分子は整数 [−dim, dim]、dim=64=2^6 なので結果は float64 で**厳密**
      (2^53 精度の壁にも掛からない)。生の int→float キャストの精度喪失を完全に回避。
    - **速い**: 64bit XOR + ハードウェア popcount (int.bit_count) で O(1)。ベクトル展開不要。

    dim≤64 のときに有効 (id_to_unit_vector の dim と一致させること)。dim>64 で
    blake2b 展開を使う場合はこの式は適用外 (ベクトル経路を使う)。
    """
    if dim > 64:
        raise ValueError("id_cosine は dim<=64 のビット表現専用 (dim>64 はベクトル経路を使う)")
    mask = (1 << dim) - 1
    disagree = ((id_a ^ id_b) & mask).bit_count()
    return (dim - 2 * disagree) / dim


def _l2_normalize(arr: Any) -> Any:
    """行ごとに L2 正規化 (cosine = 内積 の不変条件を保つ)。ゼロ行でも発散しない。"""
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        return a / max(float(np.linalg.norm(a)), 1e-12)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(norms, 1e-12)


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


# 疑問詞 (英) + 助動詞始まり。日本語は末尾 か/? で判定。
_QUESTION_WORDS = (
    "what", "where", "who", "whom", "whose", "when", "why", "which", "how",
    "is", "are", "am", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "shall", "may", "might", "have", "has", "had",
)


def is_question(annotation_norm: str) -> bool:
    """アノテーション (正規化短句) が疑問文か (= 事実でない) をヒューリスティック判定。

    事実検索から質問を除外するための軽量判定 (LLM 不要)。英: 疑問詞/助動詞始まり、
    日本語: 末尾 か / ? の有無。誤判定はありうる (PoC レベル) が、計算ゼロで効く。
    """
    s = annotation_norm.strip()
    if not s:
        return False
    if s.endswith("?") or s.endswith("か") or "?" in s:
        return True
    first = s.split()[0] if s.split() else ""
    return first in _QUESTION_WORDS


# 命令・依頼の動詞始まり + 「let's/let me」。これらは答え (事実) ではない。
_REQUEST_WORDS = (
    "suggest", "name", "tell", "describe", "list", "give", "explain", "show",
    "provide", "recommend", "switch", "say", "write", "find", "let's", "lets",
    "let", "please",
)


def is_request(annotation_norm: str) -> bool:
    """アノテーションが命令・依頼か (= 事実でない) を判定。

    「suggest one pasta dish」「let's talk about cooking」「name one composer」等は
    答えでなく依頼 → 事実検索から除外する (質問とは別カテゴリ、計算ゼロ)。
    """
    s = annotation_norm.strip().lower()
    if not s:
        return False
    first = s.split()[0] if s.split() else ""
    return first in _REQUEST_WORDS


def is_fact(annotation_norm: str) -> bool:
    """アノテーションが事実 (平叙文) か = 質問でも依頼でもない。"""
    return not is_question(annotation_norm) and not is_request(annotation_norm)


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
        # 埋め込みは **連続 float32 行列** (容量倍々で増やす) として保持する。
        # 行は単位 L2 ノルムに正規化済 (不変条件) ⇒ cosine = 内積 = 単一 matmul。
        # 大規模 (大きい桁数=多数の語) でも query 毎の vstack を避けられる。
        self._matrix: Any = None                  # (capacity, d) float32 or None
        self._n_rows = 0
        self._counts: list[int] = []              # 行 -> 出現回数
        self._roles: list[str | None] = []        # 行 -> 初出時の発話者ロール (user/assistant/None)
        self._is_q: list[bool] = []               # 行 -> 質問か
        self._non_fact: list[bool] = []           # 行 -> 非事実か (質問 or 依頼; 事実検索の除外用)
        # 共起連結: (行a, 行b) -> 共起回数 (同一 group= 会話 turn 窓 内で一緒に出た)。
        # cosine が繋げない「質問→その答え」を会話の隣接構造で橋渡しする連結性グラフ。
        self._cooc: dict[tuple[int, int], int] = {}
        self._recent_group: list[tuple[int, int]] = []  # (group_id, 行) の直近履歴
        self._n_instances = 0                     # 観測したアノテーション延べ数
        self._n_encoded = 0                       # 実際に encoder を呼んだユニーク数
        # int8 量子化キャッシュ (大規模 cosine の省メモリ近似; 行追加で無効化)
        self._int8: Any = None
        if path is not None and path.exists():
            self._load(path)

    # -- 内部: 連続行列の確保 -------------------------------------------------

    def _ensure_capacity(self, dim: int, need: int) -> None:
        import numpy as np

        if self._matrix is None:
            cap = max(8, need)
            self._matrix = np.zeros((cap, dim), dtype=np.float32)
            return
        cap = self._matrix.shape[0]
        if self._n_rows + need > cap:
            new_cap = cap
            while self._n_rows + need > new_cap:
                new_cap *= 2
            grown = np.zeros((new_cap, self._matrix.shape[1]), dtype=np.float32)
            grown[: self._n_rows] = self._matrix[: self._n_rows]
            self._matrix = grown

    # -- 取り込み -------------------------------------------------------------

    def add_text(
        self,
        text: str,
        *,
        source: str | None = None,
        role: str | None = None,
        group: int | None = None,
        adjacency_window: int = 1,
    ) -> list[int]:
        """テキストを分割し、新規ユニークのみ符号化して、**アノテーション番号 (uint64) 列**を返す。

        role は発話者 (例 "user"/"assistant") — 初出時に行へ記録。
        group は会話 turn 等のまとまり ID。同一/隣接 group (``adjacency_window`` ターン以内)
        に出たアノテーション同士に**共起エッジ**を張る = cosine が繋げない「質問→その答え」を
        会話の隣接構造で橋渡しする連結性グラフ (差別化の核)。group=None なら共起は張らない。
        """
        import numpy as np

        anns = split_annotations(text)
        new = [a for a in dict.fromkeys(anns) if a not in self._ann2idx]
        if new:
            vecs = np.asarray(self._encoder.encode_texts(new), dtype=np.float32)
            # 単位ノルム不変条件を強制 (encoder のドリフト/数値誤差を吸収)
            vecs = _l2_normalize(vecs).astype(np.float32)
            self._ensure_capacity(vecs.shape[1], len(new))
            for a, v in zip(new, vecs):
                aid = annotation_id(a)
                existing = self._id2idx.get(aid)
                if existing is not None:
                    # 異なる文字列が同一 uint64 に衝突 → fail-closed (黙って上書きしない)
                    raise ValueError(
                        f"annotation id collision (uint64) between "
                        f"{self._ids[existing]!r}={self.annotations[existing]!r} and {a!r}"
                    )
                row = self._n_rows
                self._ann2idx[a] = row
                self._ids.append(aid)
                self._id2idx[aid] = row
                self._matrix[row] = v
                self._n_rows += 1
                self._counts.append(0)
                self._roles.append(role)
                self._is_q.append(is_question(a))
            self._n_encoded += len(new)
            self._int8 = None  # 量子化キャッシュ無効化
        ids = []
        rows_here: list[int] = []
        for a in anns:
            row = self._ann2idx[a]
            self._counts[row] += 1
            ids.append(self._ids[row])
            rows_here.append(row)
        self._n_instances += len(anns)

        # 共起エッジ: group 指定時のみ。同一/隣接 group の行同士を連結。
        if group is not None and rows_here:
            # 隣接 group 内の既出行 (window 以内) を集める
            neighbors = {
                r for g, r in self._recent_group if abs(g - group) <= adjacency_window
            }
            for r in rows_here:
                for nb in neighbors | set(rows_here):
                    if nb != r:
                        key = (min(r, nb), max(r, nb))
                        self._cooc[key] = self._cooc.get(key, 0) + 1
            self._recent_group.extend((group, r) for r in rows_here)
            # 窓外の履歴を捨てる (メモリ抑制)
            self._recent_group = [
                (g, r) for g, r in self._recent_group if group - g <= adjacency_window
            ]
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

    def embedding_matrix(self) -> Any:
        """有効行のみの連続 float32 行列ビュー (n, d)。query 毎の再構築コストなし。"""
        if self._n_rows == 0:
            raise ValueError("store is empty")
        return self._matrix[: self._n_rows]

    def int8_matrix(self) -> tuple[Any, float]:
        """単位ノルム行列を int8 量子化した (Q, scale) を返す (大規模 cosine の省メモリ近似)。

        行が単位ノルムなので各成分は [-1,1] → ``round(x*127)`` で int8 化。
        cosine(a,b) ≈ (Qa·Qb) * scale, scale = 1/127² 。メモリは float32 の 1/4。
        結果はキャッシュし、行追加で無効化する。
        """
        import numpy as np

        if self._n_rows == 0:
            raise ValueError("store is empty")
        if self._int8 is None:
            M = self._matrix[: self._n_rows]
            self._int8 = np.clip(np.round(M * 127.0), -127, 127).astype(np.int8)
        return self._int8, 1.0 / (127.0 * 127.0)

    def id_neighbors(self, annotation_id_: int, k: int = 5, dim: int = 64) -> list[tuple[int, float]]:
        """**ID 空間** (識別) の厳密コサイン近傍を popcount で返す (float 精度喪失なし)。

        意味的近傍 (:meth:`neighbors`, CLIP 埋め込み) とは別物 — こちらは ID のビット一致度。
        全 ID との XOR+popcount は整数演算なので誤差ゼロ。
        """
        target = self._ids[self._id2idx[annotation_id_]]
        scored = [
            (i, id_cosine(target, aid, dim))
            for i, aid in enumerate(self._ids)
            if aid != target
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def neighbors(self, idx: int, k: int = 5) -> list[tuple[int, float]]:
        """アノテーション行 idx の近傍 (cosine 降順, 自身を除く) — 連結性の最小クエリ。"""
        import numpy as np

        M = self.embedding_matrix()
        sims = M @ M[idx]
        order = np.argsort(sims)[::-1]
        return [(int(j), float(sims[int(j)])) for j in order if int(j) != idx][:k]

    def query(
        self,
        text: str,
        k: int = 5,
        *,
        quantized: bool = False,
        exclude_questions: bool = False,
        role: str | None = None,
    ) -> list[tuple[int, float]]:
        """自由テキスト 1 件でストアを cosine 検索 (符号化 1 回; キャッシュには入れない)。

        返り値は (行, cosine) の降順 top-k。
        - ``quantized=True``: int8 近似経路 (大規模・省メモリ; 順位ほぼ不変、score 近似)。
        - ``exclude_questions=True``: 質問アノテーションを除外し**事実 (平叙文) のみ**返す
          (質問クエリが他の質問文ばかり拾う問題への対策)。
        - ``role``: 指定ロール (例 "user") の行のみに絞る。
        """
        import numpy as np

        q = _l2_normalize(np.asarray(self._encoder.encode_texts([text])[0], dtype=np.float32))
        if quantized:
            Q, scale = self.int8_matrix()
            qi = np.clip(np.round(q * 127.0), -127, 127).astype(np.int8)
            sims = (Q.astype(np.int32) @ qi.astype(np.int32)) * scale
        else:
            sims = self.embedding_matrix() @ q
        order = np.argsort(sims)[::-1]
        out: list[tuple[int, float]] = []
        for j in order:
            jj = int(j)
            if exclude_questions and self._is_q[jj]:
                continue
            if role is not None and self._roles[jj] != role:
                continue
            out.append((jj, float(sims[jj])))
            if len(out) >= k:
                break
        return out

    @property
    def n_cooccur_edges(self) -> int:
        """共起エッジ数 (連結性グラフの規模)。"""
        return len(self._cooc)

    def cooccur_neighbors(self, row: int, k: int = 5) -> list[tuple[int, int]]:
        """行の共起近傍 (会話隣接で一緒に出た行) を共起回数降順で返す。"""
        hits = [
            (b if a == row else a, c)
            for (a, b), c in self._cooc.items()
            if a == row or b == row
        ]
        hits.sort(key=lambda t: t[1], reverse=True)
        return hits[:k]

    def query_connected(
        self,
        text: str,
        k: int = 5,
        *,
        seed_k: int = 5,
        want_facts: bool = True,
        boost: float = 1.0,
    ) -> list[tuple[int, float, str]]:
        """連結性検索: cosine 事実検索を base に、**共起エッジで答え (事実) へホップ**して加点。

        cosine 単独では「質問」と「その答え」を繋げない (head-to-head で全 encoder R@1=0)。
        会話隣接の共起グラフを 1 ホップ展開して橋渡しする = 差別化の核。

        **単調改善設計**: cosine 事実検索の結果を base スコアとして必ず保持し、共起ホップは
        加点 (max マージ) のみ — よって cosine 単独を下回らない。
        返り値: (行, スコア, 由来) の降順。由来 = "cosine" | "cooccur"。
        """
        import numpy as np

        # base: cosine 事実検索 (全行を score 付き; want_facts なら質問除外)
        q = _l2_normalize(np.asarray(self._encoder.encode_texts([text])[0], dtype=np.float32))
        sims = self.embedding_matrix() @ q
        scored: dict[int, tuple[float, str]] = {}
        for row in range(self._n_rows):
            if want_facts and self._is_q[row]:
                continue
            scored[row] = (float(sims[row]), "cosine")

        # 共起ホップ: cosine 上位 seed (質問含む) の隣接事実を加点
        order = np.argsort(sims)[::-1][:seed_k]
        for srow in order:
            srow = int(srow)
            sim = float(sims[srow])
            neigh = self.cooccur_neighbors(srow, k=999)
            denom = max((c for _, c in neigh), default=1)  # seed の最大共起で正規化 (希釈しすぎない)
            for nb, c in neigh[:k]:
                if want_facts and self._is_q[nb]:
                    continue
                s = boost * sim * (c / denom)     # 0..boost*sim
                prev = scored.get(nb)
                if prev is None or s > prev[0]:
                    scored[nb] = (s, "cooccur")
        ranked = sorted(scored.items(), key=lambda t: t[1][0], reverse=True)
        return [(row, sc, src) for row, (sc, src) in ranked][:k]

    def role_of_row(self, row: int) -> str | None:
        """行の発話者ロール (初出時に記録)。"""
        return self._roles[row]

    def is_question_row(self, row: int) -> bool:
        """行が質問アノテーションか。"""
        return self._is_q[row]

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
            "roles": self._roles,
            "is_question": self._is_q,
            # 共起エッジ: "a,b" -> count (JSON キーは文字列)
            "cooccur": {f"{a},{b}": c for (a, b), c in self._cooc.items()},
            "n_instances": self._n_instances,
            "n_encoded": self._n_encoded,
        }
        self._path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(self._path.with_suffix(".npz"), embeddings=self.embedding_matrix())

    def _load(self, path: Path) -> None:
        import numpy as np

        meta = json.loads(path.read_text(encoding="utf-8"))
        emb = np.ascontiguousarray(np.load(path.with_suffix(".npz"))["embeddings"], dtype=np.float32)
        anns = meta["annotations"]
        if emb.shape[0] != len(anns):
            raise ValueError("annotation store is corrupt: embeddings/annotations mismatch")
        self._ann2idx = {a: i for i, a in enumerate(anns)}
        # 後方互換: 旧形式 (ids 無し) は文字列から再生成
        self._ids = [int(s) for s in meta["ids"]] if "ids" in meta else [
            annotation_id(a) for a in anns
        ]
        self._id2idx = {aid: i for i, aid in enumerate(self._ids)}
        self._matrix = emb
        self._n_rows = emb.shape[0]
        self._counts = list(meta["counts"])
        # 後方互換: roles/is_question 無し旧形式は再生成
        self._roles = list(meta["roles"]) if "roles" in meta else [None] * len(anns)
        self._is_q = (
            list(meta["is_question"]) if "is_question" in meta
            else [is_question(a) for a in anns]
        )
        self._cooc = {}
        for key, c in meta.get("cooccur", {}).items():
            a_s, b_s = key.split(",")
            self._cooc[(int(a_s), int(b_s))] = int(c)
        self._recent_group = []
        self._n_instances = int(meta["n_instances"])
        self._n_encoded = int(meta["n_encoded"])
        self._int8 = None
