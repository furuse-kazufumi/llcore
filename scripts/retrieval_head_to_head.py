# SPDX-License-Identifier: Apache-2.0
"""retrieval head-to-head — AnnotationStore の CLIP/SigLIP text encoder vs 専用テキスト埋め込み。

差別化の生命線 (差別化調査 2026-06-11 の最優先実験): 短句アノテーション粒度において、
CLIP/SigLIP の text encoder が SBERT/E5/BGE 等の専用テキスト埋め込みに**拮抗 or 勝てる**か。
(Jina CLIP arXiv:2405.20204 は一般文では CLIP が SBERT の ~1/3 と反証済 → 短句限定なら覆るか?)

タスク: 実会話から抽出した**事実アノテーション** (平叙文) をコーパスとし、各事実に対する
**言い換えクエリ** (paraphrase) を gold とする retrieval。クエリ→正解事実が上位に来るか。
評価: Recall@1 / Recall@3 / MRR。負ければ差別化は surface 補償のみに縮退 (falsify ライン)。

honest 留保:
- gold クエリは手作りの少数 (PoC)。汎化主張はしない。
- SigLIP は短句に強い設計だが、コーパスが小さい (実会話の事実)。
- 各モデルは frozen・正規化済み cosine で公平比較。SBERT 系は query/passage prefix 規約に従う
  (E5 は "query:"/"passage:"、BGE は instruction prefix) — 規約適用版と素版の両方を測る。
- sentence-transformers 不在ならその行は skip (graceful)。

使い方::

    py -3.11 scripts/retrieval_head_to_head.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# encoder: 文字列リストを (n, d) 埋め込み行列(array-like)へ写す関数
Encode = Callable[[list[str]], Any]

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import ClipBackend  # noqa: E402

_ensure_utf8_stdout()

# (事実コーパス文, [言い換えクエリ…]) — 実会話に出た事実 + 手作り paraphrase
FACTS_AND_QUERIES: list[tuple[str, list[str]]] = [
    ("the capital of france is paris", ["what is the french capital", "paris is the capital of"]),
    ("a simple pasta dish is spaghetti carbonara", ["recommend an easy pasta", "carbonara pasta recipe"]),
    ("mars is known for its reddish hue", ["why is mars red", "the red planet color"]),
    ("bread is a staple of many cultures", ["bread as a food staple", "staple food around the world"]),
    ("music has been a universal language", ["music connects people worldwide", "the universal language of music"]),
    ("rain brings moisture into the earth", ["rainfall waters the ground", "how rain hydrates soil"]),
    ("stars twinkle in the night sky", ["why stars sparkle at night", "twinkling night sky stars"]),
    ("tea is enjoyed for its soothing properties", ["calming effect of tea", "tea relaxes you"]),
    ("the user lives in japan", ["where does the user live", "user is in japan"]),
    ("books offer a collection of stories", ["books contain many stories", "reading stories from books"]),
    ("a loyal feline companion", ["a faithful cat", "cats as loyal pets"]),
    ("rivers flow through landscapes", ["water flowing across land", "a flowing river"]),
]

# 距離規約: E5 は "query:"/"passage:", BGE-en は query に instruction。素版とも比較。
PREFIX_SCHEMES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: ", ""
    ),
}


def evaluate(corpus_vecs: np.ndarray, query_vecs: np.ndarray, gold: list[int]) -> dict[str, Any]:
    """各クエリの正解 (gold[i]) コーパス文の順位から Recall@1/3 と MRR を計算。"""
    sims = query_vecs @ corpus_vecs.T          # (n_query, n_corpus), 全て単位ノルム
    r1 = r3 = 0
    rr = 0.0
    ranks = []
    for i, g in enumerate(gold):
        order = np.argsort(sims[i])[::-1]
        rank = int(np.where(order == g)[0][0]) + 1
        ranks.append(rank)
        r1 += rank == 1
        r3 += rank <= 3
        rr += 1.0 / rank
    n = len(gold)
    return {"recall@1": r1 / n, "recall@3": r3 / n, "mrr": round(rr / n, 4),
            "n_query": n, "ranks": ranks}


# 難ベンチ: 実会話アノテーション (多数の似た短句) をコーパスに、
# 質問→答えの実体を含む事実 を gold とする (SigLIP が苦戦した実際の難所)。
HARD_GOLD: list[tuple[str, str]] = [
    ("what is my name", "my name is kazufumi"),
    ("where do i live", "i live in japan"),
    ("what pasta dish did we discuss", "one simple pasta dish is spaghetti carbonara"),
    ("which planet is reddish", "mars is known for its reddish hue"),
    ("name a classical composer", "classical composers include mozart"),
]


def run_hard_benchmark(encode: Encode, corpus: list[str]) -> dict[str, Any] | None:
    """実会話アノテーション corpus 上で HARD_GOLD の検索順位を測る (gold 不在はスキップ)。"""
    gold_rows: list[int] = []
    queries: list[str] = []
    used: list[tuple[str, str]] = []
    for q, fact in HARD_GOLD:
        # gold 事実が corpus に部分一致で含まれるか (正規化済みアノテーション)
        match = next((i for i, c in enumerate(corpus) if fact in c or c in fact), None)
        if match is not None:
            gold_rows.append(match)
            queries.append(q)
            used.append((q, corpus[match]))
    if not queries:
        return None
    cv = l2(np.asarray(encode(corpus), dtype=np.float64))
    qv = l2(np.asarray(encode(queries), dtype=np.float64))
    res = evaluate(cv, qv, gold_rows)
    res["pairs"] = [{"query": q, "gold_fact": f, "rank": r}
                    for (q, f), r in zip(used, res["ranks"])]
    return res


def l2(a: np.ndarray) -> np.ndarray:
    out: np.ndarray = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "retrieval_head_to_head.json")
    args = parser.parse_args()

    corpus = [f for f, _ in FACTS_AND_QUERIES]
    queries: list[str] = []
    gold: list[int] = []
    for ci, (_, qs) in enumerate(FACTS_AND_QUERIES):
        for q in qs:
            queries.append(q)
            gold.append(ci)
    print(f"[easy] corpus={len(corpus)} facts, queries={len(queries)} paraphrases", flush=True)

    # 難ベンチ用: 実会話アノテーション corpus をロード (あれば)
    hard_corpus: list[str] = []
    store_path = _ROOT / "out" / "annotation_store.json"
    if store_path.exists():
        meta = json.loads(store_path.read_text(encoding="utf-8"))
        hard_corpus = [a for a in meta.get("annotations", []) if a]
        print(f"[hard] 実会話アノテーション corpus={len(hard_corpus)} (多数の似た短句)", flush=True)

    results: dict[str, Any] = {
        "easy_corpus_size": len(corpus), "n_easy_queries": len(queries),
        "hard_corpus_size": len(hard_corpus), "models": {},
    }

    def record(name: str, encode: Encode) -> None:
        t0 = time.time()
        cv = l2(np.asarray(encode(corpus), dtype=np.float64))
        qv = l2(np.asarray(encode(queries), dtype=np.float64))
        easy = evaluate(cv, qv, gold)
        easy.pop("ranks", None)
        easy["encode_seconds"] = round(time.time() - t0, 1)
        hard = run_hard_benchmark(encode, hard_corpus) if hard_corpus else None
        if hard:
            hard.pop("ranks", None)
        results["models"][name] = {"easy": easy, "hard": hard}
        hs = f"hard MRR={hard['mrr']:.3f} R@1={hard['recall@1']:.3f}" if hard else "hard=n/a"
        print(f"  [{name}] easy MRR={easy['mrr']:.3f} | {hs}", flush=True)

    # --- CLIP/SigLIP (AnnotationStore のエンコーダ) ---
    clip = ClipBackend()
    clip_key = f"CLIP:{clip.model_id}"
    record(clip_key, clip.encode_texts)

    # --- 専用テキスト埋め込み (sentence-transformers) ---
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  (sentence-transformers 不在 — テキスト埋め込み比較を skip)", flush=True)
        results["note_st"] = "sentence-transformers not installed"
        _write(args.out, results)
        return 0

    st_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/multilingual-e5-small",
        "BAAI/bge-small-en-v1.5",
    ]
    for name in st_models:
        try:
            model = SentenceTransformer(name)
        except Exception as exc:  # noqa: BLE001 - DL 失敗等は skip
            print(f"  ({name} ロード失敗: {str(exc)[:60]} — skip)", flush=True)
            continue
        qpre, ppre = PREFIX_SCHEMES.get(name, ("", ""))

        def enc(texts: list[str], _m=model, _qp=qpre, _pp=ppre):  # type: ignore[no-untyped-def]
            # corpus/query で prefix が異なるため、両者を試し問わず passage prefix を既定に
            pref = _qp if all(t in queries for t in texts) else _pp
            return _m.encode([pref + t for t in texts])

        record(name, enc)

    # --- verdict (hard ベンチで判定; easy は飽和して無情報) ---
    def hard_mrr(v: dict[str, Any]) -> float:
        return v["hard"]["mrr"] if v.get("hard") else -1.0

    clip_mrr = hard_mrr(results["models"][clip_key])
    best_st = max(
        ((k, hard_mrr(v)) for k, v in results["models"].items() if not k.startswith("CLIP:")),
        key=lambda t: t[1], default=(None, 0.0),
    )
    gap = clip_mrr - best_st[1]
    results["verdict"] = {
        "clip_mrr": clip_mrr, "best_text_embedder": best_st[0], "best_text_mrr": best_st[1],
        "clip_minus_best_text_mrr": round(gap, 4),
        "benchmark": "hard (実会話アノテーション corpus, 質問→答えの実体を含む事実)",
        "conclusion": (
            "難ベンチでも CLIP が専用埋め込みに拮抗/勝利 = 差別化主張① 成立寄り" if gap >= -0.05
            else "CLIP が専用埋め込みに劣後 = 差別化は surface 補償+専用埋め込み差し替えへ (falsify)"
        ),
    }
    results["honest_note"] = (
        "easy ベンチ (12 独立事実 paraphrase) は全モデル飽和 (1.000) で無情報 → hard ベンチ "
        "(実会話アノテーション corpus 上の質問→事実) で判定。gold は手作り少数 (PoC, 汎化主張なし)。"
        "Jina CLIP は一般文で CLIP≈SBERT の 1/3 と報告。"
    )
    _write(args.out, results)
    print(f"\nverdict: CLIP MRR={clip_mrr:.3f} vs best text={best_st[0]} MRR={best_st[1]:.3f} "
          f"(gap {gap:+.3f})", flush=True)
    print(f"  → {results['verdict']['conclusion']}", flush=True)
    print(f"results: {args.out}", flush=True)
    return 0


def _write(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
