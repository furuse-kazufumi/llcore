# SPDX-License-Identifier: Apache-2.0
"""RAD コーパス取込 PoC (M3.0) — 世界知識 md docs を AnnotationStore へ注入し効果と干渉を実測する。

M3 仮説: 既製の世界知識コーパス (RAD docs) を AnnotationStore に取り込めば、会話では
言及されていない世界知識質問にも grounding できる。本 PoC は loop_engineering_corpus
(39 md docs, 4 スコープ, 日英混在) を最初の取込対象として 3 条件を測る:

- (a) 会話のみ store × 世界知識 probe → floor 確認 (ほぼ全滅が期待値)
- (b) 会話+RAD store × 世界知識 probe → 注入の効果
- (c) 会話+RAD store × 既存 22 会話 probe → store 大規模化による会話 retrieval の干渉
  (M1 の MiniLM cosine MRR 0.947 が劣化しないか)

検索は cosine 経路 (store.query, exclude_questions=True) 固定。encoder = MiniLM 固定。

honest 留保:
- 世界知識 probe (WORLD_PROBES) は**測定前に事前登録し、測定後に変更しない**
  (cherry-pick 禁止 — 失敗 probe も結果に残す)。
- gold はキーワード包含 (verbatim が正)。corpus アノテーションに実在する語のみ gold とする。
- corpus は日英混在 (本文の大半は日本語 + 英語術語埋込)。MiniLM (all-MiniLM-L6-v2) は
  英語中心の encoder のため、日本語文中の英語術語への match は不利な条件 — 結果はこの
  混在条件込みの実測値であり、英語専用 corpus での上限ではない。
- markdown 除去は素朴 (frontmatter / コードフェンス / 見出し記号 / リンク / URL を rule-based
  で剥がすのみ)。除去漏れの記号断片がアノテーションに混入しうる。
- corpus は 1 分野 (loop engineering) のみ・probe は自作 18 問の小規模 PoC — 一般化は主張しない。

使い方::

    py -3.11 scripts/rad_ingest_poc.py [--out PATH] [--corpus DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for p in (str(_SRC), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, SentenceEncoderBackend  # noqa: E402

import connectivity_bench  # noqa: E402  (scripts/ 同居 — SOURCES/PROBES/ingest/rank_of/mrr を再利用)

_ensure_utf8_stdout()

DEFAULT_CORPUS = Path(r"D:\docs\loop_engineering_corpus_src")

# -- markdown の素朴除去 -------------------------------------------------------

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.M)
_BLOCKQUOTE = re.compile(r"^>\s?", re.M)
_EMPH = re.compile(r"(\*\*|\*|__|`)")
_TABLE_PIPE = re.compile(r"^\|.*\|\s*$", re.M)


def strip_markdown(text: str) -> str:
    """frontmatter / コードフェンス / 見出し記号 / リンク / URL 等を素朴に剥がす。

    完全な markdown parser ではない (PoC)。リンクはアンカーテキストを残し URL を捨てる。
    コードフェンスは中身ごと捨てる (数式・コード断片は短句アノテーションに不向き)。
    """
    t = _FRONTMATTER.sub("", text)
    t = _CODE_FENCE.sub(" ", t)
    t = _MD_IMAGE.sub(" ", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _URL.sub(" ", t)
    t = _TABLE_PIPE.sub(" ", t)
    t = _HEADING.sub("", t)
    t = _BULLET.sub("", t)
    t = _BLOCKQUOTE.sub("", t)
    t = _EMPH.sub("", t)
    return t


def ingest_corpus(store: AnnotationStore, corpus_root: Path) -> dict[str, object]:
    """corpus の md docs を doc 単位で取込む。group は doc ごとに採番 (10 刻み —
    adjacency_window=1 の共起が doc を跨いで張られないよう間隔を空ける)。"""
    docs = sorted(corpus_root.glob("**/*.md"))
    if not docs:
        raise FileNotFoundError(f"no md docs under {corpus_root}")
    t0 = time.perf_counter()
    per_doc: list[dict[str, object]] = []
    for di, doc in enumerate(docs):
        body = strip_markdown(doc.read_text(encoding="utf-8"))
        ids = store.add_text(body, source=str(doc), role="corpus", group=di * 10)
        per_doc.append({"doc": doc.relative_to(corpus_root).as_posix(), "n_annotations": len(ids)})
    elapsed = time.perf_counter() - t0
    return {
        "root": str(corpus_root),
        "n_docs": len(docs),
        "ingest_seconds": round(elapsed, 2),
        "per_doc": per_doc,
    }


# -- 世界知識 probe (事前登録 — 測定後に変更しない / cherry-pick 禁止) -----------
#
# 18 probe、4 スコープからまんべんなく (operational 4 / control 5 / learning 5 / agent 4)。
# gold は corpus アノテーション (正規化後・小文字) に verbatim 包含される語のみ。
WORLD_PROBES: list[tuple[str, list[str]]] = [
    # operational-ci-loops
    ("which deployment strategy shifts a small percentage of traffic to the new version",
     ["canary"]),
    ("what state does a gitops reconciliation loop continuously converge to",
     ["desired state"]),
    ("how does a supervisor detect a process that is alive but stuck",
     ["heartbeat", "watchdog"]),
    ("what hypothesis does a chaos engineering experiment try to disprove",
     ["steady state"]),
    # control-feedback-loops
    ("what error signal does pid control compute every cycle",
     ["setpoint"]),
    ("which autonomic computing loop runs monitor analyze plan execute over shared knowledge",
     ["mape-k"]),
    ("what are the four stages of boyd's decision loop",
     ["observe-orient-decide-act", "ooda"]),
    ("which circuit breaker state lets limited test requests through to probe recovery",
     ["half-open"]),
    ("which control method repeatedly optimizes over a receding horizon",
     ["receding horizon"]),
    # learning-loops
    ("what model does rlhf learn from human preference rankings",
     ["reward model", "bradley-terry"]),
    ("how did alphazero improve its policy through games against itself",
     ["self-play", "mcts"]),
    ("what problem in continual learning does elastic weight consolidation mitigate",
     ["catastrophic forgetting"]),
    ("how does active learning choose which examples to label next",
     ["uncertainty sampling"]),
    ("which operator makes value iteration converge to a unique fixed point",
     ["bellman"]),
    # autonomous-agent-loops
    ("where does reflexion store its verbal self-reflections across trials",
     ["episodic memory"]),
    ("which agent stores successful executable code in a searchable library",
     ["skill library", "voyager"]),
    ("how does tree of thoughts abandon a bad reasoning path",
     ["backtracking"]),
    ("which technique plans verification questions to reduce hallucination",
     ["verification questions", "hallucination"]),
]


# -- 測定 ----------------------------------------------------------------------


def run_probes(
    store: AnnotationStore, probes: list[tuple[str, list[str]]]
) -> dict[str, object]:
    """cosine 経路 (query, k=10, 事実のみ) で probe 集合を測り R@1/R@3/MRR を返す。

    計算は connectivity_bench と同一 (rank_of / mrr を import して使用)。
    """
    ann = store.annotations
    fetch: Callable[[str], list[str]] = lambda q: [  # noqa: E731  (bench と同形の検索クロージャ)
        ann[i] for i, _ in store.query(q, k=10, exclude_questions=True)
    ]
    ranks: list[int] = []
    per_probe: list[dict[str, object]] = []
    for q, gold in probes:
        r = connectivity_bench.rank_of(fetch(q), gold)
        ranks.append(r)
        per_probe.append({"query": q, "gold": gold, "rank": r})
    n = len(ranks)
    return {
        "n_probes": n,
        "R@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "R@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
        "MRR": round(connectivity_bench.mrr(ranks), 4),
        "per_probe": per_probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "rad_ingest_poc.json")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help="RAD corpus root (md docs)")
    args = parser.parse_args()

    encoder = SentenceEncoderBackend()  # M1 確定の MiniLM backend に固定

    # store A: 会話のみ (connectivity_bench と同じ ingest)
    store_conv = AnnotationStore(encoder)
    t0 = time.perf_counter()
    n_turns = connectivity_bench.ingest(store_conv)
    conv_seconds = time.perf_counter() - t0
    n_conv_ann = len(store_conv.annotations)
    print(f"[conv store] {n_turns} turns -> {n_conv_ann} annotations "
          f"({conv_seconds:.2f}s)", flush=True)

    # store B: 会話 + RAD corpus
    store_full = AnnotationStore(encoder)
    connectivity_bench.ingest(store_full)
    corpus_info = ingest_corpus(store_full, args.corpus)
    n_full_ann = len(store_full.annotations)
    stats_full = store_full.stats()
    print(f"[full store] +{corpus_info['n_docs']} docs -> {n_full_ann} annotations "
          f"(corpus ingest {corpus_info['ingest_seconds']}s, "
          f"encode_saved_ratio={stats_full['encode_saved_ratio']})", flush=True)

    # 3 条件 + 参考の in-run 会話 baseline (M1 正本と同条件の追試)
    cond_a = run_probes(store_conv, WORLD_PROBES)
    cond_b = run_probes(store_full, WORLD_PROBES)
    cond_c = run_probes(store_full, connectivity_bench.PROBES)
    cond_c0 = run_probes(store_conv, connectivity_bench.PROBES)  # 干渉比較の同一 run 内基準

    print(f"\n{'condition':34s}  R@1   R@3   MRR", flush=True)
    for name, c in (("a: conv-only / world probes", cond_a),
                    ("b: conv+RAD  / world probes", cond_b),
                    ("c: conv+RAD  / conv probes", cond_c),
                    ("c0: conv-only / conv probes", cond_c0)):
        print(f"{name:34s}  {c['R@1']:.3f} {c['R@3']:.3f} {c['MRR']:.4f}", flush=True)

    interference = float(cond_c0["MRR"]) - float(cond_c["MRR"])  # type: ignore[arg-type]
    results: dict[str, object] = {
        "encoder": "minilm (SentenceEncoderBackend, all-MiniLM-L6-v2)",
        "query_path": "store.query(q, k=10, exclude_questions=True)  # cosine",
        "stores": {
            "conv_only": {"n_turns": n_turns, "n_annotations": n_conv_ann,
                          "n_cooccur_edges": store_conv.n_cooccur_edges,
                          "stats": store_conv.stats()},
            "conv_plus_rad": {"n_annotations": n_full_ann,
                              "n_corpus_annotations_unique": n_full_ann - n_conv_ann,
                              "n_cooccur_edges": store_full.n_cooccur_edges,
                              "stats": stats_full},
        },
        "corpus": corpus_info,
        "conditions": {
            "a_conv_only_world_probes": cond_a,
            "b_conv_plus_rad_world_probes": cond_b,
            "c_conv_plus_rad_conv_probes": cond_c,
            "c0_conv_only_conv_probes_baseline": cond_c0,
        },
        "verdict": {
            "world_floor_MRR": cond_a["MRR"],
            "world_injected_MRR": cond_b["MRR"],
            "injection_helps": float(cond_b["MRR"]) > float(cond_a["MRR"]),  # type: ignore[arg-type]
            "conv_baseline_MRR": cond_c0["MRR"],
            "conv_after_injection_MRR": cond_c["MRR"],
            "conv_interference_MRR_drop": round(interference, 4),
            "conv_interference": interference > 0.0,
        },
        "honest_note": (
            "WORLD_PROBES は測定前に事前登録した 18 問 (測定後の変更なし・失敗 probe も保持)。"
            "gold=キーワード包含。corpus は日英混在 1 分野のみ — MiniLM は英語中心 encoder の"
            "ため日本語文中の英語術語 match には不利な条件での実測。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    v = results["verdict"]
    print(f"\nverdict: world {v['world_floor_MRR']} -> {v['world_injected_MRR']} "  # type: ignore[index]
          f"(injection_helps={v['injection_helps']}), "  # type: ignore[index]
          f"conv interference drop = {v['conv_interference_MRR_drop']}", flush=True)  # type: ignore[index]
    print(f"results: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
