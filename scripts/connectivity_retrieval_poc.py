# SPDX-License-Identifier: Apache-2.0
"""連結性検索 PoC — cosine が繋げない『質問→答え』を会話隣接の共起グラフで橋渡し。

優先3 (差別化の本命) の第一実証。head-to-head で全 encoder が hard R@1=0 だった
(似た短句 101 個から質問→答えを cosine 単独で繋げない) のに対し、**会話の turn 隣接を
共起エッジ**として持てば、過去の同型質問を seed に 1 ホップで答え (事実) へ到達できるか。

設計:
  1. 実会話 transcript を turn 番号つきで AnnotationStore に投入 (group=turn, 同一会話内で連番)。
     → 質問アノテーションとその答え事実が turn 隣接で共起エッジを得る。
  2. hard クエリ (新規質問) を (a) cosine 単独 query と (b) query_connected で検索し、
     答えの実体を含む事実の順位を比較。

honest 留保:
  - 共起は『会話で隣接した』だけの弱い信号 (因果/正解保証ではない)。
  - encoder は CLIP のまま (head-to-head で MiniLM がやや上だが R@1=0 は同じ — 本 PoC の主眼は
    encoder でなくグラフが cosine 単独を上回るか)。
  - gold 判定はキーワード包含 (verbatim が正)。

使い方::

    py -3.11 scripts/connectivity_retrieval_poc.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, ClipBackend  # noqa: E402

_ensure_utf8_stdout()

# 共起グラフを張るため turn 隣接が密な会話を使う (staged smoke / verified chat / endurance)
SOURCES = [
    _ROOT / "out" / "chat_staged_smoke_results.json",
    _ROOT / "out" / "chat_endurance_results.json",
    _ROOT / "research" / "rllm_pivot" / "phase2_demo_verified_chat_results.json",
]

# hard クエリ: 新規質問 → 答えの実体 (gold キーワード)
PROBES = [
    ("what is my name", ["kazufumi"]),
    ("where do i live", ["japan"]),
    ("what pasta dish", ["carbonara"]),
]


def ingest(store: AnnotationStore) -> int:
    """各会話を turn 番号つきで投入 (group=会話内 turn, 会話間は大ギャップで分離)。"""
    base = 0
    n = 0
    for src in SOURCES:
        if not src.exists():
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        turns = d.get("turns") or d.get("conversation") or []
        for ti, turn in enumerate(turns):
            g = base + ti
            if isinstance(turn.get("prompt"), str):
                store.add_text(turn["prompt"], role="user", group=g)
            if isinstance(turn.get("reply"), str):
                store.add_text(turn["reply"], role="assistant", group=g)
            if isinstance(turn.get("user"), str):
                store.add_text(turn["user"], role="user", group=g)
            if isinstance(turn.get("assistant"), str):
                store.add_text(turn["assistant"], role="assistant", group=g)
            n += 1
        base += len(turns) + 100  # 会話間ギャップ (跨り共起を防ぐ)
    return n


def rank_of_keyword(hits_anns: list[str], expect: list[str]) -> int:
    """gold キーワードを含む最初のヒットの順位 (1-based)。無ければ 0。"""
    for rank, a in enumerate(hits_anns, 1):
        if any(k in a.lower() for k in expect):
            return rank
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "connectivity_retrieval_poc.json")
    args = parser.parse_args()

    clip = ClipBackend()
    store = AnnotationStore(clip)
    n_turns = ingest(store)
    ann = store.annotations
    print(f"ingested {n_turns} turns, {len(ann)} unique annotations, "
          f"{store.n_cooccur_edges} co-occurrence edges", flush=True)

    def rr(rank: int) -> float:
        return 1.0 / rank if rank else 0.0

    results = []
    cos_hits = conn_hits = improved = 0
    cos_mrr = conn_mrr = 0.0
    for q, expect in PROBES:
        cos = [ann[i] for i, _ in store.query(q, k=10)]
        conn = [ann[r] for r, _, _ in store.query_connected(q, k=10, seed_k=5, want_facts=True)]
        r_cos = rank_of_keyword(cos, expect)
        r_conn = rank_of_keyword(conn, expect)
        cos_hits += int(r_cos == 1)
        conn_hits += int(r_conn == 1)
        cos_mrr += rr(r_cos)
        conn_mrr += rr(r_conn)
        # 改善 = connected の方が上位 (cosine 圏外 >10 を 0 扱いにすると不利なので rr で比較)
        improved += int(rr(r_conn) > rr(r_cos))
        print(f"\nQ: {q}  (gold {expect})", flush=True)
        print(f"  cosine   rank={r_cos or '>10'}: {cos[:4]}", flush=True)
        print(f"  connected rank={r_conn or '>10'}: {conn[:4]}", flush=True)
        results.append({"query": q, "expect": expect, "cosine_rank": r_cos,
                        "connected_rank": r_conn, "cosine_top": cos[:5], "connected_top": conn[:5]})

    n = len(PROBES)
    payload = {
        "n_turns": n_turns, "n_annotations": len(ann), "n_cooccur_edges": store.n_cooccur_edges,
        "summary": {
            "cosine_R@1": cos_hits, "connected_R@1": conn_hits, "n_probes": n,
            "cosine_MRR": round(cos_mrr / n, 4), "connected_MRR": round(conn_mrr / n, 4),
            "probes_improved_by_connectivity": improved,
        },
        "verdict": (
            f"連結性グラフが cosine 単独を上回る (MRR {cos_mrr/n:.3f}→{conn_mrr/n:.3f}, "
            f"{improved}/{n} probe で順位改善 — 共起ホップで答えへ到達)"
            if conn_mrr > cos_mrr else
            "連結性グラフは cosine を上回らず (共起信号不足/会話隣接が薄い)"
        ),
        "honest_note": (
            "R@1 は 0 でも rank/MRR は改善しうる (答えが top1 でなく上位へ)。"
            "共起は会話隣接のみの弱信号 + hub ノード (『new topic』等が全てに共起) が rank1 を占める "
            "→ 次は hub 抑制 (IDF) + entity coref エッジ。encoder=CLIP のまま。gold=キーワード包含。"
        ),
        "probes": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary: cosine R@1={cos_hits}/{len(PROBES)} → connected R@1={conn_hits}/{len(PROBES)}", flush=True)
    print(f"verdict: {payload['verdict']}", flush=True)
    print(f"results: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
