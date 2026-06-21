# SPDX-License-Identifier: Apache-2.0
"""連結性検索ベンチ (M1.2) — hard probe を拡張し hub 抑制強度を確信を持って調整する。

優先3 PoC の 3 probe は過剰適合域だった (probe ごとに最適が入替)。本ベンチは実会話アノテーションに
根ざした **20+ の質問→答え probe** で cosine / connected(IDF on/off) を比較し、hub 抑制の効果を測る。

各 probe = (新規質問, 答えに含まれる gold キーワード)。gold はストアの実アノテーションに存在する事実。

honest 留保:
- gold はキーワード包含 (verbatim が正)。会話に実在する事実のみを対象。
- 小型 LLM 生成ゆえ事実誤り (kazuhiro 等) も混在 — 正答キーワードのみ gold とする。
- encoder は --encoder で選択 (clip=SigLIP / minilm=all-MiniLM-L6-v2)。
  head-to-head (out/retrieval_head_to_head.json) で MiniLM 優位の実測済。

M1 追補 (2026-06-12): entity-coref エッジ (cosine base + 加算マージン) を比較対象に追加。
22-probe 訂正の受入基準 = entity 変種が cosine MRR を下回らないこと (broad 非破壊)。

使い方::

    py -3.11 scripts/connectivity_bench.py [--out PATH] [--encoder clip|minilm]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, ClipBackend, SentenceEncoderBackend  # noqa: E402

_ensure_utf8_stdout()

SOURCES = [
    _ROOT / "out" / "chat_staged_smoke_results.json",
    _ROOT / "out" / "chat_endurance_results.json",
    _ROOT / "research" / "rllm_pivot" / "phase2_demo_verified_chat_results.json",
]

# 20+ probe: 新規質問 → 答えの gold キーワード (実会話アノテーションに実在する事実)
PROBES: list[tuple[str, list[str]]] = [
    ("what is my name", ["kazufumi"]),
    ("where do i live", ["japan", "tokyo"]),
    ("what pasta dish did we discuss", ["carbonara"]),
    ("which planet is reddish", ["mars"]),
    ("name a classical composer", ["beethoven"]),
    ("what is the capital of france", ["paris"]),
    ("what is the capital of germany", ["berlin"]),
    ("what is the capital of italy", ["rome"]),
    ("what is two plus two", ["four", "2 + 2 = 4"]),
    ("what is 10 plus 15", ["25"]),
    ("what is 25 doubled", ["50"]),
    ("who wrote to kill a mockingbird", ["harper lee"]),
    ("name a famous novel", ["mockingbird"]),
    ("name a sport with a ball", ["soccer"]),
    ("how many teams in a soccer match", ["two teams"]),
    ("how many planets in the solar system", ["eight"]),
    ("what is carbonara made with", ["guanciale", "eggs", "parmesan"]),
    ("when did beethoven live", ["classical period"]),
    ("describe rain", ["clouds", "moisture", "condense"]),
    ("what dish has tomato sauce", ["tomato", "margherita", "pasta"]),
    ("name an italian dish", ["carbonara", "margherita", "pizza"]),
    ("what is the reddish planet known for", ["reddish", "mars"]),
]


def ingest(store: AnnotationStore) -> int:
    base = 0
    n = 0
    for src in SOURCES:
        if not src.exists():
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        turns = d.get("turns") or d.get("conversation") or []
        for ti, turn in enumerate(turns):
            g = base + ti
            for key, role in (("prompt", "user"), ("reply", "assistant"),
                              ("user", "user"), ("assistant", "assistant")):
                if isinstance(turn.get(key), str):
                    store.add_text(turn[key], role=role, group=g)
            n += 1
        base += len(turns) + 100
    return n


def rank_of(hits_anns: list[str], expect: list[str]) -> int:
    for rank, a in enumerate(hits_anns, 1):
        if any(k in a.lower() for k in expect):
            return rank
    return 0


def mrr(ranks: list[int]) -> float:
    return sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "connectivity_bench.json")
    parser.add_argument("--encoder", choices=("clip", "minilm"), default="clip",
                        help="clip=SigLIP (cross-modal) / minilm=all-MiniLM-L6-v2 (text-only)")
    args = parser.parse_args()

    encoder = ClipBackend() if args.encoder == "clip" else SentenceEncoderBackend()
    store = AnnotationStore(encoder)
    n_turns = ingest(store)
    ann = store.annotations
    print(f"ingested {n_turns} turns, {len(ann)} annotations, {store.n_cooccur_edges} edges, "
          f"{len(PROBES)} probes", flush=True)

    methods: dict[str, Callable[[str], list[Any]]] = {
        "cosine": lambda q: [ann[i] for i, _ in store.query(q, k=10, exclude_questions=True)],
        "connected_noIDF": lambda q: [ann[r] for r, _, _ in
                                      store.query_connected(q, k=10, hub_suppression=False)],
        "connected_IDF": lambda q: [ann[r] for r, _, _ in
                                    store.query_connected(q, k=10, hub_suppression=True)],
        "entity": lambda q: [ann[r] for r, _, _ in
                             store.query_connected(q, k=10, entity_hop=True)],
    }
    results: dict[str, Any] = {"encoder": args.encoder,
                                  "n_turns": n_turns, "n_annotations": len(ann),
                                  "n_edges": store.n_cooccur_edges, "n_probes": len(PROBES),
                                  "methods": {}, "per_probe": []}
    ranks_by_method: dict[str, list[int]] = {m: [] for m in methods}
    for q, expect in PROBES:
        row: dict[str, Any] = {"query": q, "gold": expect}
        for m, fn in methods.items():
            r = rank_of(fn(q), expect)
            ranks_by_method[m].append(r)
            row[m] = r
        results["per_probe"].append(row)

    print(f"\n{'method':18s}  R@1   R@3   MRR", flush=True)
    for m, ranks in ranks_by_method.items():
        r1 = sum(1 for r in ranks if r == 1) / len(ranks)
        r3 = sum(1 for r in ranks if r and r <= 3) / len(ranks)
        mr = mrr(ranks)
        results["methods"][m] = {"R@1": round(r1, 3), "R@3": round(r3, 3), "MRR": round(mr, 4)}
        print(f"{m:18s}  {r1:.3f} {r3:.3f} {mr:.3f}", flush=True)

    cos = results["methods"]["cosine"]["MRR"]
    idf = results["methods"]["connected_IDF"]["MRR"]
    noidf = results["methods"]["connected_noIDF"]["MRR"]
    ent = results["methods"]["entity"]["MRR"]
    results["verdict"] = {
        "cosine_MRR": cos, "connected_noIDF_MRR": noidf, "connected_IDF_MRR": idf,
        "entity_MRR": ent,
        "connectivity_helps": idf > cos or noidf > cos,
        "idf_helps": idf > noidf,
        # 受入基準 (事前登録): entity は broad を壊さない (cosine 以上) こと
        "entity_non_harmful": ent >= cos,
        "entity_helps": ent > cos,
        "conclusion": (
            f"cosine {cos:.3f} / connected IDF {idf:.3f} / noIDF {noidf:.3f} / "
            f"entity {ent:.3f}。entity-coref は "
            f"{'改善' if ent > cos else ('非破壊 (同等)' if ent == cos else '★悪化 (受入基準 FAIL)')}。"
        ),
    }
    results["honest_note"] = (
        f"20+ probe で 3 probe の過剰適合を脱した評価。gold=キーワード包含。encoder={args.encoder}。"
        "entity マージンは entity_boost=0.1 固定 (probe での調整はしない — 過剰適合防止)。"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nverdict: {results['verdict']['conclusion']}", flush=True)
    print(f"results: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
