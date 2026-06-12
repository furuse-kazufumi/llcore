# SPDX-License-Identifier: Apache-2.0
"""会話トピック重複 corpus の干渉測定 (M3 検証 iii) — astrophysics corpus を段階注入する。

背景:
    M3.0 (39 docs) は会話干渉ゼロ、M3.1 検証 (i) (56 倍) では劣化が**トピック重複 probe に
    集中**した (loop probe が evolution/agents corpus に埋もれ、トピック非重複の会話 probe は
    軽微)。→ 仮説「埋もれの主因は store の絶対規模ではなくトピック重複」。本 PoC はこれを
    **会話 probe 側で直接検証**する: 会話の天文ターン (mars が reddish / eight planets) と
    トピックが重なる astrophysics corpus (arXiv abstracts, 3,757 md) を段階注入し、
    会話 22 probe — 特に天文 3 probe — が選択的に埋もれるかを実測する。

設計:
    - store ベース = 会話 + loop_engineering corpus (M3.0 と同構成)、encoder = MiniLM 固定
      (M3.1 検証 (ii) で続投確定)。
    - 注入段階 = 100 / 400 / 800 docs。サブサンプルは **nested 等間隔** (800 の選択列から
      400 を、400 から 100 を等間隔抽出) — incremental ingest で encode を重複させず、
      かつ各段階が deterministic (乱数なし)。
    - 各段階で会話 22 probe + loop 世界知識 18 probe (いずれも事前登録済み・変更禁止) を測定。
      loop probe は「トピック非重複側の対照群」: astrophysics は loop engineering とも
      重ならないので、仮説が正しければ loop probe は (i) のような大幅劣化を示さないはず。

honest 留保:
    - 天文 3 probe ("which planet is reddish" / "how many planets in the solar system" /
      "what is the reddish planet known for") は既存 PROBES のサブセット参照であり、
      probe の変更・追加ではない (注目表示用)。
    - gold はキーワード包含 (verbatim)。corpus 側 annotation が会話 gold 文字列
      (mars/eight 等) を偶然含めば「正解扱い」になりうる — per-probe の hit が会話由来か
      corpus 由来かは区別しない (M3.0/3.1 と同じ測定系を維持)。解釈で開示する。
    - 実行は foreground 推奨 (harness background は silent kill の既知問題)。

使い方::

    py -3.11 scripts/rad_topic_overlap_poc.py [--out PATH] [--stages 100,400,800]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for p in (str(_SRC), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import SentenceEncoderBackend  # noqa: E402

import connectivity_bench  # noqa: E402
from rad_ingest_poc import WORLD_PROBES, strip_markdown  # noqa: E402
from rad_scale_poc import (  # noqa: E402
    _even_indices,
    _fmt,
    build_poc_store,
    dry_scan,
    run_probes_timed,
    store_size,
)

_ensure_utf8_stdout()

OVERLAP_CORPUS = Path(r"D:\docs\astrophysics_corpus_v2")
DEFAULT_STAGES = (100, 400, 800)
# 会話 22 probe 中、astrophysics とトピックが重なる 3 問 (既存 PROBES のサブセット参照)
ASTRO_QUERIES: tuple[str, ...] = (
    "which planet is reddish",
    "how many planets in the solar system",
    "what is the reddish planet known for",
)


def nested_even_selection(n_docs: int, stages: list[int]) -> list[list[int]]:
    """各段階の doc index 列を nested に選ぶ (大きい段階の選択列から小さい段階を等間隔抽出)。

    返り値は stages と同順の選択列。stages[k] ⊂ stages[k+1] が保証されるため、
    incremental ingest (差分のみ追加) と整合する。
    """
    sels: list[list[int]] = []
    base = list(range(n_docs))
    for want in sorted(stages, reverse=True):
        base = [base[i] for i in _even_indices(len(base), want)]
        sels.append(list(base))
    return list(reversed(sels))


def rank_changes(
    base: dict[str, object], now: dict[str, object]
) -> list[dict[str, object]]:
    base_ranks = {p["query"]: p["rank"] for p in base["per_probe"]}  # type: ignore[index]
    return [
        {"query": p["query"], "rank_base": base_ranks[p["query"]], "rank_now": p["rank"]}
        for p in now["per_probe"]  # type: ignore[index]
        if p["rank"] != base_ranks[p["query"]]
    ]


def astro_ranks(conv: dict[str, object]) -> dict[str, int]:
    return {
        p["query"]: p["rank"]  # type: ignore[misc]
        for p in conv["per_probe"]  # type: ignore[index]
        if p["query"] in ASTRO_QUERIES
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "out" / "rad_topic_overlap_poc.json")
    parser.add_argument("--stages", type=str, default=",".join(map(str, DEFAULT_STAGES)),
                        help="注入段階の docs 数 (comma 区切り, 昇順)")
    args = parser.parse_args()
    stages = sorted(int(s) for s in args.stages.split(","))

    t_start = time.perf_counter()
    print(f"=== (iii) topic-overlap: dry-scan {OVERLAP_CORPUS.name} ===", flush=True)
    scan = dry_scan(OVERLAP_CORPUS)
    sels = nested_even_selection(len(scan), stages)
    for want, sel in zip(stages, sels):
        uniq = set()
        for i in sel:
            uniq |= scan[i][1]
        print(f"[plan] stage {want}: {len(sel)} docs, ~{len(uniq)} unique annotations",
              flush=True)

    encoder = SentenceEncoderBackend()  # MiniLM (M3.1 (ii) で続投確定)
    store, base_info = build_poc_store(encoder)
    base_conv = run_probes_timed(store, connectivity_bench.PROBES)
    base_world = run_probes_timed(store, WORLD_PROBES)
    print(f"[base ] conv 22: {_fmt(base_conv)}", flush=True)
    print(f"[base ] loop 18: {_fmt(base_world)}", flush=True)
    print(f"[base ] astro probes: {astro_ranks(base_conv)}", flush=True)

    results: dict[str, object] = {
        "overlap_corpus": str(OVERLAP_CORPUS),
        "corpus_docs_total": len(scan),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "query_path": "store.query(q, k=10, exclude_questions=True)  # cosine",
        "subsample": "nested 等間隔 (800→400→100, deterministic, 乱数なし) + incremental ingest",
        "astro_probe_queries": list(ASTRO_QUERIES),
        "base": {"build": base_info, "store": store_size(store),
                 "conv_probes": base_conv, "world_probes": base_world},
        "stages": [],
    }

    ingested: set[int] = set()
    di = base_info["n_loop_docs"]  # type: ignore[assignment]
    for want, sel in zip(stages, sels):
        new = [i for i in sel if i not in ingested]
        t0 = time.perf_counter()
        n_ann = 0
        for i in new:
            doc = scan[i][0]
            body = strip_markdown(doc.read_text(encoding="utf-8"))
            n_ann += len(store.add_text(body, source=str(doc), role="corpus",
                                        group=di * 10))
            di += 1
        ingested |= set(new)
        ingest_s = round(time.perf_counter() - t0, 2)
        size = store_size(store)
        conv = run_probes_timed(store, connectivity_bench.PROBES)
        world = run_probes_timed(store, WORLD_PROBES)
        stage_result = {
            "stage_docs": want,
            "docs_added": len(new),
            "annotation_instances_added": n_ann,
            "ingest_seconds": ingest_s,
            "store": size,
            "conv_probes": conv,
            "world_probes": world,
            "conv_rank_changes_vs_base": rank_changes(base_conv, conv),
            "world_rank_changes_vs_base": rank_changes(base_world, world),
            "astro_probe_ranks": astro_ranks(conv),
        }
        results["stages"].append(stage_result)  # type: ignore[union-attr]
        print(f"[{want:>4} docs] store {size['n_annotations']} ann "
              f"(+{len(new)} docs in {ingest_s}s)", flush=True)
        print(f"[{want:>4} docs] conv 22: {_fmt(conv)}", flush=True)
        print(f"[{want:>4} docs] loop 18: {_fmt(world)}", flush=True)
        print(f"[{want:>4} docs] astro probes: {stage_result['astro_probe_ranks']}",
              flush=True)

    results["total_seconds"] = round(time.perf_counter() - t_start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\ntotal {results['total_seconds']}s\nresults: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
