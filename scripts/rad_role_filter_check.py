# SPDX-License-Identifier: Apache-2.0
"""role フィルタの防衛効果の実証 (M3 スコープ絞り込み) — (iii) の +800 store で再測する。

M3 検証 (iii) (`M3_TOPIC_OVERLAP_2026_06_12.md`) の構造的知見:
    トピック重複 corpus 注入で会話 probe の rank 1 を奪った annotation は全て role="corpus"
    だった。→ `AnnotationStore.query(exclude_roles={"corpus"})` で会話検索のスコープを
    絞れば、会話 22 probe は corpus 注入前の成績 (R@1 0.909 / MRR 0.947) に復元するはず。
    本スクリプトはこれを (iii) と同一の +800 docs store で実測する。

3 条件:
    - conv_nofilter: フィルタなし ((iii) の +800 と同条件 = in-run 再現)
    - conv_exclude_corpus: exclude_roles={"corpus"} (スコープ絞り込み)
    - world_only_corpus: 参考 — loop 18 probe を role="corpus" のみで (positive 絞り込みの併用例)

使い方::

    py -3.11 scripts/rad_role_filter_check.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, SentenceEncoderBackend  # noqa: E402

import connectivity_bench  # noqa: E402
from rad_ingest_poc import WORLD_PROBES, strip_markdown  # noqa: E402
from rad_scale_poc import _fmt, build_poc_store, dry_scan  # noqa: E402
from rad_topic_overlap_poc import OVERLAP_CORPUS, nested_even_selection  # noqa: E402

_ensure_utf8_stdout()


def run_probes_filtered(
    store: AnnotationStore,
    probes: list[tuple[str, list[str]]],
    **query_kwargs: Any,
) -> dict[str, Any]:
    """rad_scale_poc.run_probes_timed と同一計算だが query() のフィルタ引数を渡せる版。"""
    ann = store.annotations
    ranks: list[int] = []
    per_probe: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q, gold in probes:
        t0 = time.perf_counter()
        hits = [ann[i] for i, _ in store.query(q, k=10, exclude_questions=True,
                                               **query_kwargs)]
        latencies.append(time.perf_counter() - t0)
        r = connectivity_bench.rank_of(hits, gold)
        ranks.append(r)
        per_probe.append({"query": q, "gold": gold, "rank": r})
    n = len(ranks)
    return {
        "n_probes": n,
        "R@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "R@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
        "MRR": round(connectivity_bench.mrr(ranks), 4),
        "query_latency_mean_s": round(statistics.mean(latencies), 4),
        "query_latency_median_s": round(statistics.median(latencies), 4),
        "per_probe": per_probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "out" / "rad_role_filter_check.json")
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=== role filter check: rebuild (iii) +800 store ===", flush=True)
    scan = dry_scan(OVERLAP_CORPUS)
    sel800 = nested_even_selection(len(scan), [100, 400, 800])[2]
    store, base_info = build_poc_store(SentenceEncoderBackend())
    di = base_info["n_loop_docs"]
    for i in sel800:
        doc = scan[i][0]
        store.add_text(strip_markdown(doc.read_text(encoding="utf-8")),
                       source=str(doc), role="corpus", group=di * 10)
        di += 1
    print(f"[store] {len(store.annotations)} annotations", flush=True)

    nofilter = run_probes_filtered(store, connectivity_bench.PROBES)
    excl = run_probes_filtered(store, connectivity_bench.PROBES,
                               exclude_roles={"corpus"})
    world_corpus = run_probes_filtered(store, WORLD_PROBES, role="corpus")
    print(f"[conv  nofilter      ] {_fmt(nofilter)}", flush=True)
    print(f"[conv  exclude corpus] {_fmt(excl)}", flush=True)
    print(f"[world role=corpus   ] {_fmt(world_corpus)}", flush=True)

    # 復元判定: exclude_roles で corpus 注入前 (= M3.0 / M1 正本) の値に戻ったか
    reference = {"R@1": 0.909, "R@3": 1.000, "MRR": 0.9470}
    restored = all(abs(float(excl[k]) - v) < 5e-4 for k, v in reference.items())
    print(f"[restore] conv == pre-injection reference (0.947): {restored}", flush=True)

    results = {
        "store": {"n_annotations": len(store.annotations),
                  "composition": "会話 + loop39 + astrophysics 800 docs ((iii) と同一構成)"},
        "reference_pre_injection": reference,
        "conv_nofilter": nofilter,
        "conv_exclude_corpus": excl,
        "world_role_corpus": world_corpus,
        "conv_restored_to_reference": restored,
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\ntotal {results['total_seconds']}s\nresults: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
