# SPDX-License-Identifier: Apache-2.0
"""ANN (faiss HNSW) 経路の実測 — exact 総当たり cosine との recall / latency 比較。

ROADMAP M3 残課題「ANN 化 (10 万 annotations 級で総当たり cosine が限界)」の実装検証。
(iii) +800 と同一構成の 23k store で、事前登録済み 40 probe (loop 18 + 会話 22) の
exact / ann 両経路を比較する。

測定項目:
    - recall@10: ann の top-10 が exact の top-10 をどれだけ再現するか (probe 平均)
    - rank 一致率: top-10 の順序まで完全一致した probe の割合
    - latency: exact vs ann の query 平均 (ms) + index 構築時間
    - フィルタ併用: loop 18 probe を domain="loop" + ann=True で (over-fetch 経路の実測)

honest 留保: 23k 行の総当たりは数十 ms で済むため、この規模で ANN の速度メリットは
出ない可能性が高い (HNSW の本領は 10 万行超)。本測定の主目的は **正しさ (recall) の
実測開示**と、全量取込 (~48 分野) 前の配線確認。

使い方::

    py -3.11 scripts/rad_ann_check.py [--out PATH]
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
from llcore.clip import SentenceEncoderBackend  # noqa: E402

import connectivity_bench  # noqa: E402
from rad_domain_filter_check import build_domain_tagged_store  # noqa: E402
from rad_ingest_poc import WORLD_PROBES  # noqa: E402
from rad_role_filter_check import run_probes_filtered  # noqa: E402
from rad_scale_poc import _fmt  # noqa: E402

_ensure_utf8_stdout()


def compare_recall(store: Any, probes: list[tuple[str, list[str]]],
                   k: int = 10) -> dict[str, Any]:
    """各 probe で exact / ann の top-k 行集合を比較する。"""
    recalls: list[float] = []
    exact_lat: list[float] = []
    ann_lat: list[float] = []
    n_order_match = 0
    for q, _gold in probes:
        t0 = time.perf_counter()
        exact = store.query(q, k=k, exclude_questions=True)
        exact_lat.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        approx = store.query(q, k=k, ann=True, exclude_questions=True)
        ann_lat.append(time.perf_counter() - t0)
        e_rows = [r for r, _ in exact]
        a_rows = [r for r, _ in approx]
        recalls.append(len(set(e_rows) & set(a_rows)) / max(len(e_rows), 1))
        if e_rows == a_rows:
            n_order_match += 1
    return {
        "n_probes": len(probes),
        "recall_at_10_mean": round(statistics.mean(recalls), 4),
        "recall_at_10_min": round(min(recalls), 4),
        "order_exact_match_ratio": round(n_order_match / len(probes), 3),
        "exact_latency_mean_ms": round(statistics.mean(exact_lat) * 1000, 1),
        "ann_latency_mean_ms": round(statistics.mean(ann_lat) * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "rad_ann_check.json")
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=== ann check: rebuild 23k store ===", flush=True)
    store, info = build_domain_tagged_store(SentenceEncoderBackend())
    print(f"[store] {info['n_annotations']} annotations ({info['build_seconds']}s)",
          flush=True)

    t0 = time.perf_counter()
    store.ann_index()
    index_build_s = round(time.perf_counter() - t0, 2)
    print(f"[index] HNSW build {index_build_s}s", flush=True)

    all_probes = list(WORLD_PROBES) + list(connectivity_bench.PROBES)
    cmp = compare_recall(store, all_probes)
    print(f"[recall@10] mean {cmp['recall_at_10_mean']}  min {cmp['recall_at_10_min']}  "
          f"order-match {cmp['order_exact_match_ratio']}", flush=True)
    print(f"[latency] exact {cmp['exact_latency_mean_ms']}ms vs "
          f"ann {cmp['ann_latency_mean_ms']}ms", flush=True)

    # フィルタ併用 (over-fetch 経路): domain="loop" の MRR が exact 経路と一致するか
    loop_exact = run_probes_filtered(store, WORLD_PROBES, domain="loop")
    loop_ann = run_probes_filtered(store, WORLD_PROBES, domain="loop", ann=True)
    print(f"[loop exact domain=loop] {_fmt(loop_exact)}", flush=True)
    print(f"[loop ann   domain=loop] {_fmt(loop_ann)}", flush=True)
    mrr_match = abs(float(loop_exact["MRR"]) - float(loop_ann["MRR"])) < 5e-4

    results = {
        "store": {**info},
        "hnsw": {"M": 32, "efConstruction": 80, "index_build_seconds": index_build_s},
        "recall_comparison_40_probes": cmp,
        "loop_domain_exact": loop_exact,
        "loop_domain_ann": loop_ann,
        "loop_domain_mrr_match": mrr_match,
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[filter] domain=loop MRR exact==ann: {mrr_match}", flush=True)
    print(f"\ntotal {results['total_seconds']}s\nresults: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
