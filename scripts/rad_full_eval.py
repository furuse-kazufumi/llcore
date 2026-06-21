# SPDX-License-Identifier: Apache-2.0
"""RAD 全量 store の事前登録測定 — 世界知識 30 probe + 会話防衛 + 全量規模 ANN。

ROADMAP M3 残項目「世界知識注入が retrieval/grounding を改善するか実測」の本測定。
取込 (rad_full_ingest.py) が保存した store を **load して** 測る (encode なし) —
save/load roundtrip の大規模実証も兼ねる。

事前登録:
  - probe = rad_full_probes.FULL_PROBES (30 問, コミット済み) + WORLD_PROBES (loop 18)
    + connectivity_bench.PROBES (会話 22)。本スクリプトもコミットで固定。
  - 判定軸 (結果を見る前に固定):
    E1 世界知識の取得: FULL_PROBES 各分野で domain スコープ時 MRR > nofilter MRR
       (51 分野混載の埋もれをスコープが救う — (iii) の 5 分野一般化)。
       nofilter MRR 自体は埋もれの実測値として開示 (基準値なし、初測定)。
    E2 会話防衛: 会話 22 probe exclude_roles={"corpus"} で MRR 0.947 (M3 系列の
       基準値) を維持 (±5e-3)。100 万 ann 級でも role 防衛が成立するか。
    E3 ANN 支配項逆転: 70 probe で ann=True の latency < exact latency
       (M3_ANN_HNSW の予告「~10 万行超で逆転」の検証) + recall@10 開示。
  - honest 留保: FULL_PROBES の gold は agent による verbatim 検証済みだが、
    全量 store では**他分野 doc が同じキーワードを含む可能性** (overcount) と
    **取込時の文分割で gold が分断される可能性** (undercount) の両方がある。
    per-probe rank を JSON に保存し事後検証可能にする。

使い方::

    py -3.11 scripts/rad_full_eval.py [--store PATH] [--out PATH]
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
from rad_full_probes import FULL_PROBES  # noqa: E402
from rad_ingest_poc import WORLD_PROBES  # noqa: E402
from rad_role_filter_check import run_probes_filtered  # noqa: E402
from rad_scale_poc import _fmt, process_rss_mb  # noqa: E402

_ensure_utf8_stdout()


def compare_exact_vs_ann(
    store: Any, probes: list[tuple[str, list[str]]], k: int = 10
) -> dict[str, Any]:
    """exact / ann の top-k 再現率と latency (rad_full_ingest と同一計算)。"""
    recalls: list[float] = []
    e_lat: list[float] = []
    a_lat: list[float] = []
    n_order = 0
    for q, _gold in probes:
        t0 = time.perf_counter()
        exact = store.query(q, k=k, exclude_questions=True)
        e_lat.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        approx = store.query(q, k=k, ann=True, exclude_questions=True)
        a_lat.append(time.perf_counter() - t0)
        e_rows = [r for r, _ in exact]
        a_rows = [r for r, _ in approx]
        recalls.append(len(set(e_rows) & set(a_rows)) / max(len(e_rows), 1))
        n_order += int(e_rows == a_rows)
    return {
        "n_probes": len(probes),
        "recall_at_10_mean": round(statistics.mean(recalls), 4),
        "recall_at_10_min": round(min(recalls), 4),
        "order_exact_match_ratio": round(n_order / len(probes), 3),
        "exact_latency_mean_ms": round(statistics.mean(e_lat) * 1000, 1),
        "ann_latency_mean_ms": round(statistics.mean(a_lat) * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store", type=Path, default=_ROOT / "out" / "rad_full_store.json")
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "rad_full_eval.json")
    args = parser.parse_args()

    t_start = time.perf_counter()
    print(f"=== full eval: load {args.store} ===", flush=True)
    t0 = time.perf_counter()
    store = AnnotationStore(SentenceEncoderBackend(), path=args.store)
    load_s = round(time.perf_counter() - t0, 1)
    print(f"[load] {len(store.annotations)} 行 ({load_s}s, RSS {process_rss_mb()}MB)",
          flush=True)

    results: dict[str, Any] = {"n_annotations": len(store.annotations), "load_seconds": load_s,
                               "domains": {}}

    # E1: 世界知識 30 probe — nofilter vs domain スコープ
    for dom, probes in FULL_PROBES.items():
        nof = run_probes_filtered(store, probes)
        scoped = run_probes_filtered(store, probes, domain=dom)
        results["domains"][dom] = {"nofilter": nof, "domain_scoped": scoped}
        print(f"[{dom:20s}] nofilter {_fmt(nof)}", flush=True)
        print(f"[{dom:20s}] scoped   {_fmt(scoped)}", flush=True)

    # loop 18 probe (M3 系列の継続測定)
    loop_nof = run_probes_filtered(store, WORLD_PROBES)
    loop_dom = run_probes_filtered(store, WORLD_PROBES, domain="loop_engineering")
    results["loop_nofilter"] = loop_nof
    results["loop_domain_scoped"] = loop_dom
    print(f"[loop_engineering    ] nofilter {_fmt(loop_nof)}", flush=True)
    print(f"[loop_engineering    ] scoped   {_fmt(loop_dom)}", flush=True)

    # E2: 会話防衛
    conv_nof = run_probes_filtered(store, connectivity_bench.PROBES)
    conv_excl = run_probes_filtered(store, connectivity_bench.PROBES,
                                    exclude_roles={"corpus"})
    results["conv_nofilter"] = conv_nof
    results["conv_exclude_corpus"] = conv_excl
    print(f"[conv 22             ] nofilter {_fmt(conv_nof)}", flush=True)
    print(f"[conv 22             ] excl     {_fmt(conv_excl)}", flush=True)

    # E3: ANN (全量規模での支配項逆転の検証)
    t0 = time.perf_counter()
    store.ann_index()
    idx_s = round(time.perf_counter() - t0, 1)
    print(f"[index] HNSW build {idx_s}s", flush=True)
    all_probes = ([p for ps in FULL_PROBES.values() for p in ps]
                  + list(WORLD_PROBES) + list(connectivity_bench.PROBES))
    cmp = compare_exact_vs_ann(store, all_probes)
    results["hnsw_index_build_seconds"] = idx_s
    results["exact_vs_ann_70_probes"] = cmp
    print(f"[ann] recall@10 {cmp['recall_at_10_mean']} (min {cmp['recall_at_10_min']}) "
          f"| exact {cmp['exact_latency_mean_ms']}ms vs ann {cmp['ann_latency_mean_ms']}ms",
          flush=True)

    # 事前登録判定
    e1 = {d: float(v["domain_scoped"]["MRR"]) > float(v["nofilter"]["MRR"])
          for d, v in results["domains"].items()}
    e2 = abs(float(conv_excl["MRR"]) - 0.9470) < 5e-3
    e3 = (float(cmp["ann_latency_mean_ms"]) < float(cmp["exact_latency_mean_ms"]))
    results["verdict"] = {
        "E1_domain_scope_beats_nofilter": e1,
        "E2_conv_defense_0947_maintained": e2,
        "E3_ann_faster_than_exact": e3,
    }
    results["rss_mb_final"] = process_rss_mb()
    results["total_seconds"] = round(time.perf_counter() - t_start, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("\n=== 事前登録判定 ===", flush=True)
    for k, v in results["verdict"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"total {results['total_seconds']}s\nresults: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
