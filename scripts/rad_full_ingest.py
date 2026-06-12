# SPDX-License-Identifier: Apache-2.0
"""RAD 全量取込 (M3) — 49+2 corpus を AnnotationStore へ永続化し、全量規模で再測する。

ROADMAP M3「RAD コーパス全量取込 → 連結性グラフ = 世界知識グラフ」の実走。
これまでの系列 (PoC 39 docs → (i) 1,989 docs/60k ann → (iii) +800 docs/23k ann) の
最終段 = D:/docs 配下の全 corpus (~17.8k docs、~50 万 annotations 見込み)。

設計 (M3 系列の流儀を踏襲):
    - domain 規約 = dir 名から ``_corpus_v2`` / ``_corpus_src`` を機械的に除去
      (例 astrophysics_corpus_v2 → "astrophysics")。loop_engineering_corpus_src →
      "loop_engineering" — これまでの実験ローカル名 "loop" とは異なるが、49 分野では
      機械的規約の方が事故らない (probe 測定側で読み替え、ドキュメントに明記)。
    - role="corpus" / **group=None = 共起エッジを張らない**。doc 内全ペア共起は
      ~44.8k docs で ~2,000 万エッジ (dict 数 GB + save 膨張) になり、M1 実証では
      cooccur hop は MiniLM で効果ゼロ〜微害 — 現時点で張る実益がない (honest:
      「連結性グラフ = 世界知識グラフ」のエッジ層は将来の再設計課題として開示)。
    - 永続化 = AnnotationStore(path=...).save() → JSON (メタ) + npz (embeddings)。
      encode ~90 分 (実測 96 ann/s 級) は一度きり、以後は load で再利用する運用。
    - checkpoint: 前回 save から +50k annotations で中間 save + progress JSON 更新。
      クラッシュ時は --resume で「progress 上完了済みの corpus」をスキップして続行
      (部分取込 corpus は store ごと checkpoint 時点に戻っているため二重取込なし —
      progress と store save は常に同時更新)。
    - silent cap なし: cap 自体を設けない (全量)。メモリは RSS を corpus ごとに log。

測定 (取込後、全量 store で):
    - 会話 22 probe: nofilter vs exclude_roles={"corpus"} — 17.8k docs (corpus 圧 22 倍)
      での会話防衛が (iii) の結論どおり成立するか
    - loop 18 probe: nofilter vs domain="loop_engineering" — 分野スコープ復元
    - exact vs ann (HNSW): 40 probe recall@10 + latency — ~50 万行で支配項が
      encode → 総当たり matmul に逆転し ANN の速度メリットが初めて出るかの実測
      (23k では 19.6→16.9ms で「速くならない」が実測済 = M3_ANN_HNSW_2026_06_12.md)

honest 留保:
    - probe は事前登録済み 18+22 問のみ (変更・cherry-pick 禁止)。全量規模での
      gold キーワード衝突 (他分野 doc が verbatim を含む) は undercount/overcount
      どちらも起こりうる — per-probe を JSON に保存し事後検証可能にする。
    - 1 プロセス内で encode し続けるため、途中の torch/HF の状態劣化は未知。
      checkpoint 再開で結果が変わらないことは未検証 (再開時は同一 doc 順を維持)。

使い方::

    py -3.11 scripts/rad_full_ingest.py [--out PATH] [--store PATH] [--resume]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, SentenceEncoderBackend  # noqa: E402

import connectivity_bench  # noqa: E402
from rad_ingest_poc import WORLD_PROBES, strip_markdown  # noqa: E402
from rad_role_filter_check import run_probes_filtered  # noqa: E402
from rad_scale_poc import _fmt, process_rss_mb  # noqa: E402

_ensure_utf8_stdout()

DOCS_ROOT = Path(r"D:\docs")
# _corpus_v2 glob に乗らない取込対象 (M3 系列で使用してきた src 形式 corpus)
EXTRA_CORPORA = (
    DOCS_ROOT / "loop_engineering_corpus_src",
    DOCS_ROOT / "language_corpus_src",
)
# checkpoint は「行数」OR「経過時間」の早い方 (2026-06-12 教訓: 200k 行のみの閾値では
# セッション死亡時に 2 corpus 分 (~15 分) の encode を全損した — 時間軸も切る)
CHECKPOINT_EVERY_ANN = 100_000
CHECKPOINT_EVERY_SEC = 900.0


def corpus_domains() -> list[tuple[Path, str]]:
    """取込対象 (corpus root, domain 名) の一覧。dir 名から suffix を機械的に除去。"""
    pairs = [
        (r, r.name.removesuffix("_corpus_v2"))
        for r in sorted(DOCS_ROOT.glob("*_corpus_v2"))
        if r.is_dir()
    ]
    for r in EXTRA_CORPORA:
        if r.is_dir():
            pairs.append((r, r.name.removesuffix("_corpus_src")))
    if not pairs:
        raise FileNotFoundError(f"no corpus dirs under {DOCS_ROOT}")
    return pairs


def load_progress(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"done_domains": [], "per_corpus": []}


def checkpoint(store: AnnotationStore, progress: dict[str, object], path: Path) -> float:
    """store save + progress JSON を同時更新 (resume の整合性条件)。返り値 = save 秒。"""
    t0 = time.perf_counter()
    store.save()
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    return round(time.perf_counter() - t0, 1)


def ingest_full(
    store: AnnotationStore,
    pairs: list[tuple[Path, str]],
    progress: dict[str, object],
    progress_path: Path,
) -> dict[str, object]:
    """全 corpus を取込む。corpus 単位で進捗 log + RSS 記録、+50k ann ごとに checkpoint。"""
    done: set[str] = set(progress["done_domains"])  # type: ignore[arg-type]
    last_saved_rows = len(store.annotations)
    last_saved_t = time.perf_counter()
    t_all = time.perf_counter()
    for ci, (root, dom) in enumerate(pairs, 1):
        docs = sorted(root.glob("**/*.md"))
        if dom in done:
            print(f"[{ci}/{len(pairs)}] {dom}: skip (resume済 {len(docs)} docs)", flush=True)
            continue
        t0 = time.perf_counter()
        n_ann = 0
        for doc in docs:
            body = strip_markdown(doc.read_text(encoding="utf-8"))
            n_ann += len(store.add_text(body, source=str(doc), role="corpus",
                                        domain=dom))
        info = {
            "domain": dom,
            "docs": len(docs),
            "n_annotation_instances": n_ann,
            "n_store_rows": len(store.annotations),
            "ingest_seconds": round(time.perf_counter() - t0, 1),
            "rss_mb": process_rss_mb(),
        }
        progress["per_corpus"].append(info)  # type: ignore[union-attr]
        done.add(dom)
        progress["done_domains"] = sorted(done)
        print(f"[{ci}/{len(pairs)}] {dom}: {info['docs']} docs, +{n_ann} ann "
              f"(計 {info['n_store_rows']} 行, {info['ingest_seconds']}s, "
              f"RSS {info['rss_mb']}MB)", flush=True)
        grown = len(store.annotations) - last_saved_rows >= CHECKPOINT_EVERY_ANN
        stale = time.perf_counter() - last_saved_t >= CHECKPOINT_EVERY_SEC
        if grown or stale:
            s = checkpoint(store, progress, progress_path)
            last_saved_rows = len(store.annotations)
            last_saved_t = time.perf_counter()
            print(f"[checkpoint] {last_saved_rows} 行 save ({s}s, "
                  f"trigger={'rows' if grown else 'time'})", flush=True)
    s = checkpoint(store, progress, progress_path)
    print(f"[checkpoint] final {len(store.annotations)} 行 save ({s}s)", flush=True)
    return {
        "n_corpora": len(pairs),
        "docs_total": sum(int(c["docs"]) for c in progress["per_corpus"]),  # type: ignore[union-attr,call-overload]
        "n_store_rows": len(store.annotations),
        "ingest_seconds_total": round(time.perf_counter() - t_all, 1),
        "rss_mb_final": process_rss_mb(),
    }


def compare_exact_vs_ann(
    store: AnnotationStore, probes: list[tuple[str, list[str]]], k: int = 10
) -> dict[str, object]:
    """exact / ann の top-k 再現率と latency (rad_ann_check.compare_recall と同一計算)。"""
    recalls: list[float] = []
    exact_lat: list[float] = []
    ann_lat: list[float] = []
    n_order = 0
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
        n_order += int(e_rows == a_rows)
    return {
        "n_probes": len(probes),
        "recall_at_10_mean": round(statistics.mean(recalls), 4),
        "recall_at_10_min": round(min(recalls), 4),
        "order_exact_match_ratio": round(n_order / len(probes), 3),
        "exact_latency_mean_ms": round(statistics.mean(exact_lat) * 1000, 1),
        "ann_latency_mean_ms": round(statistics.mean(ann_lat) * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "rad_full_ingest.json")
    parser.add_argument("--store", type=Path,
                        default=_ROOT / "out" / "rad_full_store.json")
    parser.add_argument("--resume", action="store_true",
                        help="checkpoint (store + progress) から再開")
    args = parser.parse_args()
    progress_path = args.store.with_name(args.store.stem + "_progress.json")

    t_start = time.perf_counter()
    pairs = corpus_domains()
    print(f"=== full ingest: {len(pairs)} corpora from {DOCS_ROOT} ===", flush=True)

    if args.resume and args.store.exists():
        store = AnnotationStore(SentenceEncoderBackend(), path=args.store)
        progress = load_progress(progress_path)
        n_done = len(progress['done_domains'])  # type: ignore[arg-type]
        print(f"[resume] {len(store.annotations)} 行 / 完了 {n_done} corpus から再開",
              flush=True)
    else:
        store = AnnotationStore(SentenceEncoderBackend(), path=args.store)
        progress = {"done_domains": [], "per_corpus": []}
        n_conv = connectivity_bench.ingest(store)
        print(f"[base] 会話 {n_conv} turns → {len(store.annotations)} 行", flush=True)

    ingest_info = ingest_full(store, pairs, progress, progress_path)
    print(f"[ingest] {ingest_info['n_store_rows']} 行 "
          f"({ingest_info['ingest_seconds_total']}s, RSS {ingest_info['rss_mb_final']}MB)",
          flush=True)

    # --- 全量規模での再測 -------------------------------------------------------
    conv_nof = run_probes_filtered(store, connectivity_bench.PROBES)
    conv_excl = run_probes_filtered(store, connectivity_bench.PROBES,
                                    exclude_roles={"corpus"})
    loop_nof = run_probes_filtered(store, WORLD_PROBES)
    loop_dom = run_probes_filtered(store, WORLD_PROBES, domain="loop_engineering")
    print(f"[conv nofilter      ] {_fmt(conv_nof)}", flush=True)
    print(f"[conv excl corpus   ] {_fmt(conv_excl)}", flush=True)
    print(f"[loop nofilter      ] {_fmt(loop_nof)}", flush=True)
    print(f"[loop domain=loop_engineering] {_fmt(loop_dom)}", flush=True)

    t0 = time.perf_counter()
    store.ann_index()
    index_build_s = round(time.perf_counter() - t0, 1)
    print(f"[index] HNSW build {index_build_s}s", flush=True)
    cmp = compare_exact_vs_ann(store, list(WORLD_PROBES) + list(connectivity_bench.PROBES))
    print(f"[recall@10] mean {cmp['recall_at_10_mean']} min {cmp['recall_at_10_min']} "
          f"order-match {cmp['order_exact_match_ratio']}", flush=True)
    print(f"[latency] exact {cmp['exact_latency_mean_ms']}ms vs "
          f"ann {cmp['ann_latency_mean_ms']}ms", flush=True)

    store_files = {
        "store_json_mb": round(args.store.stat().st_size / 2**20, 1),
        "store_npz_mb": round(args.store.with_suffix(".npz").stat().st_size / 2**20, 1),
    }
    results = {
        "ingest": ingest_info,
        "per_corpus": progress["per_corpus"],
        "store_files": store_files,
        "conv_nofilter": conv_nof,
        "conv_exclude_corpus": conv_excl,
        "loop_nofilter": loop_nof,
        "loop_domain_scoped": loop_dom,
        "hnsw": {"M": 32, "efConstruction": 80, "index_build_seconds": index_build_s},
        "exact_vs_ann_40_probes": cmp,
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\ntotal {results['total_seconds']}s\nresults: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
