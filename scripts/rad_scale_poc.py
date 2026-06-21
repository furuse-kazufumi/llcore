# SPDX-License-Identifier: Apache-2.0
"""RAD 大規模化 + 多言語 encoder PoC (M3.1) — 取込 PoC (M3.0) の次の検証 2 点を実測する。

検証 (i) 大規模化での会話干渉再測:
    M3.0 は 39 docs (+974 annotations, 計 1,071) で会話干渉ゼロだった。本 PoC は 3 分野の
    corpus (language_corpus_src 64 md / evolution_corpus_v2 764 md / agents_corpus_v2 1170 md)
    を追加し **annotations 上限 60,000** まで取り込んで再測する:
    (a) 既存 22 会話 probe (connectivity_bench.PROBES) — baseline MRR 0.947 からの劣化
    (b) 既存世界知識 18 probe (rad_ingest_poc.WORLD_PROBES) — loop_engineering 由来の答えが
        他分野 docs に埋もれて劣化するか
    上限超過時は doc 単位の均等サブサンプル (silent cap 禁止 — 取込/除外数を必ず log)。
    メモリ (embedding 行列 + RSS) と query レイテンシ (store.query は全行 matmul) も記録。

検証 (ii) 多言語 encoder head-to-head:
    M3.0 の失敗 5 probe は日英混在 gold の undercount が主因 (post-hoc 分析)。会話 +
    loop_engineering store (PoC と同構成) を 3 構成で作り再測する:
    - minilm (all-MiniLM-L6-v2, M3.0 と同一 = in-run 再現)
    - e5_noprefix (intfloat/multilingual-e5-small, prefix なし)
    - e5_prefix (同, E5 規約 "passage: "/"query: " prefix — retrieval_head_to_head の
      PREFIX_SCHEMES と同じ規約。prefix あり/なし両方を測って開示する)

honest 留保:
- probe は M3.0 / M1 で事前登録済みの 18+22 問をそのまま使う (変更・cherry-pick 禁止)。
- gold はキーワード包含 (verbatim)。日英混在 corpus では undercount しうる (M3.0 で開示済)。
- サブサンプルは deterministic (corpus ごとに等間隔の doc index) — 乱数なし・再現可能。
- query レイテンシは store.query 1 回 (query 文の encode + 全行 matmul) の wall-clock。
  encoder の encode 時間を含む実測値であり matmul 単体ではない。
- RSS は Windows API (GetProcessMemoryInfo) の WorkingSetSize — 取得失敗時は None (fail-soft)。

使い方::

    py -3.11 scripts/rad_scale_poc.py [--out PATH] [--cap N] [--skip-scale] [--skip-multilingual]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for p in (str(_SRC), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, SentenceEncoderBackend  # noqa: E402
from llcore.clip.annotations import split_annotations  # noqa: E402
from llcore.runtime.rss import working_set_mb as _working_set_mb  # noqa: E402

import connectivity_bench  # noqa: E402  (scripts/ 同居 — PROBES/ingest/rank_of/mrr を再利用)
from rad_ingest_poc import WORLD_PROBES, strip_markdown  # noqa: E402  (M3.0 の流儀を再利用)

_ensure_utf8_stdout()

POC_CORPUS = Path(r"D:\docs\loop_engineering_corpus_src")
SCALE_CORPORA: list[Path] = [
    Path(r"D:\docs\language_corpus_src"),     # 言語学 (64 md, 日本語主体 + frontmatter)
    Path(r"D:\docs\evolution_corpus_v2"),     # 進化計算 papers (764 md, 英語)
    Path(r"D:\docs\agents_corpus_v2"),        # agent papers/SKILL (1170 md, 英語)
]
ANNOTATION_CAP = 60_000  # store 全体 (会話 + loop + scale corpora) の上限
E5_MODEL = "intfloat/multilingual-e5-small"
# E5 の prefix 規約 (retrieval_head_to_head.PREFIX_SCHEMES と同一値)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# M3.0 (out/rad_ingest_poc.json) で rank=0 (圏外) だった 5 probe — 改善追跡用の固定リスト。
# probe 自体は WORLD_PROBES のまま (変更しない)。これは表示用の参照に過ぎない。
POC_FAILED_QUERIES: tuple[str, ...] = (
    "what error signal does pid control compute every cycle",
    "which control method repeatedly optimizes over a receding horizon",
    "what model does rlhf learn from human preference rankings",
    "which agent stores successful executable code in a searchable library",
    "how does tree of thoughts abandon a bad reasoning path",
)


def process_rss_mb() -> float | None:
    """現プロセスの WorkingSetSize (MB)。失敗時は None (fail-soft)。

    実体は単一情報源 :func:`llcore.runtime.rss.working_set_mb`(他ハーネスと同じ WinAPI 呼び出し)。
    """
    return _working_set_mb()


# -- 多言語 encoder 用 prefix wrapper ------------------------------------------


class PrefixedEncoder:
    """E5 系の "query: "/"passage: " 規約を AnnotationStore の encoder protocol 上で実現する。

    AnnotationStore は ingest (add_text) も検索 (query) も同じ ``encode_texts`` を呼ぶため、
    モード切替 (``mode`` 属性) で prefix を出し分ける。prefix は encoder 内部でのみ付与され、
    store に保持される正規化アノテーション文字列は不変 (dedup キーも不変)。
    """

    def __init__(self, inner: SentenceEncoderBackend,
                 query_prefix: str, passage_prefix: str) -> None:
        self._inner = inner
        self._qp = query_prefix
        self._pp = passage_prefix
        self.mode: str = "passage"  # "passage" (ingest) / "query" (検索)
        self.model_id = inner.model_id

    def encode_texts(self, texts: Sequence[str]) -> Any:
        pre = self._qp if self.mode == "query" else self._pp
        return self._inner.encode_texts([pre + t for t in texts])


# -- corpus dry-scan + 均等サブサンプル ----------------------------------------


def dry_scan(corpus_root: Path) -> list[tuple[Path, set[str]]]:
    """corpus の md docs を読み、doc ごとのユニークアノテーション集合を返す (encode なし)。"""
    docs = sorted(corpus_root.glob("**/*.md"))
    if not docs:
        raise FileNotFoundError(f"no md docs under {corpus_root}")
    return [
        (d, set(split_annotations(strip_markdown(d.read_text(encoding="utf-8")))))
        for d in docs
    ]


def _even_indices(n: int, keep: int) -> list[int]:
    """n docs から keep 件を等間隔に選ぶ deterministic な index 列 (乱数なし)。"""
    if keep >= n:
        return list(range(n))
    if keep <= 0:
        return []
    return sorted({int(i * n / keep) for i in range(keep)})


def plan_subsample(
    scans: list[list[tuple[Path, set[str]]]],
    base_unique: set[str],
    cap: int,
) -> tuple[list[list[int]], int]:
    """store 全体のユニーク annotations が cap 以下に収まる最大の均等サブサンプルを求める。

    keep 比率 r を二分探索し、corpus ごとに等間隔で round(n*r) docs を選んだときの
    ユニーク合計 (base = 会話 + loop_engineering 込) を厳密にシミュレートする。
    返り値: (corpus ごとの選択 doc index 列, シミュレート上のユニーク合計)。
    """

    def union_size(ratio: float) -> tuple[int, list[list[int]]]:
        seen = set(base_unique)
        selected: list[list[int]] = []
        for scan in scans:
            idx = _even_indices(len(scan), round(len(scan) * ratio))
            selected.append(idx)
            for i in idx:
                seen |= scan[i][1]
        return len(seen), selected

    full, sel_full = union_size(1.0)
    if full <= cap:
        return sel_full, full
    lo, hi = 0.0, 1.0  # lo = 可行, hi = 不可行 (full > cap)
    best_sel: list[list[int]] = [[] for _ in scans]
    best_n = len(base_unique)
    for _ in range(20):  # 1/2^20 << 1/1998 doc — doc 粒度では十分収束
        mid = (lo + hi) / 2
        n, sel = union_size(mid)
        if n <= cap:
            lo, best_sel, best_n = mid, sel, n
        else:
            hi = mid
    return best_sel, best_n


def ingest_scale_corpora(
    store: AnnotationStore,
    scans: list[list[tuple[Path, set[str]]]],
    selected: list[list[int]],
    roots: list[Path],
    group_start: int,
) -> dict[str, Any]:
    """選択済み docs を取込む。group は全 corpus 通しの doc 連番 × 10 (M3.0 と同じ 10 刻み —
    adjacency_window=1 の共起が doc を跨がない)。取込/除外数を corpus ごとに必ず log する。"""
    di = group_start
    per_corpus: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for root, scan, idx in zip(roots, scans, selected):
        t0 = time.perf_counter()
        n_ann = 0
        for i in idx:
            doc = scan[i][0]
            body = strip_markdown(doc.read_text(encoding="utf-8"))
            n_ann += len(store.add_text(body, source=str(doc), role="corpus", group=di * 10))
            di += 1
        info = {
            "root": str(root),
            "docs_total": len(scan),
            "docs_ingested": len(idx),
            "docs_excluded_by_cap": len(scan) - len(idx),
            "n_annotation_instances": n_ann,
            "ingest_seconds": round(time.perf_counter() - t0, 2),
        }
        per_corpus.append(info)
        print(f"[scale ingest] {root.name}: {info['docs_ingested']}/{info['docs_total']} docs "
              f"(excluded {info['docs_excluded_by_cap']}) in {info['ingest_seconds']}s",
              flush=True)
    return {
        "per_corpus": per_corpus,
        "docs_total": sum(len(s) for s in scans),
        "docs_ingested": sum(len(i) for i in selected),
        "ingest_seconds_total": round(time.perf_counter() - t_all, 2),
    }


# -- 測定 (rad_ingest_poc.run_probes と同一計算 + per-query レイテンシ) ---------


def run_probes_timed(
    store: AnnotationStore, probes: list[tuple[str, list[str]]]
) -> dict[str, Any]:
    """cosine 経路 (query, k=10, 事実のみ) で R@1/R@3/MRR + query レイテンシを測る。

    rank 計算は connectivity_bench.rank_of / mrr (M3.0 と同一)。レイテンシは store.query
    1 回 (query encode + 全行 matmul + top-k) の wall-clock 秒。
    """
    ann = store.annotations
    ranks: list[int] = []
    per_probe: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q, gold in probes:
        t0 = time.perf_counter()
        hits = [ann[i] for i, _ in store.query(q, k=10, exclude_questions=True)]
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


def build_poc_store(encoder: Any) -> tuple[AnnotationStore, dict[str, Any]]:
    """M3.0 と同構成の store (会話 + loop_engineering corpus) を作る。"""
    store = AnnotationStore(encoder)
    t0 = time.perf_counter()
    n_turns = connectivity_bench.ingest(store)
    docs = sorted(POC_CORPUS.glob("**/*.md"))
    for di, doc in enumerate(docs):
        body = strip_markdown(doc.read_text(encoding="utf-8"))
        store.add_text(body, source=str(doc), role="corpus", group=di * 10)
    info = {
        "n_turns": n_turns,
        "n_loop_docs": len(docs),
        "n_annotations": len(store.annotations),
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    return store, info


def store_size(store: AnnotationStore) -> dict[str, Any]:
    """store 規模 (annotations / 共起エッジ / 埋め込み行列メモリ / プロセス RSS)。"""
    mat = store.embedding_matrix()
    return {
        "n_annotations": len(store.annotations),
        "n_cooccur_edges": store.n_cooccur_edges,
        "embedding_matrix_shape": list(mat.shape),
        "embedding_matrix_mb": round(mat.nbytes / (1024 * 1024), 1),
        "process_rss_mb": process_rss_mb(),
        "stats": store.stats(),
    }


def _fmt(c: dict[str, Any]) -> str:
    return (f"R@1 {c['R@1']:.3f}  R@3 {c['R@3']:.3f}  MRR {c['MRR']:.4f}  "
            f"lat {c['query_latency_mean_s']*1000:.1f}ms")


# -- 検証 (i): 大規模化 ---------------------------------------------------------


def verify_scale(minilm_poc_store: AnnotationStore,
                 poc_world: dict[str, Any], poc_conv: dict[str, Any],
                 cap: int) -> dict[str, Any]:
    """conv + loop + 3 corpora (cap=60k) の大規模 store で会話干渉と世界知識劣化を測る。"""
    print(f"\n=== (i) scale: corpora dry-scan (cap={cap}) ===", flush=True)
    scans = [dry_scan(root) for root in SCALE_CORPORA]
    base_unique = set(minilm_poc_store.annotations)  # 会話 + loop の既存ユニーク
    selected, simulated_total = plan_subsample(scans, base_unique, cap)
    for root, scan, idx in zip(SCALE_CORPORA, scans, selected):
        print(f"[plan] {root.name}: keep {len(idx)}/{len(scan)} docs", flush=True)
    print(f"[plan] simulated unique annotations = {simulated_total} (cap {cap})", flush=True)

    encoder = SentenceEncoderBackend()  # MiniLM 固定 (M1/M3.0 と同一)
    store, poc_info = build_poc_store(encoder)
    scale_info = ingest_scale_corpora(store, scans, selected, SCALE_CORPORA,
                                      group_start=poc_info["n_loop_docs"])
    size = store_size(store)
    assert size["n_annotations"] == simulated_total, "dry-scan シミュレートと実取込が不一致"
    print(f"[big store] {size['n_annotations']} annotations, "
          f"{size['n_cooccur_edges']} edges, matrix {size['embedding_matrix_mb']}MB, "
          f"RSS {size['process_rss_mb']}MB", flush=True)

    conv = run_probes_timed(store, connectivity_bench.PROBES)
    world = run_probes_timed(store, WORLD_PROBES)
    print(f"[big store] conv 22 probes : {_fmt(conv)}", flush=True)
    print(f"[big store] world 18 probes: {_fmt(world)}", flush=True)

    # per-probe の劣化点 (in-run baseline = 会話+loop store との差分)
    base_conv_ranks = {p["query"]: p["rank"] for p in poc_conv["per_probe"]}
    base_world_ranks = {p["query"]: p["rank"] for p in poc_world["per_probe"]}
    conv_changed = [
        {"query": p["query"], "rank_poc": base_conv_ranks[p["query"]], "rank_scale": p["rank"]}
        for p in conv["per_probe"] if p["rank"] != base_conv_ranks[p["query"]]
    ]
    world_changed = [
        {"query": p["query"], "rank_poc": base_world_ranks[p["query"]], "rank_scale": p["rank"]}
        for p in world["per_probe"] if p["rank"] != base_world_ranks[p["query"]]
    ]
    return {
        "cap": cap,
        "subsample": {
            "method": "doc 単位の等間隔サブサンプル (corpus ごと, deterministic, 乱数なし)",
            "simulated_unique_total": simulated_total,
        },
        "poc_base_build": poc_info,
        "scale_ingest": scale_info,
        "store": size,
        "conv_probes": conv,
        "world_probes": world,
        "conv_rank_changes_vs_in_run_poc_store": conv_changed,
        "world_rank_changes_vs_in_run_poc_store": world_changed,
    }


# -- 検証 (ii): 多言語 encoder head-to-head -------------------------------------


def verify_multilingual() -> dict[str, Any]:
    """会話 + loop store (M3.0 構成) を 3 encoder 構成で作り、世界知識 18 + 会話 22 を測る。"""
    print("\n=== (ii) multilingual head-to-head (conv + loop store) ===", flush=True)
    configs: dict[str, Any] = {}
    minilm_store: AnnotationStore | None = None

    def run_config(name: str, encoder: Any, query_mode: bool = False) -> AnnotationStore:
        store, info = build_poc_store(encoder)
        if query_mode:
            encoder.mode = "query"  # 以降の store.query は query prefix で符号化
        world = run_probes_timed(store, WORLD_PROBES)
        conv = run_probes_timed(store, connectivity_bench.PROBES)
        configs[name] = {"build": info, "store": store_size(store),
                         "world_probes": world, "conv_probes": conv}
        print(f"[{name}] world: {_fmt(world)}", flush=True)
        print(f"[{name}] conv : {_fmt(conv)}", flush=True)
        return store

    minilm_store = run_config("minilm", SentenceEncoderBackend())
    run_config("e5_noprefix", SentenceEncoderBackend(E5_MODEL))
    run_config(
        "e5_prefix",
        PrefixedEncoder(SentenceEncoderBackend(E5_MODEL), E5_QUERY_PREFIX, E5_PASSAGE_PREFIX),
        query_mode=True,
    )

    # M3.0 失敗 5 問の per-probe 改善有無 (3 構成横並び)
    failed5 = []
    for q in POC_FAILED_QUERIES:
        row: dict[str, Any] = {"query": q}
        for name, cfg in configs.items():
            rank = next(p["rank"] for p in cfg["world_probes"]["per_probe"]
                        if p["query"] == q)
            row[name] = rank
        failed5.append(row)
    return {
        "store_composition": "会話 (connectivity_bench.SOURCES) + loop_engineering corpus "
                             "(M3.0 と同構成)",
        "e5_prefix_scheme": {"query": E5_QUERY_PREFIX, "passage": E5_PASSAGE_PREFIX},
        "configs": configs,
        "poc_failed_5_probes": failed5,
        "_minilm_store": minilm_store,  # (i) の in-run baseline に再利用 (JSON 化前に除去)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "out" / "rad_scale_poc.json")
    parser.add_argument("--cap", type=int, default=ANNOTATION_CAP)
    parser.add_argument("--skip-scale", action="store_true", help="検証 (i) を省略")
    parser.add_argument("--skip-multilingual", action="store_true", help="検証 (ii) を省略")
    args = parser.parse_args()

    t_start = time.perf_counter()
    results: dict[str, Any] = {
        "poc_reference": {
            "source": "out/rad_ingest_poc.json (M3.0 正本)",
            "world_MRR": 0.6389, "conv_MRR": 0.9470, "n_annotations": 1071,
        },
        "encoders": {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "e5": E5_MODEL},
        "query_path": "store.query(q, k=10, exclude_questions=True)  # cosine",
    }

    multilingual = None if args.skip_multilingual else verify_multilingual()
    if multilingual is not None:
        minilm_store = multilingual.pop("_minilm_store")
        results["ii_multilingual"] = multilingual
    else:
        minilm_store, _ = build_poc_store(SentenceEncoderBackend())

    if not args.skip_scale:
        minilm_cfg = (multilingual or {}).get("configs", {}).get("minilm")
        if minilm_cfg is None:  # --skip-multilingual 時は in-run baseline をここで測る
            minilm_cfg = {"world_probes": run_probes_timed(minilm_store, WORLD_PROBES),
                          "conv_probes": run_probes_timed(minilm_store, connectivity_bench.PROBES)}
        results["i_scale"] = verify_scale(
            minilm_store, minilm_cfg["world_probes"], minilm_cfg["conv_probes"], args.cap)

    results["total_seconds"] = round(time.perf_counter() - t_start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ntotal {results['total_seconds']}s\nresults: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
