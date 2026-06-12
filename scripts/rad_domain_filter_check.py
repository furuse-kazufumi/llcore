# SPDX-License-Identifier: Apache-2.0
"""domain タグ (分野スコープ) の防衛効果の実証 — corpus 間食い合いの復元測定。

role フィルタの限界 (`rad_role_filter_check.py` / NEXT_SESSION 2026-06-12):
    loop / astro はどちらも role="corpus" のため、role では corpus 間の食い合いを
    防げない。(iii) +800 astro store で loop 18 probe は role="corpus" 絞り込みでも
    MRR 0.4895 (M3.0 単独取込時の 0.639 から押し下げられたまま)。

仮説 (設計 (b) per-row domain タグ):
    `query(domain="loop")` で検索スコープを分野単位に絞れば、loop 18 probe は
    astro 800 docs 混入下でも M3.0 単独取込時の成績 (R@1 0.611 / MRR 0.6389) に
    復元するはず。本スクリプトはこれを (iii) と同一規模の store で実測する。

4 条件:
    - world_nofilter: フィルタなし (床 = 最悪値の確認)
    - world_role_corpus: role="corpus" (role の限界の in-run 再現 ≈ 0.4895)
    - world_domain_loop: domain="loop" (本命 — 復元判定対象)
    - conv_exclude_corpus: 会話 22 probe exclude_roles={"corpus"} (回帰確認 = 0.947 維持)

honest 留保: domain="loop" は会話行 (domain=None) も除外する。M3.0 の 0.639 は
会話 97 annotations 込み store での測定だが、loop probe の gold は全て corpus 由来
なので比較は成立する (会話行が loop probe の rank を奪った事例は M3.0 に無い)。

使い方::

    py -3.11 scripts/rad_domain_filter_check.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
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
from rad_scale_poc import POC_CORPUS, _fmt, dry_scan  # noqa: E402
from rad_topic_overlap_poc import OVERLAP_CORPUS, nested_even_selection  # noqa: E402

_ensure_utf8_stdout()


def build_domain_tagged_store(encoder: object) -> tuple[AnnotationStore, dict[str, object]]:
    """(iii) +800 と同一構成の store を domain タグ付きで作る。

    会話 (domain=None) + loop corpus (domain="loop") + astro 800 docs (domain="astro")。
    """
    store = AnnotationStore(encoder)
    t0 = time.perf_counter()
    n_turns = connectivity_bench.ingest(store)
    loop_docs = sorted(POC_CORPUS.glob("**/*.md"))
    di = 0
    for doc in loop_docs:
        store.add_text(strip_markdown(doc.read_text(encoding="utf-8")),
                       source=str(doc), role="corpus", domain="loop", group=di * 10)
        di += 1
    scan = dry_scan(OVERLAP_CORPUS)
    sel800 = nested_even_selection(len(scan), [100, 400, 800])[2]
    for i in sel800:
        doc = scan[i][0]
        store.add_text(strip_markdown(doc.read_text(encoding="utf-8")),
                       source=str(doc), role="corpus", domain="astro", group=di * 10)
        di += 1
    info = {
        "n_turns": n_turns,
        "n_loop_docs": len(loop_docs),
        "n_astro_docs": len(sel800),
        "n_annotations": len(store.annotations),
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    return store, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "out" / "rad_domain_filter_check.json")
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=== domain filter check: rebuild (iii) +800 store with domain tags ===",
          flush=True)
    store, info = build_domain_tagged_store(SentenceEncoderBackend())
    print(f"[store] {info['n_annotations']} annotations "
          f"(loop {info['n_loop_docs']} docs + astro {info['n_astro_docs']} docs, "
          f"{info['build_seconds']}s)", flush=True)

    nofilter = run_probes_filtered(store, WORLD_PROBES)
    role_corpus = run_probes_filtered(store, WORLD_PROBES, role="corpus")
    domain_loop = run_probes_filtered(store, WORLD_PROBES, domain="loop")
    conv_excl = run_probes_filtered(store, connectivity_bench.PROBES,
                                    exclude_roles={"corpus"})
    print(f"[world nofilter     ] {_fmt(nofilter)}", flush=True)
    print(f"[world role=corpus  ] {_fmt(role_corpus)}", flush=True)
    print(f"[world domain=loop  ] {_fmt(domain_loop)}", flush=True)
    print(f"[conv  excl corpus  ] {_fmt(conv_excl)}", flush=True)

    # 復元判定: domain="loop" で M3.0 単独取込時 (astro 混入前) の値に戻ったか
    reference = {"R@1": 0.611, "R@3": 0.611, "MRR": 0.6389}
    restored = all(float(domain_loop[k]) >= v - 5e-4 for k, v in reference.items())
    conv_ok = abs(float(conv_excl["MRR"]) - 0.9470) < 5e-4
    print(f"[restore] world domain=loop >= pre-astro reference (0.639): {restored}",
          flush=True)
    print(f"[regress] conv exclude_roles still 0.947: {conv_ok}", flush=True)

    results = {
        "store": {**info,
                  "composition": "会話 + loop39 (domain=loop) + astro 800 (domain=astro)"},
        "reference_pre_astro_M3_0": reference,
        "world_nofilter": nofilter,
        "world_role_corpus": role_corpus,
        "world_domain_loop": domain_loop,
        "conv_exclude_corpus": conv_excl,
        "world_restored_to_reference": restored,
        "conv_regression_ok": conv_ok,
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\ntotal {results['total_seconds']}s\nresults: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
