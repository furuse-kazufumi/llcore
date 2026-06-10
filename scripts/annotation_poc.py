# SPDX-License-Identifier: Apache-2.0
"""アノテーション PoC — 実会話 transcript を分割/dedup/CLIP 符号化し、節約率と連結性を実測。

ユーザー設計 (2026-06-11)「テキスト→アノテーション分割 / ユニーク保持 / CLIP テキスト
エンコーダ=アノテーションエンコーダ → 計算量規模を抑える」の最初の実データ検証。

データ: 本セッションで生成済みの**実会話** (チャットスモーク / 検証付き会話デモ / 耐久試験
の verbatim transcript)。同じ定型句 (挨拶・名前確認 等) が複数会話に出るため、dedup の
節約が実測できる。

測る物:
  1. 計算節約: encode_saved_ratio = 1 - (符号化したユニーク数 / アノテーション延べ数)
  2. 連結性: 自由クエリの近傍 (例: "what is the user's name" → 名前系アノテーション)
  3. cross-modal 連結性: 合成図形画像 → 最近傍アノテーション (CLIP 同一空間の確認)

honest 留保: 節約率はデータの重複度に依存する (会話が多様なほど下がる)。連結性の
近傍は定性確認であり精度主張ではない。cross-modal は合成図形 (非自然画像) で弱い可能性。

使い方::

    py -3.11 scripts/annotation_poc.py [--model ID] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, ClipBackend  # noqa: E402

_ensure_utf8_stdout()

# 本セッションで生成済みの実会話 transcript (存在するものだけ使う)
TRANSCRIPT_SOURCES = [
    _ROOT / "out" / "chat_staged_smoke_results.json",
    _ROOT / "out" / "chat_staged_smoke_360m.json",
    _ROOT / "out" / "chat_smoke_repro_a.json",
    _ROOT / "out" / "chat_smoke_repro_b.json",
    _ROOT / "out" / "chat_endurance_results.json",
    _ROOT / "research" / "rllm_pivot" / "phase2_demo_verified_chat_results.json",
]

CONNECTIVITY_QUERIES = [
    "what is the user's name?",
    "a simple pasta recipe",
    "a planet in space",
    "greeting someone politely",
]

SHAPE_IMAGES = sorted((_ROOT / "out" / "clip_smoke_shapes").glob("*.png"))


def extract_texts(path: Path) -> list[str]:
    """各 transcript JSON 形式から user/assistant テキストを順序つきで取り出す。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for turn in d.get("turns", []):  # smoke / endurance 形式
        for key in ("prompt", "reply"):
            if isinstance(turn.get(key), str):
                out.append(turn[key])
    for turn in d.get("conversation", []):  # verified chat demo 形式
        for key in ("user", "assistant"):
            if isinstance(turn.get(key), str):
                out.append(turn[key])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--out", type=Path, default=_ROOT / "out" / "annotation_poc_results.json"
    )
    args = parser.parse_args()

    backend = ClipBackend(model_id=args.model)
    store = AnnotationStore(backend, path=_ROOT / "out" / "annotation_store.json")

    # 1. 実会話を全部流す
    t0 = time.time()
    n_texts = 0
    sources_used: list[str] = []
    for src in TRANSCRIPT_SOURCES:
        if not src.exists():
            continue
        texts = extract_texts(src)
        for text in texts:
            store.add_text(text, source=src.name)
        n_texts += len(texts)
        sources_used.append(src.name)
    ingest_seconds = time.time() - t0
    stats = store.stats()
    print(f"sources: {len(sources_used)} transcripts, {n_texts} texts", flush=True)
    print(f"annotations: unique={stats['unique_annotations']} / "
          f"instances={stats['total_instances']} / encoded={stats['encoder_calls_texts']}",
          flush=True)
    print(f"★ encode_saved_ratio = {stats['encode_saved_ratio']:.1%} "
          f"(ingest {ingest_seconds:.1f}s)", flush=True)

    # 2. 連結性クエリ (定性)
    ann = store.annotations
    connectivity: dict[str, list] = {}
    print("\n=== 連結性クエリ (近傍 top3) ===", flush=True)
    for q in CONNECTIVITY_QUERIES:
        hits = [(ann[i], round(s, 4)) for i, s in store.query(q, k=3)]
        connectivity[q] = hits
        print(f'  "{q}"', flush=True)
        for a, s in hits:
            print(f"     {s:+.4f}  {a[:70]}", flush=True)

    # 3. cross-modal: 図形画像 → 最近傍アノテーション (同一空間の確認)
    cross_modal: dict[str, list] = {}
    if SHAPE_IMAGES:
        import numpy as np

        M = store.embedding_matrix()
        I_ = backend.encode_images(SHAPE_IMAGES)
        print("\n=== cross-modal (画像 → アノテーション近傍 top3) ===", flush=True)
        for path, vec in zip(SHAPE_IMAGES, I_):
            sims = M @ vec
            order = np.argsort(sims)[::-1][:3]
            hits = [(ann[int(j)], round(float(sims[int(j)]), 4)) for j in order]
            cross_modal[path.name] = hits
            print(f"  {path.name}: {hits}", flush=True)

    store.save()
    results = {
        "model": backend.model_id,
        "sources": sources_used,
        "n_texts": n_texts,
        "stats": stats,
        "ingest_seconds": round(ingest_seconds, 1),
        "connectivity_queries": connectivity,
        "cross_modal": cross_modal,
        "note": (
            "節約率はデータ重複度依存。連結性は定性確認 (精度主張なし)。"
            "cross-modal は合成図形のため弱い可能性 (verbatim 記録)。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults: {args.out}", flush=True)
    print(f"store  : {_ROOT / 'out' / 'annotation_store.json'} (+ .npz)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
