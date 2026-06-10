# SPDX-License-Identifier: Apache-2.0
"""アノテーション → 生成前段 PoC — ストア検索を frozen LLM の文脈注入に使う。

ユーザー着想 (2026-06-11)「テキストエンコーダーに渡す文面を、生成する前段として利用可能
かもしれない」の実データ検証。

仕組み:
  1. 過去の実会話から構築済みの AnnotationStore (ユニーク値 + CLIP 埋め込みキャッシュ) を開く
  2. **新規セッション** (履歴なし) の質問をストアで近傍検索
     — 計算コスト = 質問 1 件の符号化 + キャッシュ済み埋め込みとの内積のみ
  3. 上位アノテーションを system prompt に「過去のメモ」として注入し、frozen SmolLM2 が
     **セッションを跨いだ事実** (名前/居住地/話題) に答えられるかを baseline (注入なし) と対照

honest 留保:
- 成功はモデルの文脈追従力に依存する (360M は不完全)。失敗も verbatim 記録。
- 注入されるのは過去会話の生テキスト断片 — 取り違い (誤った近傍) はそのまま現れる。
- auto-check はキーワードヒューリスティック (verbatim が正)。

使い方::

    py -3.11 scripts/annotation_generation_poc.py [--out PATH]
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

from llcore.chat import ChatSession, GenerationSettings, TransformersBackend  # noqa: E402
from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402
from llcore.clip import AnnotationStore, ClipBackend  # noqa: E402

_ensure_utf8_stdout()

STORE_PATH = _ROOT / "out" / "annotation_store.json"

# (質問, 期待キーワード) — 期待される事実は過去会話 transcript にのみ存在する
PROBES = [
    ("What is my name?", ["kazufumi"]),
    ("Where do I live?", ["japan"]),
    ("What pasta dish did we talk about before?", ["carbonara", "tomato", "spaghetti"]),
]

SYSTEM_BASE = "You are a helpful, concise assistant. Answer the user's questions directly and briefly."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=4, help="注入する近傍アノテーション数")
    parser.add_argument(
        "--out", type=Path, default=_ROOT / "out" / "annotation_generation_poc_results.json"
    )
    args = parser.parse_args()

    if not STORE_PATH.exists():
        print(f"ERROR: {STORE_PATH} が無い (先に scripts/annotation_poc.py を実行)", flush=True)
        return 1

    clip = ClipBackend()
    store = AnnotationStore(clip, path=STORE_PATH)
    ann = store.annotations
    llm = TransformersBackend(seed=20260611)
    settings = GenerationSettings(max_new_tokens=64)

    results: list[dict[str, object]] = []
    n_base_ok = n_aug_ok = 0
    t0 = time.time()
    for question, expect in PROBES:
        # 検索 (生成前段): 質問 1 件の符号化 + 内積のみ。
        # ★事実抽出: exclude_questions=True で質問文を除外し平叙文 (事実候補) のみ拾う
        #   (質問クエリが他の質問文ばかり拾う honest 問題への対策)
        t_r = time.time()
        hits = [(ann[i], s) for i, s in store.query(question, k=args.k, exclude_questions=True)]
        retrieval_seconds = time.time() - t_r
        notes = "; ".join(a for a, _ in hits)

        # baseline: 注入なしの新規セッション
        base_session = ChatSession(llm, system_prompt=SYSTEM_BASE, settings=settings)
        base_reply = base_session.ask(question)

        # augmented: 過去メモを system に注入した新規セッション
        aug_system = (
            f"{SYSTEM_BASE} You have these notes from past conversations with this user: "
            f"{notes}. Use them when relevant."
        )
        aug_session = ChatSession(llm, system_prompt=aug_system, settings=settings)
        aug_reply = aug_session.ask(question)

        base_ok = any(k in base_reply.lower() for k in expect)
        aug_ok = any(k in aug_reply.lower() for k in expect)
        n_base_ok += int(base_ok)
        n_aug_ok += int(aug_ok)
        print(f"\nQ: {question}", flush=True)
        print(f"  retrieval ({retrieval_seconds*1000:.0f}ms): {[a[:40] for a, _ in hits]}", flush=True)
        print(f"  baseline : {base_reply[:90]}  [{'ok' if base_ok else 'ng'}]", flush=True)
        print(f"  augmented: {aug_reply[:90]}  [{'ok' if aug_ok else 'ng'}]", flush=True)
        results.append({
            "question": question, "expect": expect,
            "retrieved": [(a, round(s, 4)) for a, s in hits],
            "retrieval_seconds": round(retrieval_seconds, 3),
            "baseline_reply": base_reply, "baseline_ok": base_ok,
            "augmented_reply": aug_reply, "augmented_ok": aug_ok,
        })

    payload = {
        "store_stats": store.stats(),
        "chat_model": llm.model_id,
        "clip_model": clip.model_id,
        "k": args.k,
        "summary": {"baseline_ok": n_base_ok, "augmented_ok": n_aug_ok, "n_probes": len(PROBES)},
        "total_seconds": round(time.time() - t0, 1),
        "note": (
            "新規セッション (履歴なし) で過去会話の事実に答えられるか。"
            "検索コスト = 質問符号化 + キャッシュ埋め込み内積のみ。verbatim が正。"
        ),
        "probes": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary: baseline {n_base_ok}/{len(PROBES)} → augmented {n_aug_ok}/{len(PROBES)}", flush=True)
    print(f"results: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
