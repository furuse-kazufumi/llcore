# SPDX-License-Identifier: Apache-2.0
"""段階的会話スモークテスト — llcore.chat の実モデル検証 (CPU, on-prem)。

ユーザー指示「会話の内容を段階的に変えてみてください」(2026-06-10) に基づき、
**1 つの会話セッション内で内容を 4 段階に変えながら** 基本会話能力を測る:

  stage 1: 挨拶 / 自己紹介 (open-ended)
  stage 2: 単純事実 Q&A (正解が一意)
  stage 3: 文脈引継ぎ (前ターンで与えた情報の想起 = multi-turn 機能の本丸)
  stage 4: 話題転換 (cooking → space — 履歴が長くなっても脱線しないか)

honest 設計:
- auto-check はヒューリスティック (キーワード包含)。PASS/FAIL は参考値で、
  実出力 verbatim を JSON に全記録する方を正とする。
- 小型 instruct モデル (135M) は事実 Q&A を間違えることがある。失敗も削除せず
  記録する (feedback_benchmark_honest_disclosure)。

使い方::

    py -3.11 scripts/chat_staged_smoke.py [--model ID] [--seed N] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.chat import ChatSession, GenerationSettings, TransformersBackend  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (stage名, [(prompt, 期待キーワード list | None)…])
# 期待キーワード None = open-ended (auto-check 対象外、coherence は人間/上位 LLM が判断)
StageList = list[tuple[str, list[tuple[str, list[str] | None]]]]

STAGES_EN: StageList = [
    (
        "stage1_greeting",
        [
            ("Hello! Who are you?", None),
        ],
    ),
    (
        "stage2_simple_qa",
        [
            ("What is the capital of France?", ["paris"]),
            ("What is 2 + 2?", ["4", "four"]),
        ],
    ),
    (
        "stage3_context_carryover",
        [
            ("My name is Kazufumi. Please remember my name.", None),
            ("What is my name?", ["kazufumi"]),
        ],
    ),
    (
        "stage4_topic_shift",
        [
            (
                "Let's change the topic to cooking. "
                "Suggest one simple pasta dish in one sentence.",
                ["pasta", "spaghetti", "penne", "carbonara", "aglio"],
            ),
            (
                "New topic: space. Name one planet in our solar system.",
                [
                    "mercury", "venus", "earth", "mars",
                    "jupiter", "saturn", "uranus", "neptune",
                ],
            ),
        ],
    ),
]


def check(reply: str, expect: list[str] | None) -> str:
    """ヒューリスティック判定: expected | unexpected | open_ended。"""
    if expect is None:
        return "open_ended"
    low = reply.lower()
    return "expected" if any(k in low for k in expect) else "unexpected"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None, help="HF モデル ID (default: SmolLM2-135M-Instruct)")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "chat_staged_smoke_results.json",
    )
    args = parser.parse_args()

    backend = TransformersBackend(model_id=args.model, seed=args.seed)
    settings = GenerationSettings(max_new_tokens=args.max_new_tokens)
    session = ChatSession(backend, settings=settings)

    results: list[dict[str, object]] = []
    n_expected = n_unexpected = 0
    t_start = time.time()
    for stage_name, turns in STAGES:
        print(f"\n=== {stage_name} ===", flush=True)
        for prompt, expect in turns:
            print(f"you> {prompt}", flush=True)
            t0 = time.time()
            reply = session.ask(prompt)
            elapsed = time.time() - t0
            verdict = check(reply, expect)
            if verdict == "expected":
                n_expected += 1
            elif verdict == "unexpected":
                n_unexpected += 1
            print(f"llcore> {reply}", flush=True)
            print(f"  [{elapsed:.1f}s, auto-check: {verdict}]", flush=True)
            results.append(
                {
                    "stage": stage_name,
                    "prompt": prompt,
                    "reply": reply,
                    "elapsed_s": round(elapsed, 2),
                    "auto_check": verdict,
                    "expected_keywords": expect,
                }
            )

    payload = {
        "model": backend.model_id,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": backend.load_seconds,
        "total_seconds": round(time.time() - t_start, 1),
        "auto_check_summary": {
            "expected": n_expected,
            "unexpected": n_unexpected,
            "open_ended": sum(1 for r in results if r["auto_check"] == "open_ended"),
        },
        "note": (
            "auto-check はキーワード包含ヒューリスティック。実出力 verbatim が正。"
            "段階的会話 = 1 セッション内で 挨拶→Q&A→文脈引継ぎ→話題転換。"
        ),
        "turns": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary: expected={n_expected} unexpected={n_unexpected} "
          f"(load {backend.load_seconds:.1f}s, total {payload['total_seconds']}s)")
    print(f"results: {args.out}")
    # 終了コード: 文脈引継ぎ (stage3) と単純 Q&A (stage2) が全滅なら 1 (基本会話不成立)
    critical = [r for r in results if r["expected_keywords"] is not None]
    return 0 if any(r["auto_check"] == "expected" for r in critical) else 1


if __name__ == "__main__":
    raise SystemExit(main())
