# SPDX-License-Identifier: Apache-2.0
"""耐久会話プローブ — 20 ターン × 複数話題転換での安定性確認 (CPU, on-prem)。

段階的会話スモーク (chat_staged_smoke.py) の長尺版。1 セッションで話題を
6 回転換しながら、途中で過去文脈への参照質問を混ぜ、以下を測る:

- 20 ターン完走するか (エラー/発散/空応答なし)
- 文脈参照 (名前・直前話題) が後半でも維持されるか
- ターン所要時間が後半で劣化しないか (履歴増加の影響)

honest: 参照キーワード判定はヒューリスティック。verbatim JSON が正。

使い方::

    py -3.11 scripts/chat_endurance_probe.py [--model ID] [--out PATH]
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
from llcore.chat.__main__ import _ensure_utf8_stdout  # noqa: E402

_ensure_utf8_stdout()

# (prompt, 文脈参照の期待キーワード list | None)
TURNS: list[tuple[str, list[str] | None]] = [
    ("Hello! My name is Kazufumi and I live in Japan.", None),
    ("What is the capital of France?", ["paris"]),
    ("And the capital of Germany?", ["berlin"]),
    ("Let's talk about cooking. Name one Italian dish.", None),
    ("Is that dish usually served hot or cold?", None),
    ("New topic: animals. Name one animal that can fly.", None),
    ("What is my name?", ["kazufumi"]),  # 7 ターン目で名前想起
    ("Switch to math. What is 10 + 15?", ["25", "twenty-five"]),
    ("And 25 doubled?", ["50", "fifty"]),
    ("Now music: name one classical composer.", None),
    ("In which century did that composer live, roughly?", None),
    ("New topic: weather. Describe rain in one sentence.", None),
    ("Where do I live?", ["japan"]),  # 13 ターン目で居住地想起
    ("Switch to sports. Name one sport played with a ball.", None),
    ("How many players per team in that sport, roughly?", None),
    ("Now books: name one famous novel.", None),
    ("Who wrote it?", None),
    ("Back to geography: what is the capital of Italy?", ["rome"]),
    ("What is my name again?", ["kazufumi"]),  # 19 ターン目で再想起
    ("Thank you! Say goodbye in one short sentence.", None),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "chat_endurance_results.json",
    )
    args = parser.parse_args()

    backend = TransformersBackend(model_id=args.model, seed=args.seed)
    session = ChatSession(
        backend, settings=GenerationSettings(max_new_tokens=args.max_new_tokens)
    )

    results: list[dict[str, object]] = []
    elapsed_times: list[float] = []  # typed parallel accumulator (results values are object)
    n_ref_ok = n_ref_ng = 0
    for i, (prompt, expect) in enumerate(TURNS, start=1):
        t0 = time.time()
        reply = session.ask(prompt)
        elapsed = time.time() - t0
        elapsed_times.append(elapsed)
        verdict = "open_ended"
        if expect is not None:
            low = reply.lower()
            verdict = "expected" if any(k in low for k in expect) else "unexpected"
            if verdict == "expected":
                n_ref_ok += 1
            else:
                n_ref_ng += 1
        print(f"[{i:2d}] you> {prompt}")
        print(f"     llcore> {reply[:100]}{'…' if len(reply) > 100 else ''}")
        print(f"     [{elapsed:.1f}s, {verdict}]", flush=True)
        results.append(
            {
                "turn": i,
                "prompt": prompt,
                "reply": reply,
                "elapsed_s": round(elapsed, 2),
                "check": verdict,
            }
        )

    first_half = elapsed_times[:10]
    second_half = elapsed_times[10:]
    payload = {
        "model": backend.model_id,
        "seed": args.seed,
        "turns_completed": len(results),
        "context_ref_ok": n_ref_ok,
        "context_ref_ng": n_ref_ng,
        "mean_elapsed_first_half_s": round(sum(first_half) / len(first_half), 2),
        "mean_elapsed_second_half_s": round(sum(second_half) / len(second_half), 2),
        "turns": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n20 ターン完走: 文脈参照 {n_ref_ok} ok / {n_ref_ng} ng, "
        f"前半平均 {payload['mean_elapsed_first_half_s']}s → "
        f"後半平均 {payload['mean_elapsed_second_half_s']}s"
    )
    print(f"results: {args.out}")
    return 0 if n_ref_ok > n_ref_ng else 1


if __name__ == "__main__":
    raise SystemExit(main())
