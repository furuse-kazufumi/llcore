# SPDX-License-Identifier: Apache-2.0
"""Conversation smoke for llcore's OWN native runtime (not an HF wrapper).

`llcore.chat` productizes a transformers-backed chat; this script instead drives a real
Qwen2.5-Instruct entirely through llcore's *own* forward (`src/llcore/runtime/qwen2.py`, golden-matched
to HF) with KV-cache greedy decoding. It is the honest demonstration of goal part (1) — "llcore が
まともに会話可能" — because the inference is llcore's code, not a `transformers` call. The conversational
*capability* comes from the pretrained Qwen weights; llcore's contribution is the verified on-prem
runtime that runs them (and, separately, the proxy-v2 memory↔quality research on top).

honest scope: 0.5B is the fast tier (good at general JP Q&A, weak at arithmetic — a model limit, not a
runtime bug); 1.5B is the "まともに会話" tier (`--model-dir D:/models/Qwen2.5-1.5B-Instruct`). Outputs
are printed verbatim — failures are kept, not hidden (feedback_benchmark_honest_disclosure).

    py -3.11 scripts/chat_native_qwen.py [--model-dir DIR] [--max-new N]
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from llcore.runtime.loader import load_qwen2

PROMPTS = [
    ("挨拶", "こんにちは。あなたは何ができますか？一文で答えてください。"),
    ("事実Q&A", "日本の首都はどこですか？地名だけ答えてください。"),
    ("文脈引継ぎ(1)", "私の名前はカズフミです。覚えておいてください。"),
    ("文脈引継ぎ(2)", "私の名前は何でしたか？"),
    ("算数(限界の正直記録)", "3 たす 5 はいくつですか？数字だけ答えてください。"),
]


@torch.no_grad()
def generate(model: object, tok: object, messages: list[dict[str, str]], max_new: int) -> str:
    """Greedy decode through llcore's native forward with a growing KV cache."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
    ids = tok(text, return_tensors="pt").input_ids  # type: ignore[operator]
    out = model(ids, return_cache=True)  # type: ignore[operator]
    logits, cache = out
    eos = getattr(tok, "eos_token_id", None)
    toks: list[int] = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax())
        if eos is not None and nxt == eos:
            break
        toks.append(nxt)
        logits, cache = model(torch.tensor([[nxt]]), past=cache, return_cache=True)  # type: ignore[operator]
    return str(tok.decode(toks, skip_special_tokens=True)).strip()  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="llcore native-runtime conversation smoke")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    model, tok, p = load_qwen2(args.model_dir)
    print(f"[load] {args.model_dir} via llcore native forward ({time.perf_counter() - t0:.1f}s)\n", flush=True)

    history: list[dict[str, str]] = [
        {"role": "system", "content": "あなたは親切な日本語アシスタントです。簡潔に答えてください。"}
    ]
    for label, prompt in PROMPTS:
        history.append({"role": "user", "content": prompt})
        ts = time.perf_counter()
        reply = generate(model, tok, history, args.max_new)
        history.append({"role": "assistant", "content": reply})
        print(f"[{label}] U: {prompt}", flush=True)
        print(f"          A: {reply}   ({time.perf_counter() - ts:.1f}s)\n", flush=True)
    print("[done] llcore native runtime held a multi-turn Japanese conversation.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
