# SPDX-License-Identifier: Apache-2.0
"""一番正直な証明 — 同じ Qwen 重みを HF transformers と llcore 自前 forward で動かし出力一致を実演。

記事 #43 の主張「llcore は HF の薄いラッパーでなく、自分で書いた推論コード (runtime/qwen2.py) で
事前学習済み重みを動かしている」を、誰でも再現できる形で証明する。

手順 (同一条件):
  1. 同じローカル Qwen ディレクトリの重みを (a) HF AutoModelForCausalLM (b) llcore load_qwen2 で読む。
  2. 同じ system+user メッセージを同じ chat template でプロンプト化。
  3. 同じ greedy デコード (do_sample=False, repetition_penalty=1.0, 同じ max_new) で生成。
  4. 出力トークン列が一致するか / 最初のステップの logits 最大差を表示。

honest: greedy (決定論) でのみ完全一致を主張する。サンプリングは乱数経路が別なので一致を狙わない。
capability (賢さ) は Qwen の重み由来で本スクリプトが作るものではない — 証明するのは「同じ計算を
している」ことだけ。出力は verbatim、不一致も隠さない (feedback_benchmark_honest_disclosure)。

    py -3.11 scripts/prove_native_matches_hf.py [--model-dir DIR] [--max-new N]
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch

from llcore.runtime.loader import load_qwen2

SYSTEM = "あなたは簡潔に答える日本語アシスタントです。"
PROMPTS = [
    "日本の首都はどこ？一言で。",
    "3 + 5 = ?",
    "水の化学式は？",
    "富士山の高さは？数字を添えて。",
    "Pythonでhello worldを出力する一行は？",
]


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _messages(user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def hf_greedy(model: Any, tok: Any, user: str, max_new: int) -> list[int]:
    """HF 純正 greedy デコード。生成された新規トークン id 列を返す。"""
    inp = tok.apply_chat_template(
        _messages(user), add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0
        )
    return [int(t) for t in out[0, inp["input_ids"].shape[1] :].tolist()]


def native_greedy(model: Any, tok: Any, user: str, max_new: int) -> tuple[list[int], Any]:
    """llcore 自前 forward greedy デコード。新規トークン id 列と最初の step の logits を返す。"""
    text = tok.apply_chat_template(_messages(user), tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    eos = getattr(tok, "eos_token_id", None)
    toks: list[int] = []
    first_logits: Any = None
    with torch.no_grad():
        logits, cache = model(ids, return_cache=True)
        first_logits = logits[0, -1].float().clone()
        for _ in range(max_new):
            nxt = int(logits[0, -1].argmax().item())
            if eos is not None and nxt == eos:
                break
            toks.append(nxt)
            logits, cache = model(torch.tensor([[nxt]]), past=cache, return_cache=True)
    return toks, first_logits


def hf_first_logits(model: Any, tok: Any, user: str) -> Any:
    """HF のプロンプト末尾 (次トークン) logits — native と数値比較するため。"""
    inp = tok.apply_chat_template(
        _messages(user), add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float()


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="prove llcore native forward == HF transformers")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct",
                    help="ローカル Qwen ディレクトリ (config.json + safetensors + tokenizer)")
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("一番正直な証明: 同じ Qwen 重み × {HF transformers, llcore 自前 forward} → 出力一致")
    print(f"  model: {args.model_dir}  (greedy, max_new={args.max_new})")
    print("=" * 70)

    t0 = time.perf_counter()
    hf_tok = AutoTokenizer.from_pretrained(args.model_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    print(f"[load] HF transformers       ({time.perf_counter() - t0:.1f}s)", flush=True)
    t0 = time.perf_counter()
    nat_model, nat_tok, _ = load_qwen2(args.model_dir)
    print(f"[load] llcore native forward ({time.perf_counter() - t0:.1f}s)\n", flush=True)

    n_match = 0
    max_logit_diff = 0.0
    for user in PROMPTS:
        hf_toks = hf_greedy(hf_model, hf_tok, user, args.max_new)
        nat_toks, nat_first = native_greedy(nat_model, nat_tok, user, args.max_new)
        hf_first = hf_first_logits(hf_model, hf_tok, user)
        diff = float((hf_first - nat_first).abs().max().item())
        max_logit_diff = max(max_logit_diff, diff)
        same = hf_toks == nat_toks
        n_match += int(same)
        hf_text = hf_tok.decode(hf_toks, skip_special_tokens=True).strip()
        nat_text = nat_tok.decode(nat_toks, skip_special_tokens=True).strip()
        print(f"Q: {user}")
        print(f"   HF     : {hf_text!r}")
        print(f"   native : {nat_text!r}")
        print(f"   → tokens {'一致 ✓' if same else '不一致 ✗'} "
              f"(len HF={len(hf_toks)} native={len(nat_toks)}), "
              f"first-step logits |Δ|max={diff:.2e}\n", flush=True)

    print("=" * 70)
    print(f"結果: {n_match}/{len(PROMPTS)} プロンプトで出力トークン完全一致 / "
          f"logits 最大差 {max_logit_diff:.2e}")
    if n_match == len(PROMPTS):
        print("→ llcore 自前 forward は HF transformers と同一の計算をしている (証明成立)。")
    else:
        print("→ 不一致あり。差分を上記で確認 (隠さない)。")
    print("=" * 70)
    return 0 if n_match == len(PROMPTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
