# SPDX-License-Identifier: Apache-2.0
"""一番正直な証明 — 同じ Qwen 重みを HF transformers と llcore 自前 forward で動かし出力一致を実演。

記事 #43 の主張「llcore は HF の薄いラッパーでなく、自分で書いた推論コード (runtime/qwen2.py) で
事前学習済み重みを動かしている」を、誰でも再現できる形で証明する。2 段構成:

  [定性] 人間が読める数問の greedy 生成を HF / native で並べる (掴み・目視確認)。
  [定量] コーパスを teacher-forcing 1 パスで両者に流し、**全 N トークン位置**の
         next-token argmax 一致率と logits の max/mean |Δ| を出す (N=数千、統計的強度)。
         N=5 では「たまたま一致」を排除できない (記事 #43 第4章の N 不足批判が自分に返る)
         ため、定量で N を 3 桁に上げて初めて「同一計算」を主張する。

honest: greedy / argmax (決定論) でのみ一致を主張する。capability (賢さ) は Qwen の重み由来で
本スクリプトが作るものではない — 証明するのは「同じ計算をしている」ことだけ。不一致も隠さない
(feedback_benchmark_honest_disclosure)。

    py -3.11 scripts/prove_native_matches_hf.py [--model-dir DIR] [--bulk-tokens N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

from llcore.lm.device import resolve_device
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
            stream.reconfigure(encoding="utf-8", errors="replace")


def _messages(user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---- 定性: 人間が読める greedy 生成の並置 -------------------------------------------------


def hf_greedy(model: Any, tok: Any, user: str, max_new: int, device: torch.device) -> list[int]:
    inp = tok.apply_chat_template(
        _messages(user), add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0
        )
    toks = [int(t) for t in out[0, inp["input_ids"].shape[1] :].tolist()]
    # HF generate は停止時に EOS を出力列へ含めるが native は EOS で break して含めない。
    # コンテンツトークンを揃えて比較するため末尾の EOS を 1 つだけ取り除く。
    eos = getattr(tok, "eos_token_id", None)
    if toks and eos is not None and toks[-1] == eos:
        toks = toks[:-1]
    return toks


def native_greedy(model: Any, tok: Any, user: str, max_new: int, device: torch.device) -> list[int]:
    text = tok.apply_chat_template(_messages(user), tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    eos = getattr(tok, "eos_token_id", None)
    toks: list[int] = []
    with torch.no_grad():
        logits, cache = model(ids, return_cache=True)
        for _ in range(max_new):
            nxt = int(logits[0, -1].argmax().item())
            if eos is not None and nxt == eos:
                break
            toks.append(nxt)
            logits, cache = model(torch.tensor([[nxt]], device=device), past=cache, return_cache=True)
    return toks


def qualitative(
    hf_model: Any, hf_tok: Any, nat_model: Any, nat_tok: Any, max_new: int, device: torch.device
) -> int:
    """5 問を目視確認用に並置。一致した問の数を返す (掴みであって主たる証拠ではない)。"""
    n_match = 0
    for user in PROMPTS:
        hf_toks = hf_greedy(hf_model, hf_tok, user, max_new, device)
        nat_toks = native_greedy(nat_model, nat_tok, user, max_new, device)
        same = hf_toks == nat_toks
        n_match += int(same)
        hf_text = hf_tok.decode(hf_toks, skip_special_tokens=True).strip()
        nat_text = nat_tok.decode(nat_toks, skip_special_tokens=True).strip()
        print(f"Q: {user}")
        print(f"   HF     : {hf_text!r}")
        print(f"   native : {nat_text!r}")
        print(f"   → tokens {'一致 ✓' if same else '不一致 ✗'}\n", flush=True)
    return n_match


# ---- 定量: コーパス全位置の next-token 一致 (N=数千) ----------------------------------------


def quantitative(
    hf_model: Any, nat_model: Any, tok: Any, text: str, n_tokens: int, chunk: int,
    device: torch.device,
) -> dict[str, float]:
    """teacher-forcing 1 パスで全 N 位置の next-token argmax 一致率と logits |Δ| を測る。

    各 chunk を cold (前文脈なし) で両者に同条件で流す。N が大きいほど「たまたま一致」を排除できる。
    """
    ids_all = tok(text, return_tensors="pt").input_ids[0].to(device)
    n_tokens = min(n_tokens, int(ids_all.numel()))
    n_match = 0
    n_total = 0
    max_abs = 0.0
    sum_abs = 0.0
    n_elem = 0
    for start in range(0, n_tokens, chunk):
        stop = min(start + chunk, n_tokens)
        ids = ids_all[start:stop].unsqueeze(0)
        with torch.no_grad():
            hf_logits = hf_model(ids).logits[0].float()
            nat_logits, _ = nat_model(ids, return_cache=True)
            nat_logits = nat_logits[0].float()
        diff = (hf_logits - nat_logits).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        sum_abs += float(diff.sum().item())
        n_elem += int(diff.numel())
        n_match += int((hf_logits.argmax(-1) == nat_logits.argmax(-1)).sum().item())
        n_total += stop - start
    return {
        "n_match": float(n_match),
        "n_total": float(n_total),
        "argmax_match_pct": 100.0 * n_match / max(n_total, 1),
        "logits_max_abs": max_abs,
        "logits_mean_abs": sum_abs / max(n_elem, 1),
    }


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="prove llcore native forward == HF transformers")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=24, help="定性 greedy の生成上限")
    ap.add_argument("--corpus", default="out/corpus_aozora_multi.txt", help="定量コーパス")
    ap.add_argument("--bulk-tokens", type=int, default=2048, help="定量で比較するトークン数 N")
    ap.add_argument("--chunk", type=int, default=256, help="定量の 1 パスチャンク長")
    ap.add_argument("--bulk-chars", type=int, default=12000, help="コーパス先読み文字数")
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print("一番正直な証明: 同じ Qwen 重み × {HF transformers, llcore 自前 forward} → 出力一致")
    print(f"  model: {args.model_dir}")
    print("=" * 72)

    t0 = time.perf_counter()
    hf_tok = AutoTokenizer.from_pretrained(args.model_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    hf_model.eval()  # type: ignore[no-untyped-call]
    print(f"[load] HF transformers       ({time.perf_counter() - t0:.1f}s)", flush=True)
    t0 = time.perf_counter()
    nat_model, nat_tok, _ = load_qwen2(args.model_dir)
    print(f"[load] llcore native forward ({time.perf_counter() - t0:.1f}s)\n", flush=True)

    print("--- 定性 (掴み: 人間が読める 5 問) ---")
    n_q = qualitative(hf_model, hf_tok, nat_model, nat_tok, args.max_new)

    print("--- 定量 (主たる証拠: コーパス全位置の next-token 一致) ---")
    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        text = corpus_path.read_text(encoding="utf-8")[: args.bulk_chars]
        t0 = time.perf_counter()
        q = quantitative(hf_model, nat_model, hf_tok, text, args.bulk_tokens, args.chunk)
        nt = int(q["n_total"])
        nm = int(q["n_match"])
        print(f"  corpus={corpus_path.name}  N={nt} トークン  ({time.perf_counter() - t0:.1f}s)")
        print(f"  next-token argmax 一致: {nm}/{nt} = {q['argmax_match_pct']:.3f}%")
        print(f"  logits |Δ|  max={q['logits_max_abs']:.2e}  mean={q['logits_mean_abs']:.2e}")
        bulk_ok = nm == nt
    else:
        print(f"  (コーパス {corpus_path} が無いため定量はスキップ)")
        q = {}
        bulk_ok = False

    print("\n" + "=" * 72)
    print(f"定性: {n_q}/{len(PROMPTS)} 問一致 (目視)")
    if q:
        print(f"定量: N={int(q['n_total'])} トークン中 {q['argmax_match_pct']:.3f}% で "
              f"next-token argmax 一致 / logits 最大差 {q['logits_max_abs']:.2e}")
    if n_q == len(PROMPTS) and bulk_ok:
        print("→ 定性・定量とも完全一致。llcore 自前 forward は HF と同一の計算をしている (証明成立)。")
    elif q and q["argmax_match_pct"] >= 99.9:
        print("→ 定量はほぼ完全一致 (微差は浮動小数の丸め)。差分箇所は上記で確認。")
    else:
        print("→ 不一致あり。差分を上記で確認 (隠さない)。")
    print("=" * 72)
    return 0 if (n_q == len(PROMPTS) and bulk_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
