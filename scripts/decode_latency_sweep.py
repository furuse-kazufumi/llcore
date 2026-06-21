# SPDX-License-Identifier: Apache-2.0
"""decode 軸レイテンシ — *context age T における 1 トークン decode コスト* を実機 wall-clock で測る。

``recurrent_latency_sweep.py`` は **prefill/batch-forward 全体**(長さ T を 1 回 forward)の compute を
測った。本ハーネスはその相補となる **decode(自己回帰生成)軸** を測る:「既に長さ T の文脈がある状態で、
**次の 1 トークンを出す**のに掛かる時間」を T を変えて見る。これは streaming 生成の per-token コストで、
アーキ差が最も鋭く出る regime。

測るもの(文脈長 T を増やしたとき、**decode 1 step** の時間)
----------------------------------------------------------------
- **GPT**: 本実装 :meth:`CharGPT.generate` は **KV cache を持たない**(毎 step 全文脈を再 forward する)。
  したがって context age T での 1 decode step = 長さ T の forward 1 回 = attention O(T²)。**T とともに増大**。
- **RecurrentLM / RWKVLM**: 定数サイズ状態を T step 進めて(= warmup、非計測)状態を「熟成」させた後、
  **追加の 1 step だけ**計時する。1 step は固定コストなので **T に依らず一定(O(1))=flat** の想定。

**honest 留保(最重要):**
1. **cross-mode の絶対時間は比較不可**。recurrent/RWKV は Python per-step ループ(インタプリタ呼び出し律速)、
   GPT は 1 回の vectorized forward。読むのは **各モード内で T を変えたときの伸び方(scaling 指数 p)** のみ。
2. **この GPT は KV cache を持たない**。production の LLM serving は KV cache で decode を **O(T)/token** に落とす
   (本実装の O(T²) ではない)。ただし **cache 有りでも GPT の decode は T とともに増える**一方、recurrent は
   O(1) で **flat** のまま — 「flat vs 増大」という質的対比が load-bearing で、GPT 側の指数値は実装依存。
3. 計測は圧力のかからない小モデルで行う(RAM 圧で倍率が暴れる regime を避ける、a7 と同方針)。

非計測 prefill / 計測 1-step という**非対称**は意図的:それが「recurrent は熟成済み状態に 1 step 足すだけ /
cache 無し GPT は毎回フル再計算」というアーキの事実そのものだから。

使い方::

    py -3.11 scripts/decode_latency_sweep.py
    py -3.11 scripts/decode_latency_sweep.py --lengths 128,256,512,1024,2048 --repeats 11
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM

RESULT_PREFIX = "RESULT_JSON="
MODES = ("gpt", "recurrent", "rwkv")


@torch.no_grad()
def _decode_step_once(mode: str, model: Any, t: int, vocab: int) -> float:
    """context age T での **1 decode step** の wall-clock(秒)を返す。

    GPT は cache 無しゆえ「長さ T の forward 1 回」が 1 decode step。recurrent/RWKV は T step の
    warmup(非計測)で状態を熟成させてから **追加 1 step** だけ計時する。
    """
    if mode == "gpt":
        idx = torch.randint(0, vocab, (1, t))
        start = time.perf_counter()
        out = cast(CharGPT, model).forward_logits(idx)
        _ = float(out[:, -1, :].float().sum().item())  # 末尾 logits を実体化(lazy 回避)
        return time.perf_counter() - start
    # recurrent / rwkv: T step 進めて状態を熟成(非計測)→ 追加 1 step だけ計時。
    idx = torch.randint(0, vocab, (1,))
    state: Any = None
    for _ in range(t):
        _, state = model.step(idx, state)
    start = time.perf_counter()
    _, state = model.step(idx, state)
    _ = float(model.state_bytes(state))  # 実体化
    return time.perf_counter() - start


@torch.no_grad()
def run_worker(mode: str, t: int, n_embd: int, n_layer: int, n_head: int,
               vocab: int, repeats: int, warmup: int) -> dict[str, Any]:
    """1 つの (mode, T) で warmup 後 ``repeats`` 回測り、中央値/最小を返す(隔離 subprocess 内)。"""
    torch.manual_seed(1337)
    torch.set_num_threads(1)  # 計測の決定性を上げる(BLAS スレッド変動を抑制)
    if mode == "gpt":
        model: Any = CharGPT(GPTConfig(vocab_size=vocab, block_size=t, n_layer=n_layer,
                                       n_head=n_head, n_embd=n_embd))
    elif mode == "recurrent":
        model = RecurrentLM(RecurrentConfig(vocab_size=vocab, block_size=t, n_layer=n_layer,
                                            n_embd=n_embd, state_size=n_embd))
    else:
        model = RWKVLM(RWKVConfig(vocab_size=vocab, block_size=t, n_layer=n_layer, n_embd=n_embd))
    model.eval()

    for _ in range(max(0, warmup)):
        _decode_step_once(mode, model, t, vocab)
    samples = [_decode_step_once(mode, model, t, vocab) for _ in range(max(1, repeats))]
    return {
        "mode": mode, "t": t,
        "median_ms": round(statistics.median(samples) * 1e3, 4),
        "min_ms": round(min(samples) * 1e3, 4),
        "repeats": len(samples),
    }


def _spawn_worker(mode: str, t: int, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--t", str(t),
           "--n-embd", str(args.n_embd), "--n-layer", str(args.n_layer),
           "--n-head", str(args.n_head), "--vocab", str(args.vocab),
           "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"worker {mode}@{t} failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return cast("dict[str, Any]", json.loads(line[len(RESULT_PREFIX):]))
    raise RuntimeError(f"worker {mode}@{t} produced no result line")


def _parse_lengths(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",")]
    if not raw.strip() or any(not p for p in parts):
        raise ValueError("--lengths must be a non-empty comma list")
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--lengths must contain only integers") from exc
    if any(v <= 0 for v in values):
        raise ValueError("--lengths must be positive")
    return sorted(dict.fromkeys(values))


def _scaling_exponent(lengths: list[int], times_ms: list[float]) -> float:
    """log-log 最小二乗で time ~ T^p の指数 p を推定(flat O(1) なら ~0, O(T) なら ~1, O(T²) なら ~2)。"""
    xs = [math.log(t) for t in lengths]
    ys = [math.log(v) for v in times_ms if v > 0]
    if len(ys) != len(xs) or len(xs) < 2:
        return float("nan")
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="decode-step latency vs context age T: GPT (no KV cache, grows) vs recurrent (O(1) flat)")
    ap.add_argument("--lengths", default="128,256,512,1024,2048")
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    # recurrent/RWKV は初回 step の lazy init(状態バッファ確保等)を吸収するため warmup を多めに。
    # warmup=2 では RWKV の小 T が startup 外れ値になり、warmup>=5 で全 T が flat に落ち着くのを実測。
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json", default="out/decode_latency_sweep.json")
    ap.add_argument("--worker", choices=list(MODES), default=None)
    ap.add_argument("--t", type=int, default=None)
    args = ap.parse_args(argv)

    # --- worker role ---
    if args.worker is not None:
        if args.t is None or args.t <= 0:
            print("error: --worker requires positive --t", file=sys.stderr)
            return 2
        result = run_worker(args.worker, args.t, args.n_embd, args.n_layer, args.n_head,
                            args.vocab, args.repeats, args.warmup)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0

    # --- parent role ---
    try:
        lengths = _parse_lengths(args.lengths)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"config: n_embd={args.n_embd} L={args.n_layer} H={args.n_head} vocab={args.vocab}  "
          f"lengths={lengths} repeats={args.repeats} (decode 1-step @ context age T)")
    records: dict[str, list[dict[str, Any]]] = {m: [] for m in MODES}
    for t in lengths:
        for mode in MODES:
            records[mode].append(_spawn_worker(mode, t, args))

    print("\n| T (context age) | GPT decode ms | Recurrent decode ms | RWKV decode ms |")
    print("|" + "---|" * 4)
    for i, t in enumerate(lengths):
        print(f"| {t} | {records['gpt'][i]['median_ms']} | "
              f"{records['recurrent'][i]['median_ms']} | {records['rwkv'][i]['median_ms']} |")

    def _growth(mode: str) -> float:
        lo = records[mode][0]["median_ms"]
        hi = records[mode][-1]["median_ms"]
        return (hi / lo) if lo > 0 else float("nan")

    exponents = {m: round(_scaling_exponent(lengths, [r["median_ms"] for r in records[m]]), 3)
                 for m in MODES}
    exponents_min = {m: round(_scaling_exponent(lengths, [r["min_ms"] for r in records[m]]), 3)
                     for m in MODES}
    growth = {m: round(_growth(m), 3) for m in MODES}

    print(
        f"\n[headline] decode-step: T {lengths[0]}->{lengths[-1]}(x{lengths[-1] / lengths[0]:.0f}) "
        f"scaling 指数 p (min ベース / 括弧内 median): "
        f"GPT p~{exponents_min['gpt']}({exponents['gpt']}, KV cache 無=O(T^2)寄りで増大)/ "
        f"Recurrent p~{exponents_min['recurrent']}({exponents['recurrent']}, O(1)=flat 想定)/ "
        f"RWKV p~{exponents_min['rwkv']}({exponents['rwkv']})。"
    )
    print(
        "[honest] (1) cross-mode の絶対 ms は比較不可(recurrent=Python per-step / GPT=vectorized forward)。"
        " (2) この GPT は KV cache 無=毎 step 全文脈を再 forward。production は cache で decode を O(T)/token に"
        "落とすが、それでも T とともに増大し、recurrent の O(1) flat とは質的に異なる。読むのは各モード内の指数のみ。"
    )

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {"n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
                   "vocab": args.vocab, "lengths": lengths, "repeats": args.repeats},
        "axis": "decode-step latency vs context age T (complements recurrent_latency_sweep's prefill/batch-forward axis)",
        "records": records,
        "growth_ratio": growth,
        "scaling_exponent": exponents,
        "scaling_exponent_min": exponents_min,
        "honest": [
            "cross-mode absolute ms not comparable (recurrent=Python per-step loop vs GPT=vectorized forward)",
            "this GPT has NO KV cache (re-forwards full context each decode step => O(T^2)); production serving "
            "uses a KV cache making decode O(T)/token, but even cached GPT decode grows with T while recurrent "
            "stays O(1) flat. Only within-mode scaling exponent is load-bearing.",
        ],
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
