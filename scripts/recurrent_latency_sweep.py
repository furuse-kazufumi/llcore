# SPDX-License-Identifier: Apache-2.0
"""recurrent vs GPT の **推論レイテンシ** スイープ — メモリ軸(``recurrent_runtime_rss.py``)の
相補となる *compute 軸* を実機 wall-clock で測る。

測るもの(文脈長 T を増やしたとき)
------------------------------------
- **GPT**: block_size=T のモデルで長さ T の ``forward_logits`` を 1 回。attention スコア行列は
  [n_head,T,T] = **O(T²)** なので、wall-clock は T に対し **超線形**に伸びる想定。
- **RecurrentLM / RWKVLM**: T 個のトークンを step ループで流す。1 step は固定コストなので合計は
  **O(T)**(T に対し線形)の想定。

**honest 留保(最重要):cross-mode の絶対時間は比較してはいけない。**
recurrent/RWKV はここでは **Python の per-step ループ**(T 回の関数呼び出し)で測っており、
インタプリタ呼び出しオーバーヘッドが支配的になる。一方 GPT は **1 回の vectorized forward**。
したがって「recurrent の方が遅い/速い」を絶対値で語るのは無意味。

意味があるのは **各モード内で T を変えたときのスケーリング(伸び方の指数)** だけ:
GPT が超線形(指数 > 1、理想 O(T²) なら ~2)、recurrent/RWKV が線形(指数 ~1)に出るかを見る。
これは ``recurrent_runtime_rss.py`` が「絶対値でなく増分トレンドを読む」としたのと同じ規律。

使い方::

    py -3.11 scripts/recurrent_latency_sweep.py
    py -3.11 scripts/recurrent_latency_sweep.py --lengths 128,256,512,1024,2048 --repeats 5
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

from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM  # type: ignore[import-untyped]
from llcore.lm.rwkv import RWKVConfig, RWKVLM  # type: ignore[import-untyped]

RESULT_PREFIX = "RESULT_JSON="
MODES = ("gpt", "recurrent", "rwkv")


@torch.no_grad()
def _time_once(mode: str, model: Any, t: int, vocab: int) -> float:
    """1 回分の推論 wall-clock(秒)を返す。"""
    if mode == "gpt":
        idx = torch.randint(0, vocab, (1, t))
        start = time.perf_counter()
        out = cast(CharGPT, model).forward_logits(idx)
        _ = float(out.float().sum().item())  # 計算を実体化(lazy 回避)
        return time.perf_counter() - start
    # recurrent / rwkv: 固定状態を T ステップ更新(O(T) の per-step ループ)
    idx = torch.randint(0, vocab, (1,))
    state = None
    start = time.perf_counter()
    for _ in range(t):
        _, state = model.step(idx, state)
    _ = float(model.state_bytes(state))  # 実体化
    return time.perf_counter() - start


@torch.no_grad()
def run_worker(mode: str, t: int, n_embd: int, n_layer: int, n_head: int,
               vocab: int, repeats: int, warmup: int) -> dict[str, Any]:
    """1 つの (mode, T) で warmup 後 ``repeats`` 回測り、中央値を返す(隔離 subprocess 内)。"""
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
        _time_once(mode, model, t, vocab)
    samples = [_time_once(mode, model, t, vocab) for _ in range(max(1, repeats))]
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
    """log-log 最小二乗で time ~ T^p の指数 p を推定(O(T) なら ~1, O(T²) なら ~2)。"""
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
    ap = argparse.ArgumentParser(description="inference latency sweep: GPT (superlinear) vs recurrent (linear)")
    ap.add_argument("--lengths", default="128,256,512,1024,2048")
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json", default="out/recurrent_latency_sweep.json")
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
          f"lengths={lengths} repeats={args.repeats}")
    records: dict[str, list[dict[str, Any]]] = {m: [] for m in MODES}
    for t in lengths:
        for mode in MODES:
            records[mode].append(_spawn_worker(mode, t, args))

    print("\n| T | GPT median ms | Recurrent median ms | RWKV median ms |")
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
        f"\n[headline] T {lengths[0]}->{lengths[-1]}(x{lengths[-1] / lengths[0]:.0f}) scaling 指数 p "
        f"(min ベース=混雑に強い / 括弧内 median): "
        f"GPT p~{exponents_min['gpt']}({exponents['gpt']}, O(T^2)寄り)/ "
        f"Recurrent p~{exponents_min['recurrent']}({exponents['recurrent']}, 線形寄り)/ "
        f"RWKV p~{exponents_min['rwkv']}({exponents['rwkv']})。"
    )
    print(
        "[honest] cross-mode の絶対 ms は比較不可(recurrent は Python per-step ループ=インタプリタ律速、"
        " GPT は 1 回の vectorized forward)。読むのは各モード内の T に対する伸び方(scaling 指数)のみ。"
    )

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {"n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
                   "vocab": args.vocab, "lengths": lengths, "repeats": args.repeats},
        "records": records,
        "growth_ratio": growth,
        "scaling_exponent": exponents,
        "scaling_exponent_min": exponents_min,
        "honest_note": ("cross-mode absolute ms not comparable (recurrent uses a Python per-step "
                        "loop, GPT a single vectorized forward); read only the per-mode scaling exponent."),
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
