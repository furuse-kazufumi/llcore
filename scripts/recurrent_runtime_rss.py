# SPDX-License-Identifier: Apache-2.0
"""recurrent vs GPT の **runtime peak RSS** スイープ — 解析値を実機計測へ昇格。

`scripts/memory_footprint_harness.py` は state_bytes(実測)+ KV/attn(解析値)+ noisy RSS を
測った。本スクリプトはその (0') 後続: **実際の生成ワークロードを文脈長 T で振り、別プロセス隔離で
peak working set を実測**する。重み構成(n_embd / n_layer)は T 間で固定なので、peak WS の増分は
**純粋に文脈長依存コスト**(GPT は KV + attention 行列、recurrent/RWKV は無し)を表す。

測るもの(T を増やしたとき)
----------------------------
- **GPT**: block_size=T のモデルで長さ T の forward を 1 回。attention 行列 [n_head,T,T] は **O(T²)**、
  KV は O(T) で、これらが peak WS を押し上げる(超線形に増える想定)。
- **RecurrentLM / RWKVLM**: T 個のトークンを step ループで流す。状態は固定サイズなので peak WS は
  **T によらず平坦**になる想定。

これにより「GPT は文脈に対しメモリが膨らむ / recurrent は定数」を **解析値でなく実機 peak RSS** で示す。

honest 留保
-----------
- peak WS は torch ランタイム + 固定重み + T 依存バッファの合算。クリーンな信号は **T_min→T_max の
  増分トレンド**(固定部分は相殺)。各 (mode, T) は別プロセスで測る(プロセス生涯 peak のため)。
- GPT.generate は block_size に crop するため実行上は有界。ここでは「**block_size=T で厳密に T を
  attention する**」前提の必要量を測る(harness と同じ解釈)。
- RSS は WinAPI 実測。非 Windows / 取得失敗時は 0。

使い方::

    py -3.11 scripts/recurrent_runtime_rss.py
    py -3.11 scripts/recurrent_runtime_rss.py --lengths 256,512,1024,2048,4096
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch

from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM  # type: ignore[import-untyped]
from llcore.lm.rwkv import RWKVConfig, RWKVLM  # type: ignore[import-untyped]
from llcore.runtime.rss import peak_working_set_bytes as _peak_working_set_bytes  # type: ignore[import-untyped]

RESULT_PREFIX = "RESULT_JSON="
MODES = ("gpt", "recurrent", "rwkv")


@torch.no_grad()
def run_worker(mode: str, t: int, n_embd: int, n_layer: int, n_head: int, vocab: int) -> dict[str, Any]:
    """1 つの (mode, T) で実ワークロードを走らせ peak WS を測る(隔離 subprocess 内)。"""
    torch.manual_seed(1337)
    if mode == "gpt":
        model = CharGPT(GPTConfig(vocab_size=vocab, block_size=t, n_layer=n_layer,
                                  n_head=n_head, n_embd=n_embd))
        model.eval()
        idx = torch.randint(0, vocab, (1, t))
        out = cast(CharGPT, model).forward_logits(idx)  # 長さ T を厳密 attention
        sink = float(out.float().sum().item())
    elif mode == "recurrent":
        model = RecurrentLM(RecurrentConfig(vocab_size=vocab, block_size=t, n_layer=n_layer,
                                            n_embd=n_embd, state_size=n_embd))
        model.eval()
        idx = torch.randint(0, vocab, (1,))
        state = None
        for _ in range(t):  # 定数サイズ状態を T ステップ更新
            _, state = model.step(idx, state)
        sink = float(model.state_bytes(state))
    else:  # rwkv
        model = RWKVLM(RWKVConfig(vocab_size=vocab, block_size=t, n_layer=n_layer, n_embd=n_embd))
        model.eval()
        idx = torch.randint(0, vocab, (1,))
        state = None
        for _ in range(t):
            _, state = model.step(idx, state)
        sink = float(model.state_bytes(state))
    peak = _peak_working_set_bytes()
    return {"mode": mode, "t": t, "peak_ws_mb": round(peak / 1e6, 1), "sink": sink}


def _spawn_worker(mode: str, t: int, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--t", str(t),
           "--n-embd", str(args.n_embd), "--n-layer", str(args.n_layer),
           "--n-head", str(args.n_head), "--vocab", str(args.vocab)]
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="runtime peak RSS sweep: recurrent (flat) vs GPT (grows)")
    ap.add_argument("--lengths", default="256,512,1024,2048")
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--json", default="out/recurrent_runtime_rss.json")
    ap.add_argument("--worker", choices=list(MODES), default=None)
    ap.add_argument("--t", type=int, default=None)
    args = ap.parse_args(argv)

    # --- worker role ---
    if args.worker is not None:
        if args.t is None or args.t <= 0:
            print("error: --worker requires positive --t", file=sys.stderr)
            return 2
        result = run_worker(args.worker, args.t, args.n_embd, args.n_layer, args.n_head, args.vocab)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0

    # --- parent role ---
    try:
        lengths = _parse_lengths(args.lengths)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"config: n_embd={args.n_embd} L={args.n_layer} H={args.n_head} vocab={args.vocab}  lengths={lengths}")
    records: dict[str, list[dict[str, Any]]] = {m: [] for m in MODES}
    for t in lengths:
        for mode in MODES:
            records[mode].append(_spawn_worker(mode, t, args))

    print("\n| T | GPT peak WS | Recurrent peak WS | RWKV peak WS |")
    print("|" + "---|" * 4)
    for i, t in enumerate(lengths):
        print(
            f"| {t} | {records['gpt'][i]['peak_ws_mb']} MB | "
            f"{records['recurrent'][i]['peak_ws_mb']} MB | {records['rwkv'][i]['peak_ws_mb']} MB |"
        )

    def _growth(mode: str) -> float:
        lo = records[mode][0]["peak_ws_mb"]
        hi = records[mode][-1]["peak_ws_mb"]
        return (hi / lo) if lo > 0 else float("nan")

    print(
        f"\n[headline] T {lengths[0]}→{lengths[-1]}(×{lengths[-1] / lengths[0]:.0f})の runtime peak WS 倍率: "
        f"GPT ×{_growth('gpt'):.2f}(文脈で膨張)/ Recurrent ×{_growth('recurrent'):.2f} / "
        f"RWKV ×{_growth('rwkv'):.2f}(定数状態=ほぼ平坦)。解析値(KV 線形 / attn 二次)を実機 peak RSS で裏取り。"
    )
    print(
        "[honest] peak WS は torch ランタイム + 固定重み + T 依存バッファの合算。"
        " クリーンな信号は増分トレンド。GPT.generate は block_size crop で実行上は有界(本測は厳密長文脈想定)。"
    )

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {"n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
                   "vocab": args.vocab, "lengths": lengths},
        "records": records,
        "growth_ratio": {m: round(_growth(m), 3) for m in MODES},
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
