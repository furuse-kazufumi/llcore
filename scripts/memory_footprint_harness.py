# SPDX-License-Identifier: Apache-2.0
"""実測メモリ harness — 定数状態 recurrent vs 文脈長依存 GPT のメモリ footprint を測る。

llcore の北極星「メモリ使用効率」(2026-06-16 pivot, memory:project_llcore_memory_efficiency_pivot)
の第一歩。recurrent verdict は memory@T 曲線を「config 由来の構造プロット (実測でない)」と
明記していたので、これを **実機の数値** へ昇格させる。

測るもの (T = 文脈長 / 生成ステップ数 を増やしたとき):
  - **RecurrentLM / RWKVLM**: 各モデルの ``state_bytes(state)`` = 生成時に保持する recurrent
    状態テンソルの**実バイト数**。理論上 T に依らず一定 → これを実機 step ループで確認する (REAL)。
  - **CharGPT**: 文脈長 T を厳密に attention するのに保持すべき **KV-cache バイト**
    (= 2·n_layer·T·n_embd·4) と、forward 中に確保される **attention 行列バイト**
    (= n_head·T²·4)。これらは config 由来の解析値だが、実 forward を T 各点で走らせて
    **プロセス RSS (working set) の増分**でトレンドを裏取りする。

honest 留保:
  - ``state_bytes`` は実テンソルのバイト数 = **実測**。GPT の KV/attn バイトは **解析値**
    (config から算出) で、RSS トレンドで裏取りする (torch CPU には cuda.max_memory_allocated 相当が
    無いため厳密なテンソル単位計測は RSS で近似する)。
  - GPT の ``generate`` は文脈を block_size に **crop** するので生成長に対しては有界 = 実行上は
    block_size で頭打ち。「線形」は『block_size を伸ばして厳密長文脈を attention する』場合の
    必要量。recurrent の固定状態は原理的に任意の過去を一定サイズで運べる、という対比を示す。
  - RSS は torch のアロケータ・キャッシュ・断片化でノイジー。クリーンな信号は state_bytes (real)
    と KV/attn (解析値)。RSS は補助。**負の RSS Δ（ページアウト等）は 0 に丸める**。
  - 先頭点の lazy allocation ノイズを下げるため、既定で **1 回 warmup** してから測る。
    ただし RSS は補助指標であり、allocator/cache の影響は残る。
  - Windows telemetry の ctypes struct / WinAPI 呼び出しは、CI では fake 置換で shape 回帰のみを見る。
    実 struct layout の妥当性は live Windows 実行で確認する。

使い方::

    py -3.11 scripts/memory_footprint_harness.py
    py -3.11 scripts/memory_footprint_harness.py --lengths 64,128,256,512,1024 --json out/mem_footprint.json
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM
from llcore.runtime.rss import (
    process_memory as _process_memory,
    working_set_bytes as _working_set_bytes,
)


def _system_memory_snapshot() -> dict[str, int] | None:
    """Windows の commit/pagefile/physical memory スナップショットを返す。"""
    try:
        import ctypes.wintypes as wt

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        kernel32.GlobalMemoryStatusEx.restype = wt.BOOL

        ms = _MemoryStatusEx()
        ms.dwLength = ctypes.sizeof(ms)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return None

        snapshot = {
            "memory_load_percent": int(ms.dwMemoryLoad),
            "total_phys_bytes": int(ms.ullTotalPhys),
            "avail_phys_bytes": int(ms.ullAvailPhys),
            "total_commit_bytes": int(ms.ullTotalPageFile),
            "avail_commit_bytes": int(ms.ullAvailPageFile),
        }
        pmc = _process_memory()
        if pmc is not None:
            snapshot["process_working_set_bytes"] = pmc.working_set
            snapshot["process_pagefile_bytes"] = pmc.pagefile
            snapshot["process_peak_pagefile_bytes"] = pmc.peak_pagefile
        return snapshot
    except Exception:  # noqa: BLE001 - 補助 telemetry が取れなくても本体を落とさない
        return None


def _snapshot_summary(snapshot: dict[str, int] | None) -> dict[str, float | None] | None:
    """生 snapshot を MB / percent の簡約値へ変換する。process 側は部分成功を許容。"""
    if snapshot is None:
        return None
    return {
        "memory_load_percent": float(snapshot["memory_load_percent"]),
        "avail_phys_mb": round(snapshot["avail_phys_bytes"] / 1e6, 1),
        "avail_commit_mb": round(snapshot["avail_commit_bytes"] / 1e6, 1),
        "process_working_set_mb": (
            round(snapshot["process_working_set_bytes"] / 1e6, 1)
            if "process_working_set_bytes" in snapshot else None
        ),
        "process_pagefile_mb": (
            round(snapshot["process_pagefile_bytes"] / 1e6, 1)
            if "process_pagefile_bytes" in snapshot else None
        ),
        "process_peak_pagefile_mb": (
            round(snapshot["process_peak_pagefile_bytes"] / 1e6, 1)
            if "process_peak_pagefile_bytes" in snapshot else None
        ),
    }


@torch.no_grad()
def _recurrent_state_bytes(model: RecurrentLM | RWKVLM, t: int, warmup: int = 1) -> tuple[int, float]:
    """T 個のトークンを step ループで流し、最終状態の state_bytes と RSS 増分を返す。

    RSS Δ は補助指標であり、負値 (ページアウト等) は 0 に丸める。
    """
    model.eval()
    idx = torch.randint(0, model.config.vocab_size, (1,))

    def run() -> tuple[int, Any]:
        state: Any = None  # model は RecurrentLM|RWKVLM の union ゆえ state 型は実行時依存
        sb = 0
        for _ in range(t):
            _, state = model.step(idx, state)
            sb = model.state_bytes(state)  # T に依らず一定のはず
        return sb, state

    for _ in range(warmup):
        _, state = run()
        del state
    gc.collect()
    before = _working_set_bytes()
    sb, state = run()
    after = _working_set_bytes()
    del state
    return sb, max(0.0, (after - before) / 1e6)


def _parse_lengths(raw: str) -> list[int]:
    """comma-separated lengths を fail-closed に正規化する。

    重複は落とすが、ユーザーが与えた順序は保持する。
    """
    raw_parts = raw.split(",")
    parts = [part.strip() for part in raw_parts]
    if not raw.strip():
        raise ValueError("--lengths must contain at least one positive integer")
    if any(not part for part in parts):
        raise ValueError("--lengths must not contain empty items")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--lengths must contain only positive integers") from exc
    invalid = [value for value in values if value <= 0]
    if invalid:
        bad = ", ".join(str(value) for value in invalid)
        raise ValueError(f"--lengths must contain only positive integers; got {bad}")
    return list(dict.fromkeys(values))


def _headline_range(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the low/high-T records used for the growth headline.

    Table output preserves user order, but the headline compares the smallest and
    largest measured context lengths so growth labels remain meaningful even
    when `--lengths` is supplied in descending or mixed order.
    """
    by_t = sorted(records, key=lambda record: cast(int, record["T"]))
    return by_t[0], by_t[-1]


@torch.no_grad()
def _gpt_context(model: CharGPT, t: int, warmup: int = 1) -> tuple[int, int, float]:
    """T トークンの実 forward を走らせ、(KV-cache バイト解析値, attn 行列バイト解析値, RSS増分) を返す。

    RSS Δ は補助指標であり、負値 (ページアウト等) は 0 に丸める。
    """
    model.eval()
    cfg = model.config
    kv_bytes = 2 * cfg.n_layer * t * cfg.n_embd * 4  # k,v それぞれ [L,T,C] float32
    attn_bytes = cfg.n_head * t * t * 4              # 1 層あたり attention 行列 [nh,T,T] (transient)
    idx = torch.randint(0, cfg.vocab_size, (1, t))

    def run() -> Tensor:
        logits, _ = model(idx)
        return cast(Tensor, logits)

    for _ in range(warmup):
        out = run()
        del out
    gc.collect()
    before = _working_set_bytes()
    out = run()
    after = _working_set_bytes()
    del out
    return kv_bytes, attn_bytes, max(0.0, (after - before) / 1e6)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="empirical memory footprint: recurrent vs GPT")
    ap.add_argument("--lengths", default="64,128,256,512,1024", help="context lengths (comma)")
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--json", default="out/mem_footprint.json")
    ap.add_argument("--warmup", type=int, default=1, help="warmup forwards/step-loops before measuring RSS")
    args = ap.parse_args(argv)
    try:
        lengths = _parse_lengths(args.lengths)
        if args.warmup < 0:
            raise ValueError("--warmup must be non-negative")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    torch.manual_seed(1337)
    max_t = max(lengths)
    gpt = CharGPT(GPTConfig(vocab_size=args.vocab, block_size=max_t, n_layer=args.n_layer,
                            n_head=args.n_head, n_embd=args.n_embd))
    rec = RecurrentLM(RecurrentConfig(vocab_size=args.vocab, block_size=max_t, n_layer=args.n_layer,
                                      n_embd=args.n_embd, state_size=args.n_embd))
    rwkv = RWKVLM(RWKVConfig(vocab_size=args.vocab, block_size=max_t, n_layer=args.n_layer,
                             n_embd=args.n_embd))
    system_before = _snapshot_summary(_system_memory_snapshot())
    def _np(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())
    print(f"models: GPT {_np(gpt):,} / Recurrent {_np(rec):,} / "
          f"RWKV {_np(rwkv):,} params  (n_embd={args.n_embd} L={args.n_layer})")
    if system_before is not None:
        proc_ws = system_before["process_working_set_mb"]
        proc_commit = system_before["process_pagefile_mb"]
        print("[system] before: "
              f"load={system_before['memory_load_percent']:.0f}% "
              f"avail_phys={system_before['avail_phys_mb']:.1f}MB "
              f"avail_commit={system_before['avail_commit_mb']:.1f}MB "
              f"proc_ws={'n/a' if proc_ws is None else f'{proc_ws:.1f}MB'} "
              f"proc_commit={'n/a' if proc_commit is None else f'{proc_commit:.1f}MB'}")

    records: list[dict[str, Any]] = []
    for t in lengths:
        rec_sb, rec_rss = _recurrent_state_bytes(rec, t, warmup=args.warmup)
        rwkv_sb, rwkv_rss = _recurrent_state_bytes(rwkv, t, warmup=args.warmup)
        gpt_kv, gpt_attn, gpt_rss = _gpt_context(gpt, t, warmup=args.warmup)
        records.append({"T": t, "recurrent_state_bytes": rec_sb, "rwkv_state_bytes": rwkv_sb,
                        "gpt_kv_bytes": gpt_kv, "gpt_attn_bytes": gpt_attn,
                        "rss_mb": {"recurrent": round(rec_rss, 1), "rwkv": round(rwkv_rss, 1),
                                   "gpt": round(gpt_rss, 1)}})

    head = ("| T | Recurrent state (real) | RWKV state (real) | GPT KV-cache (calc) | "
            "GPT attn matrix (calc) | GPT fwd RSS Δ |")
    print("\n" + head)
    print("|" + "---|" * 6)
    for r in records:
        print(f"| {r['T']} | {r['recurrent_state_bytes']:,} B | {r['rwkv_state_bytes']:,} B | "
              f"{r['gpt_kv_bytes']:,} B | {r['gpt_attn_bytes']:,} B | {r['rss_mb']['gpt']} MB |")

    # Headline uses the measured min/max T range, even if the user requested
    # output rows in descending or mixed order.
    low_record, high_record = _headline_range(records)
    low_t = cast(int, low_record["T"])
    high_t = cast(int, high_record["T"])
    sb0 = cast(int, low_record["recurrent_state_bytes"])
    sbN = cast(int, high_record["recurrent_state_bytes"])
    kv0 = cast(int, low_record["gpt_kv_bytes"])
    kvN = cast(int, high_record["gpt_kv_bytes"])
    t_ratio = high_t / low_t
    print(f"\n[headline] T {low_t}→{high_t} (×{t_ratio:.2f}): "
          f"Recurrent state {sb0:,}→{sbN:,} B (×{sbN/max(1, sb0):.2f}=CONSTANT) / "
          f"GPT KV {kv0:,}→{kvN:,} B (×{kvN/max(1, kv0):.1f}=LINEAR) / GPT attn ×{(high_t/low_t)**2:.0f} (QUADRATIC). "
          f"state_bytes=実測テンソル / KV・attn=解析値 (RSS でトレンド裏取り)")
    print("[vm-note] pagefile / commit は速度向上ではなく OOM 回避の headroom 指標。"
          " avail_commit が小さいなら pagefile 設定や同時実行数を見直す。")

    system_after = _snapshot_summary(_system_memory_snapshot())
    if system_after is not None:
        proc_ws = system_after["process_working_set_mb"]
        proc_commit = system_after["process_pagefile_mb"]
        peak_proc_commit = system_after["process_peak_pagefile_mb"]
        print("[system] after: "
              f"load={system_after['memory_load_percent']:.0f}% "
              f"avail_phys={system_after['avail_phys_mb']:.1f}MB "
              f"avail_commit={system_after['avail_commit_mb']:.1f}MB "
              f"proc_ws={'n/a' if proc_ws is None else f'{proc_ws:.1f}MB'} "
              f"proc_commit={'n/a' if proc_commit is None else f'{proc_commit:.1f}MB'} "
              f"peak_proc_commit={'n/a' if peak_proc_commit is None else f'{peak_proc_commit:.1f}MB'}")

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": vars(args),
        "lengths_effective": lengths,
        "records": records,
        "system_before": system_before,
        "system_after": system_after,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
