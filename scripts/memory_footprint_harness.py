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
    と KV/attn (解析値)。RSS は補助。

使い方::

    py -3.11 scripts/memory_footprint_harness.py
    py -3.11 scripts/memory_footprint_harness.py --lengths 64,128,256,512,1024 --json out/mem_footprint.json
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
from pathlib import Path

import torch

from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM


def _working_set_bytes() -> int:
    """現在のプロセスの working set (RSS) バイト数を返す (Windows; 失敗時 0)。"""
    try:
        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        h = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)  # type: ignore[attr-defined]
        return int(c.WorkingSetSize) if ok else 0
    except Exception:  # noqa: BLE001 - RSS は補助情報、取れなくても続行
        return 0


def _rss_delta(fn) -> tuple[int, float]:
    """fn() を実行し、(結果, RSS増分MB) を返す (gc を挟んでノイズを抑える)。"""
    gc.collect()
    before = _working_set_bytes()
    out = fn()
    after = _working_set_bytes()
    del out
    return after, max(0.0, (after - before) / 1e6)


@torch.no_grad()
def _recurrent_state_bytes(model: RecurrentLM | RWKVLM, t: int) -> tuple[int, float]:
    """T 個のトークンを step ループで流し、最終状態の state_bytes と RSS 増分を返す。"""
    model.eval()
    idx = torch.randint(0, model.config.vocab_size, (1,))

    def run() -> int:
        state = None
        sb = 0
        for _ in range(t):
            _, state = model.step(idx, state)
            sb = model.state_bytes(state)  # T に依らず一定のはず
        return sb

    gc.collect()
    before = _working_set_bytes()
    sb = run()
    after = _working_set_bytes()
    return sb, max(0.0, (after - before) / 1e6)


@torch.no_grad()
def _gpt_context(model: CharGPT, t: int) -> tuple[int, int, float]:
    """T トークンの実 forward を走らせ、(KV-cache バイト解析値, attn 行列バイト解析値, RSS増分) を返す。"""
    model.eval()
    cfg = model.config
    kv_bytes = 2 * cfg.n_layer * t * cfg.n_embd * 4  # k,v それぞれ [L,T,C] float32
    attn_bytes = cfg.n_head * t * t * 4              # 1 層あたり attention 行列 [nh,T,T] (transient)
    idx = torch.randint(0, cfg.vocab_size, (1, t))

    def run() -> torch.Tensor:
        logits, _ = model(idx)
        return logits

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
    args = ap.parse_args(argv)
    torch.manual_seed(1337)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    max_t = max(lengths)

    gpt = CharGPT(GPTConfig(vocab_size=args.vocab, block_size=max_t, n_layer=args.n_layer,
                            n_head=args.n_head, n_embd=args.n_embd))
    rec = RecurrentLM(RecurrentConfig(vocab_size=args.vocab, block_size=max_t, n_layer=args.n_layer,
                                      n_embd=args.n_embd, state_size=args.n_embd))
    rwkv = RWKVLM(RWKVConfig(vocab_size=args.vocab, block_size=max_t, n_layer=args.n_layer,
                             n_embd=args.n_embd))
    print(f"models: GPT {gpt.num_params(False):,} / Recurrent {rec.num_params():,} / "
          f"RWKV {rwkv.num_params():,} params  (n_embd={args.n_embd} L={args.n_layer})")

    records = []
    for t in lengths:
        rec_sb, rec_rss = _recurrent_state_bytes(rec, t)
        rwkv_sb, rwkv_rss = _recurrent_state_bytes(rwkv, t)
        gpt_kv, gpt_attn, gpt_rss = _gpt_context(gpt, t)
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

    # headline: state constancy vs GPT growth ratio across the T range
    sb0, sbN = records[0]["recurrent_state_bytes"], records[-1]["recurrent_state_bytes"]
    kv0, kvN = records[0]["gpt_kv_bytes"], records[-1]["gpt_kv_bytes"]
    print(f"\n[headline] T {lengths[0]}→{lengths[-1]} ({lengths[-1]//lengths[0]}×): "
          f"Recurrent state {sb0:,}→{sbN:,} B (×{sbN/sb0:.2f}=CONSTANT) / "
          f"GPT KV {kv0:,}→{kvN:,} B (×{kvN/kv0:.1f}=LINEAR) / GPT attn ×{(lengths[-1]/lengths[0])**2:.0f} (QUADRATIC). "
          f"state_bytes=実測テンソル / KV・attn=解析値 (RSS でトレンド裏取り)")

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"config": vars(args), "records": records}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
