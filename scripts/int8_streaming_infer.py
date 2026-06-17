# SPDX-License-Identifier: Apache-2.0
"""int8 streaming-dequant 推論 — (a') mmap + (b) int8 を実推論パスへ統合。

メモリ効率 pivot の成長要素。これまで int8 は **storage(footprint)圧縮**まで、mmap は
**load 遅延 / working-set 上限下の完走**まで示した。本スクリプトは両者を**実際の推論パス**へ
統合する: 重みを int8 のまま resident に持ち、forward の中で**層ごとに fp32 へ dequant し
即解放**する(`Int8Linear`)。これにより fp32 の同時 materialize は **最大 1 層分**に抑えられ、
推論時の peak working set がモデルサイズの数分の一になる。

実証する 2 点(別プロセス隔離計測)
-----------------------------------
1. **メモリ削減**: 同一 int8 ソースから、(dense) 全層を一括 dequant して常駐 vs (stream) 層ごと
   dequant の peak working set を比較。stream は **int8 常駐(~1/4)+ 最大 1 層の fp32**に収まる。
2. **正当性**: dense と stream の出力 logits は **完全一致**(同じ int8・同じ算術、dequant の
   タイミングだけが違う)= メモリ最適化が結果を変えないことを示す(量子化誤差は別問題で (b) で測定済み)。

honest 留保
-----------
- 量子化対象は **`nn.Linear`(transformer ブロック + lm_head)** に限定。`nn.Embedding`(wte)と
  LayerNorm は fp32 据え置き(大型モデルでは Linear が大半=支配的)。lm_head を Int8Linear 化すると
  tied wte との重み共有は切れるが、dense/stream は同一 int8 を使うので両者は一致する。
- これは **simulated quant**(int8 を fp32 へ戻して matmul)。真の int8 GEMM(速度)は GPU 課題。
  ここで測るのは **推論時 working set**(メモリ)であって速度ではない。
- torch caching allocator の都合で peak は層 fp32 の単純合計でなく「最大層 + 再利用」に収束する想定。
  実測値をそのまま報告する。

使い方::

    py -3.11 scripts/int8_streaming_infer.py            # 既定 ~125M params で実証
    py -3.11 scripts/int8_streaming_infer.py --n-layer 8
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]

RESULT_PREFIX = "RESULT_JSON="


class _PMC(ctypes.Structure):
    # PROCESS_MEMORY_COUNTERS — mirrors the other memory harnesses.
    _fields_ = [
        ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _peak_mem() -> tuple[int, int]:
    """(peak working set bytes, peak pagefile bytes) を返す(失敗時 (0,0))。

    pagefile は「ダーティな anonymous ページの退避量」。stream は重みを int8(mmap で
    file-backed=破棄可)に保つので圧力下でも pagefile が小さく、dense は dequant した
    anonymous fp32 を退避するので大きい、という差を見るために併せて測る。
    """
    try:
        import ctypes.wintypes as wt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return (int(c.PeakWorkingSetSize), int(c.PeakPagefileUsage)) if ok else (0, 0)
    except Exception:  # noqa: BLE001 - RSS は補助指標、取れなくても続行
        return (0, 0)


def _set_working_set_cap(max_bytes: int) -> bool:
    """プロセスの working set に hard max を課す(mmap_ram_exceed_poc.py と同方式)。"""
    try:
        import ctypes.wintypes as wt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        kernel32.SetProcessWorkingSetSizeEx.argtypes = [
            wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD
        ]
        kernel32.SetProcessWorkingSetSizeEx.restype = wt.BOOL
        # QUOTA_LIMITS_HARDWS_MIN_DISABLE(0x2) | QUOTA_LIMITS_HARDWS_MAX_ENABLE(0x4)
        ok = kernel32.SetProcessWorkingSetSizeEx(
            kernel32.GetCurrentProcess(), max_bytes // 2, max_bytes, 0x2 | 0x4
        )
        return bool(ok)
    except Exception:  # noqa: BLE001 - 強制不可なら honest に False を返す
        return False


def quantize_per_channel_int8(w: Tensor) -> tuple[Tensor, Tensor]:
    """per-channel(行ごと)対称 int8 量子化(他スクリプトと同方式)。"""
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


class Int8Linear(nn.Module):
    """nn.Linear の drop-in 置換。int8 重みを resident に持ち forward 内で dequant する。

    ``qweight``(int8)と ``scale``(fp32 行ごと)は buffer。forward では一時 fp32 重みを
    生成→matmul→スコープ離脱で解放するため、fp32 の同時常駐は **この 1 層分**で済む。
    """

    qweight: Tensor
    scale: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> Int8Linear:
        """既存 nn.Linear を量子化して Int8Linear を作る。"""
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None)
        q, scale = quantize_per_channel_int8(lin.weight.data)
        m.qweight.copy_(q)
        m.scale.copy_(scale)
        if lin.bias is not None and m.bias is not None:
            m.bias.data.copy_(lin.bias.data)
        return m

    def forward(self, x: Tensor) -> Tensor:
        # 一時 fp32 重み: この行のスコープを抜けると解放される = 同時常駐は 1 層分。
        w = self.qweight.to(torch.float32) * self.scale
        return cast(Tensor, F.linear(x, w, self.bias))

    def to_fp32_linear(self) -> nn.Linear:
        """dense モード用: dequant した通常の nn.Linear を返す(fp32 を常駐させる)。"""
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        lin.weight.data = self.qweight.to(torch.float32) * self.scale
        if self.bias is not None and lin.bias is not None:
            lin.bias.data.copy_(self.bias.data)
        return lin


def _replace_modules(module: nn.Module, factory: Any, target: type) -> None:
    """``module`` 内の ``target`` 型の子を ``factory(child)`` で再帰置換する(in-place)。"""
    for name, child in list(module.named_children()):
        if isinstance(child, target):
            setattr(module, name, factory(child))
        else:
            _replace_modules(child, factory, target)


def convert_linears_to_int8(model: nn.Module) -> nn.Module:
    """model 内の全 nn.Linear を Int8Linear へ置換(int8 常駐・stream 推論用)。"""
    _replace_modules(model, Int8Linear.from_linear, nn.Linear)
    return model


def convert_int8_to_dense(model: nn.Module) -> nn.Module:
    """model 内の全 Int8Linear を fp32 nn.Linear へ戻す(全層 materialize・dense 用)。"""
    _replace_modules(model, lambda m: m.to_fp32_linear(), Int8Linear)
    return model


def _resident_bytes(model: nn.Module) -> int:
    """params + buffers の常駐バイト合計(int8 は 1B/要素で軽い)。"""
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def _save_int8_model(cfg: GPTConfig, path: Path, seed: int = 1234) -> dict[str, int]:
    """大型ランダム CharGPT を int8(Linear 置換)化して保存し、サイズ情報を返す。"""
    torch.manual_seed(seed)
    fp32 = CharGPT(cfg)
    fp32_bytes = _resident_bytes(fp32)
    int8_model = convert_linears_to_int8(fp32)  # fp32 を破壊的に置換(メモリ節約)
    int8_bytes = _resident_bytes(int8_model)
    torch.save({"config": vars(cfg), "model_state": int8_model.state_dict()}, path)
    del int8_model
    gc.collect()
    return {"fp32_bytes": fp32_bytes, "int8_bytes": int8_bytes, "file_bytes": path.stat().st_size}


def _build_int8_skeleton(cfg: GPTConfig) -> nn.Module:
    """形状だけ合った Int8Linear 骨格(load_state_dict で int8 を流し込む先)。"""
    model = CharGPT(cfg)
    convert_linears_to_int8(model)
    return model


@torch.no_grad()
def run_worker(checkpoint: Path, mode: str) -> dict[str, Any]:
    """1 モード(dense / stream)で mmap-load + forward し peak WS と checksum を返す。"""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = GPTConfig(**ckpt["config"])
    model = _build_int8_skeleton(cfg)
    model.load_state_dict(ckpt["model_state"], assign=True)  # int8 を mmap 実体で割当
    if mode == "dense":
        convert_int8_to_dense(model)  # 全層を fp32 へ materialize(常駐 ≈ 全モデル)
    model.eval()
    resident = _resident_bytes(model)

    torch.manual_seed(0)
    t = min(64, cfg.block_size)
    idx = torch.randint(0, cfg.vocab_size, (1, t))
    logits = cast(CharGPT, model).forward_logits(idx)
    checksum = float(logits.double().sum().item())
    peak = _peak_working_set_bytes()
    return {
        "mode": mode,
        "resident_mb": round(resident / 1e6, 1),
        "peak_ws_mb": round(peak / 1e6, 1),
        "checksum": checksum,
    }


def _spawn_worker(checkpoint: Path, mode: str) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
           "--checkpoint", str(checkpoint)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return cast("dict[str, Any]", json.loads(line[len(RESULT_PREFIX):]))
    raise RuntimeError(f"worker produced no {RESULT_PREFIX} line; stdout={proc.stdout[:400]!r}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="int8 streaming-dequant inference: peak WS dense vs stream")
    ap.add_argument("--n-embd", type=int, default=1024)
    ap.add_argument("--n-layer", type=int, default=10)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--out-dir", default="out/int8_streaming")
    ap.add_argument("--json", default="out/int8_streaming_infer.json")
    ap.add_argument("--worker", choices=["dense", "stream"], default=None)
    ap.add_argument("--checkpoint", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # --- worker role ---
    if args.worker is not None:
        if not args.checkpoint or not Path(args.checkpoint).exists():
            print(f"error: --worker needs an existing --checkpoint: {args.checkpoint}", file=sys.stderr)
            return 2
        result = run_worker(Path(args.checkpoint), args.worker)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0

    # --- parent role ---
    cfg = GPTConfig(vocab_size=args.vocab, block_size=args.block, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model_int8.pt"
    sizes = _save_int8_model(cfg, ckpt)

    print(
        f"model: cfg n_embd={cfg.n_embd} L={cfg.n_layer} vocab={cfg.vocab_size}  "
        f"fp32 weights={sizes['fp32_bytes']/1e6:.1f} MB  int8 resident={sizes['int8_bytes']/1e6:.1f} MB "
        f"(={sizes['int8_bytes']/sizes['fp32_bytes']:.3f})  int8 file={sizes['file_bytes']/1e6:.1f} MB"
    )

    dense = _spawn_worker(ckpt, "dense")
    stream = _spawn_worker(ckpt, "stream")
    checksum_match = dense["checksum"] == stream["checksum"]

    print("\n| mode | resident | peak WS | checksum |")
    print("|" + "---|" * 4)
    for rec in (dense, stream):
        print(f"| {rec['mode']} | {rec['resident_mb']} MB | {rec['peak_ws_mb']} MB | {rec['checksum']:.4g} |")

    reduction = (1.0 - stream["peak_ws_mb"] / dense["peak_ws_mb"]) * 100 if dense["peak_ws_mb"] else 0.0
    print(
        f"\n[headline] 同一 int8 ソースから forward。dense(全層 materialize)peak WS "
        f"{dense['peak_ws_mb']} MB vs stream(層ごと dequant)peak WS {stream['peak_ws_mb']} MB "
        f"= **{reduction:.0f}% 削減**。stream 常駐は int8({sizes['int8_bytes']/1e6:.0f} MB)+ 最大1層 fp32。"
    )
    print(f"[functional] dense と stream の logits checksum 一致: {checksum_match}(= メモリ最適化は結果不変)。")
    print(
        "[honest] 量子化は nn.Linear のみ(Embedding/LN は fp32)。simulated quant=速度未測。"
        " peak WS は torch allocator 込みの実測値。"
    )

    payload: dict[str, Any] = {
        "config": vars(cfg),
        "fp32_weight_bytes": sizes["fp32_bytes"],
        "int8_resident_bytes": sizes["int8_bytes"],
        "int8_file_bytes": sizes["file_bytes"],
        "dense": dense,
        "stream": stream,
        "checksum_match": checksum_match,
        "peak_ws_reduction_pct": round(reduction, 1),
    }
    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
