# SPDX-License-Identifier: Apache-2.0
"""RAM 超 × mmap 実証 — メモリ効率 pivot の北極星「仮想メモリ含む」の本丸。

memory:project_llcore_memory_efficiency_pivot の核心主張 = 「mmap read-only 重み(OS ページ
キャッシュ)= 仮想メモリの正しい使い方で、RAM 超モデルを cold ページ disk 常駐のまま回す
(llama.cpp 流)」。`scripts/mmap_weights_poc.py` は **load 時の遅延**までを示したが、フル forward は
全重みを touch するため、メモリ圧力が無ければ RSS は最終的にモデルサイズへ収束する(touch 行が示した)。

本 PoC が新たに実証すること
---------------------------
**Windows の working-set 上限(`SetProcessWorkingSetSizeEx` hard max)を「モデルサイズ未満」に
設定したサブプロセスで forward を完走させ、peak working set ≤ 上限 を実測する。** read-only な
mmap ページは clean なのでページアウト = 破棄(pagefile 書込み不要)、再 fault で disk から読み直す。
よって **working set(= 使える物理 RAM)がモデルより小さくても、モデルは動く**。これが「RAM 超で
回る」の機構そのもの。さらに int8 量子化でディスク footprint を ~4x 縮小できることも併せて示す。

honest 留保
-----------
- このマシンは avail RAM が限られるため、**物理 RAM 総量を超える巨大モデルを literally 作るのではなく**、
  「working-set 上限 < モデルサイズ」で同じ性質(使える RAM < モデルでも動く)を実証する。RAM 総量を
  超える超大型モデルでの検証は GPU/大 RAM 環境での将来課題。
- `SetProcessWorkingSetSizeEx` の hard-max が OS により厳密強制されるかは環境依存。**強制可否と実測
  peak WS を honest に報告**し、強制されなければそう書く(成功を偽装しない)。
- int8 はディスク footprint と mmap-load RSS まで(per-layer streaming dequant forward は将来課題)。
- 重みはランダム初期化(訓練不要)。メモリ挙動は学習済みと同一なので PoC として妥当。

使い方::

    py -3.11 scripts/mmap_ram_exceed_poc.py          # 既定 ~500MB モデルで実証
    py -3.11 scripts/mmap_ram_exceed_poc.py --n-layer 8 --cap-mb 320
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

from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]

RESULT_PREFIX = "RESULT_JSON="
# SetProcessWorkingSetSizeEx flags (winnt.h): hard-min off, hard-max on.
QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x00000004


class _PMC(ctypes.Structure):
    # PROCESS_MEMORY_COUNTERS — mirrors the other memory harnesses for consistency.
    _fields_ = [
        ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _process_memory_info() -> _PMC | None:
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
        return c if ok else None
    except Exception:  # noqa: BLE001 - RSS は補助指標、取れなくても続行
        return None


def _working_set_bytes() -> int:
    info = _process_memory_info()
    return int(info.WorkingSetSize) if info is not None else 0


def _peak_working_set_bytes() -> int:
    info = _process_memory_info()
    return int(info.PeakWorkingSetSize) if info is not None else 0


def _avail_phys_bytes() -> int:
    """利用可能な物理メモリ(Windows; 失敗時 0)。モデルサイズ安全弁に使う。"""
    try:
        import ctypes.wintypes as wt

        class _MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MS)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(ms)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return 0
        return int(ms.ullAvailPhys)
    except Exception:  # noqa: BLE001 - 取れなくても本体は続行
        return 0


def _set_working_set_cap(max_bytes: int) -> bool:
    """プロセスの working set に **hard max** を課す(成功なら True)。

    read-only mmap ページは clean なので、上限超過時に OS が破棄→再 fault でき、
    「使える物理 RAM < モデルサイズ」でも forward を完走できる、を成立させる仕掛け。
    min は max の半分にし、HARDWS_MIN_DISABLE で下限は緩める。
    """
    try:
        import ctypes.wintypes as wt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        kernel32.SetProcessWorkingSetSizeEx.argtypes = [
            wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD
        ]
        kernel32.SetProcessWorkingSetSizeEx.restype = wt.BOOL
        flags = QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE
        ok = kernel32.SetProcessWorkingSetSizeEx(
            kernel32.GetCurrentProcess(), max_bytes // 2, max_bytes, flags
        )
        return bool(ok)
    except Exception:  # noqa: BLE001 - 強制不可なら honest に False を返す
        return False


def quantize_per_channel_int8(w: Tensor) -> tuple[Tensor, Tensor]:
    """per-channel(行ごと)対称 int8 量子化(int8_quant_footprint.py と同方式)。"""
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def build_large_model(cfg: GPTConfig, seed: int = 1234) -> CharGPT:
    """大型ランダム CharGPT を生成(訓練不要=メモリ挙動は学習済みと同一)。"""
    torch.manual_seed(seed)
    model = CharGPT(cfg)
    model.eval()
    return model


def _param_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total


def save_int8_checkpoint(model: CharGPT, path: Path) -> int:
    """2-D 重みを int8 量子化してディスク保存し、実ファイルバイト数を返す。

    ディスク footprint が fp32 の ~1/4 になることを実測するための保存(streaming
    dequant forward は将来課題)。1-D params は fp32 据え置き。
    """
    state: dict[str, Any] = {"config": vars(model.config), "kind": "int8_per_channel"}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.dim() == 2:
            q, scale = quantize_per_channel_int8(p.data)
            state[f"q::{name}"] = q
            state[f"s::{name}"] = scale
        else:
            state[f"f::{name}"] = p.data
    torch.save(state, path)
    return path.stat().st_size


def run_worker(checkpoint: Path, cap_bytes: int | None) -> dict[str, Any]:
    """mmap-load + forward を実行し WS を実測(隔離 subprocess 内で呼ばれる想定)。

    cap_bytes 指定時は load 前に working-set hard max を課す。
    """
    file_bytes = checkpoint.stat().st_size
    gc.collect()
    baseline = _working_set_bytes()
    cap_set = False
    if cap_bytes is not None:
        cap_set = _set_working_set_cap(cap_bytes)

    # mmap(file-backed)+ assign=True で重みをコピーせず割り当てる。
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    cfg = GPTConfig(**ckpt["config"])
    model = CharGPT(cfg)
    model.load_state_dict(ckpt["model_state"], assign=True)
    model.eval()
    post_load = _working_set_bytes()

    # forward(全重みを touch する=メモリ圧力の本番)。決定的入力で checksum を取る。
    torch.manual_seed(0)
    t = min(64, cfg.block_size)
    idx = torch.randint(0, cfg.vocab_size, (1, t))
    with torch.no_grad():
        logits = model.forward_logits(idx)
    checksum = float(logits.double().sum().item())
    peak = _peak_working_set_bytes()
    return {
        "mode": "capped" if cap_bytes is not None else "uncapped",
        "cap_mb": round(cap_bytes / 1e6, 1) if cap_bytes is not None else None,
        "cap_set_ok": cap_set,
        "file_mb": round(file_bytes / 1e6, 1),
        "baseline_ws_mb": round(baseline / 1e6, 1),
        "post_load_ws_mb": round(post_load / 1e6, 1),
        "post_load_delta_mb": round(max(0, post_load - baseline) / 1e6, 1),
        "peak_ws_mb": round(peak / 1e6, 1),
        "checksum": checksum,
    }


def _spawn_worker(checkpoint: Path, cap_bytes: int | None) -> dict[str, Any]:
    """fresh subprocess で run_worker を実行し RESULT_JSON 行を回収する。"""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--checkpoint", str(checkpoint)]
    if cap_bytes is not None:
        cmd += ["--cap-bytes", str(cap_bytes)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return cast("dict[str, Any]", json.loads(line[len(RESULT_PREFIX):]))
    raise RuntimeError(f"worker produced no {RESULT_PREFIX} line; stdout={proc.stdout[:400]!r}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RAM-exceeding mmap PoC: run a model with WS < model size")
    ap.add_argument("--n-embd", type=int, default=1024)
    ap.add_argument("--n-layer", type=int, default=10)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--cap-mb", type=float, default=None, help="working-set hard max (MB); 既定=baseline+margin")
    ap.add_argument("--out-dir", default="out/mmap_ram_exceed")
    ap.add_argument("--json", default="out/mmap_ram_exceed_poc.json")
    # worker role
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cap-bytes", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # --- worker role: measure one (un)capped run, print RESULT_JSON ---
    if args.worker:
        if not args.checkpoint:
            print("error: --worker requires --checkpoint", file=sys.stderr)
            return 2
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 2
        result = run_worker(ckpt, args.cap_bytes)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0

    # --- parent role ---
    cfg = GPTConfig(vocab_size=args.vocab, block_size=args.block, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_large_model(cfg)
    model_bytes = _param_bytes(model)
    avail = _avail_phys_bytes()
    # 安全弁: モデルが利用可能物理 RAM の 40% を超えるなら止める(実 RAM を食い潰さない)。
    if avail > 0 and model_bytes > 0.40 * avail:
        print(
            f"error: model {model_bytes/1e6:.0f} MB exceeds 40% of avail RAM {avail/1e6:.0f} MB; "
            f"reduce --n-layer/--n-embd",
            file=sys.stderr,
        )
        return 2

    fp32_path = out_dir / "model_fp32.pt"
    n_params = sum(p.numel() for p in model.parameters())
    torch.save({"config": vars(cfg), "model_state": model.state_dict()}, fp32_path)
    fp32_file = fp32_path.stat().st_size
    int8_path = out_dir / "model_int8.pt"
    int8_file = save_int8_checkpoint(model, int8_path)

    print(
        f"model: {n_params:,} params  param_bytes={model_bytes/1e6:.1f} MB  "
        f"fp32_file={fp32_file/1e6:.1f} MB  int8_file={int8_file/1e6:.1f} MB "
        f"(int8/fp32={int8_file/fp32_file:.3f})  avail_RAM={avail/1e6:.0f} MB"
    )
    # model を解放してから子プロセス計測(親の RSS を子に持ち込まない)。
    del model
    gc.collect()

    uncapped = _spawn_worker(fp32_path, None)
    # cap 既定 = uncapped baseline + 160MB、かつ fp32 file の 75% 未満(= モデル未満を保証)。
    if args.cap_mb is not None:
        cap_bytes = int(args.cap_mb * 1e6)
    else:
        cap_bytes = int(min(uncapped["baseline_ws_mb"] + 160, fp32_file / 1e6 * 0.75) * 1e6)
    capped = _spawn_worker(fp32_path, cap_bytes)

    checksum_match = uncapped["checksum"] == capped["checksum"]
    print("\n| mode | cap | load ΔRSS | peak WS | checksum |")
    print("|" + "---|" * 5)
    for rec in (uncapped, capped):
        cap = "none" if rec["cap_mb"] is None else f"{rec['cap_mb']}MB({'set' if rec['cap_set_ok'] else 'NOT set'})"
        print(f"| {rec['mode']} | {cap} | {rec['post_load_delta_mb']} MB | {rec['peak_ws_mb']} MB | {rec['checksum']:.4g} |")

    ran_below_model = capped["peak_ws_mb"] < fp32_file / 1e6
    print(
        f"\n[headline] fp32 モデル {fp32_file/1e6:.0f} MB を、working-set 上限 {cap_bytes/1e6:.0f} MB "
        f"(< モデル)で forward 完走。capped peak WS = {capped['peak_ws_mb']} MB "
        f"{'< モデルサイズ ✓(RAM 超で回る機構を実証)' if ran_below_model else '≥ モデル(cap 強制されず=honest 記録)'}。"
        f" int8 でディスクは {int8_file/fp32_file:.2f}x に縮小。"
    )
    print(
        f"[functional] capped と uncapped の logits checksum 一致: {checksum_match}。"
    )
    print(
        "[honest] cap_set_ok と実測 peak WS をそのまま報告(強制不可ならそう記録)。"
        " 真の RAM 総量超は GPU/大 RAM 環境で要検証。int8 は disk/load まで(forward dequant は将来)。"
    )

    payload: dict[str, Any] = {
        "config": vars(cfg),
        "n_params": n_params,
        "param_bytes": model_bytes,
        "fp32_file_bytes": fp32_file,
        "int8_file_bytes": int8_file,
        "int8_over_fp32": round(int8_file / fp32_file, 4),
        "avail_phys_bytes": avail,
        "cap_bytes": cap_bytes,
        "uncapped": uncapped,
        "capped": capped,
        "checksum_match": checksum_match,
        "capped_peak_below_model": ran_below_model,
    }
    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
