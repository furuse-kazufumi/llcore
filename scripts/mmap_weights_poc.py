# SPDX-License-Identifier: Apache-2.0
"""mmap read-only 重み PoC — メモリ効率 pivot 第二歩 (a)。

llcore の北極星「メモリ使用効率(仮想メモリ含む)」(2026-06-16 pivot,
memory:project_llcore_memory_efficiency_pivot) の (a)。pivot memo の「✅ 本筋」は
*スワップに hot data を載せる* ことではなく、**working set を小さく・予測可能に**する
こと。その代表が **mmap read-only 重み**(OS ページキャッシュ=仮想メモリの正しい使い方
で、RAM 超のモデルを cold ページ disk 常駐のまま回す, llama.cpp 流)。

この PoC が実測で示すこと
-------------------------
1. **load 時 RSS**: ``torch.load(mmap=False)``(eager)は全重みを即座に anonymous メモリへ
   読み込むので RSS 増分 ≈ モデルサイズ。``torch.load(mmap=True)`` は file-backed で
   **遅延**し、load 直後の RSS 増分はごく小さい(=ヘッドライン)。
2. **on-demand fault**: mmap 後にテンソルを読む(全バイト touch)とページが順次 fault-in
   され RSS が伸びる = 「使った working set の分だけ」載る、を実機で確認。
3. **機能正当性**: ``load_state_dict(assign=True)`` で重みを mmap 実体のまま割り当てた
   モデルの forward が、eager ロード版と**同一 logits** を出す。

honest 留保
-----------
- 各モードは **別プロセス(subprocess)で隔離**して測る。torch の caching allocator や
  断片化が同一プロセス内では汚染するため、baseline → load → peak を fresh プロセスで取る。
- Windows の WorkingSet / PeakWorkingSet を WinAPI(psapi)で読む。非 Windows / 取得失敗時は
  0 を返し、その指標は「測れなかった」として扱う(本体は落とさない)。
- mmap の恩恵は「**部分 working set**・複数モデルでのページキャッシュ共有・コールド起動の
  遅延」。全重みを必ず一度に読むワークロードでは最終 RSS は eager に近づく(touch ステップが
  それを honest に示す)。「常に省メモリ」ではなく「**必要分だけ・遅延で**」が正確な主張。
- 真の RAM 超(モデル > 物理 RAM)の検証は別途大型モデルで要実測。ここでは「load 時に
  全載しない」性質を実機 RSS で示すまで。

使い方::

    py -3.11 scripts/mmap_weights_poc.py
    py -3.11 scripts/mmap_weights_poc.py --checkpoint out/lm_aozora_realp1/model.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from llcore.lm.model import CharGPT, GPTConfig
from llcore.runtime.rss import (
    peak_working_set_bytes as _peak_working_set_bytes,
    working_set_bytes as _working_set_bytes,
)

# Sentinel prefix the worker prints its single JSON result line with, so the
# parent can recover it even if torch emits warnings on stdout/stderr.
RESULT_PREFIX = "RESULT_JSON="


def _state_dict_bytes(state: dict[str, Tensor]) -> int:
    """state_dict 内テンソルの実バイト合計(numel × element_size)。"""
    return sum(t.numel() * t.element_size() for t in state.values())


def _load_model_state(path: Path, *, use_mmap: bool) -> dict[str, Tensor]:
    """checkpoint の ``model_state`` を eager / mmap で読む。

    ``weights_only=True`` で安全な型に限定しつつ、``mmap=use_mmap`` でテンソル storage の
    読み方(anonymous コピー vs file-backed 遅延)を切り替える。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=use_mmap)
    return cast("dict[str, Tensor]", ckpt["model_state"])


def _touch_all(state: dict[str, Tensor]) -> float:
    """全テンソルの全バイトを読み(reduction)、mmap ページを fault-in させる。

    返り値は副作用最適化を防ぐためのチェックサム(使い捨て)。
    """
    acc = 0.0
    for t in state.values():
        # to(float32) は f32 ではコピー無し。sum() が全要素を読む = 全ページ touch。
        acc += float(t.detach().to(torch.float32).sum().item())
    return acc


def run_worker(checkpoint: Path, *, use_mmap: bool, touch: bool) -> dict[str, Any]:
    """1 モードの RSS 計測(この関数は隔離 subprocess 内で呼ばれる想定)。"""
    file_bytes = checkpoint.stat().st_size
    gc.collect()
    baseline = _working_set_bytes()
    state = _load_model_state(checkpoint, use_mmap=use_mmap)
    post_load = _working_set_bytes()
    post_load_peak = _peak_working_set_bytes()
    result: dict[str, Any] = {
        "mode": "mmap" if use_mmap else "eager",
        "file_bytes": file_bytes,
        "n_state_tensors": len(state),
        "state_param_mb": round(_state_dict_bytes(state) / 1e6, 2),
        "baseline_ws_mb": round(baseline / 1e6, 2),
        "post_load_ws_mb": round(post_load / 1e6, 2),
        "post_load_delta_mb": round(max(0, post_load - baseline) / 1e6, 2),
        "post_load_peak_mb": round(post_load_peak / 1e6, 2),
    }
    if touch:
        _touch_all(state)
        post_touch = _working_set_bytes()
        result["post_touch_ws_mb"] = round(post_touch / 1e6, 2)
        result["post_touch_delta_mb"] = round(max(0, post_touch - baseline) / 1e6, 2)
    return result


def _spawn_worker(checkpoint: Path, *, use_mmap: bool, touch: bool) -> dict[str, Any]:
    """fresh subprocess で run_worker を実行し、RESULT_JSON 行を回収する。

    allocator/断片化の汚染を避けるため計測は必ず別プロセスで行う。
    """
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "mmap" if use_mmap else "eager",
        "--checkpoint", str(checkpoint),
    ]
    if touch:
        cmd.append("--touch")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}")
    # Recover the single sentinel-prefixed JSON line (torch warnings may precede it).
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return cast("dict[str, Any]", json.loads(line[len(RESULT_PREFIX):]))
    raise RuntimeError(f"worker produced no {RESULT_PREFIX} line; stdout={proc.stdout[:400]!r}")


def functional_check(checkpoint: Path) -> dict[str, Any]:
    """eager コピー版と mmap assign 版の forward logits が一致することを確認する。"""
    ckpt_eager = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=False)
    cfg = GPTConfig(**ckpt_eager["config"])
    model_eager = CharGPT(cfg)
    model_eager.load_state_dict(ckpt_eager["model_state"])  # 通常コピー
    model_eager.eval()

    # mmap 実体をそのままパラメータに割り当てる(assign=True ならコピーしない = llama.cpp 流)。
    state_mmap = _load_model_state(checkpoint, use_mmap=True)
    model_mmap = CharGPT(cfg)
    model_mmap.load_state_dict(state_mmap, assign=True)
    model_mmap.eval()

    # 固定入力で両者の logits を比較(eval/no_grad で決定的)。
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (1, min(8, cfg.block_size)))
    with torch.no_grad():
        logits_eager = model_eager.forward_logits(idx)
        logits_mmap = model_mmap.forward_logits(idx)
    max_abs_diff = float((logits_eager - logits_mmap).abs().max().item())
    return {"functional_match": bool(max_abs_diff == 0.0), "max_abs_logit_diff": max_abs_diff}


def _build_worker_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="mmap read-only weights PoC: load-time RSS")
    ap.add_argument("--checkpoint", default="out/lm_aozora_realp1/model.pt")
    ap.add_argument("--json", default="out/mmap_weights_poc.json")
    # --worker switches the process into the isolated measurement role.
    ap.add_argument("--worker", choices=["eager", "mmap"], default=None)
    ap.add_argument("--touch", action="store_true", help="fault in all pages after load")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_worker_parser().parse_args(argv)
    checkpoint = Path(args.checkpoint)

    # --- worker role: measure one mode and print the RESULT_JSON sentinel line ---
    if args.worker is not None:
        if not checkpoint.exists():
            print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
            return 2
        result = run_worker(checkpoint, use_mmap=args.worker == "mmap", touch=args.touch)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0

    # --- parent role: spawn isolated workers, run functional check, summarize ---
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    eager = _spawn_worker(checkpoint, use_mmap=False, touch=True)
    mmap_res = _spawn_worker(checkpoint, use_mmap=True, touch=True)
    func = functional_check(checkpoint)

    file_mb = round(checkpoint.stat().st_size / 1e6, 2)
    print(
        f"checkpoint: {checkpoint}  file={file_mb} MB  "
        f"state_tensors={eager['n_state_tensors']}  "
        f"param_bytes={eager['state_param_mb']} MB"
    )
    print("\n| mode | load ΔRSS | load peak | after-touch ΔRSS |")
    print("|" + "---|" * 4)
    for rec in (eager, mmap_res):
        print(
            f"| {rec['mode']} | {rec['post_load_delta_mb']} MB | "
            f"{rec['post_load_peak_mb']} MB | "
            f"{rec.get('post_touch_delta_mb', 'n/a')} MB |"
        )

    # Headline: load-time RSS. mmap defers; eager materializes immediately.
    eager_load = eager["post_load_delta_mb"]
    mmap_load = mmap_res["post_load_delta_mb"]
    ratio = mmap_load / eager_load if eager_load > 0 else float("nan")
    print(
        f"\n[headline] load 時 ΔRSS: eager {eager_load} MB(≈モデル {eager['state_param_mb']} MB を即全載) "
        f"vs mmap {mmap_load} MB(遅延=×{ratio:.3f})。"
        f"touch 後は mmap も {mmap_res.get('post_touch_delta_mb')} MB へ = 使った分だけ fault-in。"
    )
    print(
        f"[functional] mmap(assign=True) と eager の logits 一致: "
        f"{func['functional_match']}(max|Δ|={func['max_abs_logit_diff']:.2e})"
    )
    print(
        "[honest] RSS は別プロセス隔離計測。恩恵は部分 working set / ページキャッシュ共有 / "
        "コールド起動遅延。全載ワークロードでは最終 RSS は eager に近づく(touch 行が示す)。"
    )

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "file_bytes": checkpoint.stat().st_size,
        "eager": eager,
        "mmap": mmap_res,
        "functional": func,
        "load_delta_ratio_mmap_over_eager": round(ratio, 4) if eager_load > 0 else None,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
