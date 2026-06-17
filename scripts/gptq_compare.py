# SPDX-License-Identifier: Apache-2.0
"""GPTQ vs RTN 量子化比較 — 「2bit は RTN 超が必要」を直接検証(#3 CPU スライス)。

`quant_group_compare.py` の結論は「per-group RTN は低ビット品質を改善するが、strict cap-gate(top1
fp32 比 97% 保持)は 2bit では越えられない=RTN 超(誤差補償 or QAT)が必要」。本スクリプトは
**GPTQ(Frantar et al. 2022, 出力誤差 ‖(W−Ŵ)X‖² を入力 Hessian で最小化する誤差補償量子化)** を実装し、
RTN per-channel と同条件で held-out PPL + top-1(cap-gate)を比較する。

手順
----
1. 学習済み checkpoint をロードし、train split の一部を **校正データ**にする。
2. forward hook で各 `nn.Linear` の入力を捕捉し、入力 Hessian ``H = Σ xᵀx`` を蓄積。
3. 各 Linear を (a) RTN per-channel と (b) GPTQ(H 使用)で量子化。
4. 同一 held-out split で両者の PPL / top1 / gate を比較。**GPTQ が 2bit で cap-gate を越えるか**が眼目。

気付き(probe で確認済み)
--------------------------
GPTQ は **weight 誤差は RTN より大きくなり得るのに output 誤差は小さい** — ‖W−Ŵ‖² でなく ‖(W−Ŵ)X‖²
を最小化するため。「重みをわざと不正確にして出力を正確にする」誤差補償の本質。

honest 留保
-----------
- 量子化対象は `nn.Linear` のみ(Embedding/LN は両者 fp32)= RTN/GPTQ の比較を Linear 層で apples-to-apples
  にするため。footprint は per-channel scale 前提(GPTQ も per-row scale を原重みから固定)。
- weights-only / dequant fp32 forward の simulated quant(速度未測)。校正は train split の限定窓。

使い方::

    py -3.11 scripts/gptq_compare.py \
        --checkpoint out/lm_aozora_realp1/model.pt --corpus-file out/corpus_aozora.txt --bits 3,2
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from llcore.lm.data import train_val_split  # type: ignore[import-untyped]
from llcore.lm.eval import (  # type: ignore[import-untyped]
    held_out_report_any,
    held_out_top1_report,
    passes_capability_gate,
    passes_gate,
)
from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]
from llcore.lm.tokenizer import CharTokenizer  # type: ignore[import-untyped]


def _per_row_scale(w: Tensor, bits: int) -> Tensor:
    qmax = (1 << (bits - 1)) - 1
    return (w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax)


def quantize_rtn(w: Tensor, bits: int) -> Tensor:
    """per-channel(行ごと)対称 RTN。dequantized 重みを返す(GPTQ の baseline)。"""
    qmax = (1 << (bits - 1)) - 1
    scale = _per_row_scale(w, bits)
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def quantize_gptq(w: Tensor, h: Tensor, bits: int, blocksize: int = 128, percdamp: float = 0.01) -> Tensor:
    """GPTQ 誤差補償量子化。``(W, H)`` から dequantized 重みを返す。

    出力誤差 ‖(W−Ŵ)X‖² を最小化するよう、列を順に量子化しながら残差誤差を未量子化列へ
    Hessian の逆行列で伝播させる(Frantar et al. 2022)。scale は原重みから per-row 固定。
    """
    w = w.clone().float()
    rows, cols = w.shape
    qmax = (1 << (bits - 1)) - 1
    scale = _per_row_scale(w, bits)[:, 0]  # [rows]

    def qcol(col: Tensor) -> Tensor:
        return torch.clamp(torch.round(col / scale), -qmax, qmax) * scale

    h = h.clone().float()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(h))
    idx = torch.arange(cols)
    h[idx, idx] += damp
    # H^{-1} の上三角 Cholesky(GPTQ 標準の前処理)。
    h = torch.linalg.cholesky(h)
    h = torch.cholesky_inverse(h)
    hinv = torch.linalg.cholesky(h, upper=True)

    q = torch.zeros_like(w)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        cnt = i2 - i1
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = w1[:, i]
            d = hinv1[i, i]
            qc = qcol(col)
            q1[:, i] = qc
            e = (col - qc) / d
            # 残差誤差をブロック内の未量子化列へ伝播。
            w1[:, i:] -= e.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            err1[:, i] = e
        q[:, i1:i2] = q1
        # ブロック間の伝播。
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q


def _linear_names(model: nn.Module) -> list[str]:
    return [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]


def capture_hessians(
    model: CharGPT, calib_windows: Tensor, batch_size: int = 8
) -> dict[str, Tensor]:
    """各 nn.Linear の入力 Hessian ``Σ xᵀx`` を forward hook で蓄積する。"""
    hessians: dict[str, Tensor] = {}
    handles = []

    def make_hook(name: str) -> Any:
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            h = hessians.get(name)
            xtx = x.t() @ x
            hessians[name] = xtx if h is None else h + xtx
        return hook

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        for s in range(0, calib_windows.size(0), batch_size):
            model(calib_windows[s : s + batch_size])
    for handle in handles:
        handle.remove()
    return hessians


def _get_linear(model: nn.Module, name: str) -> nn.Linear:
    mod: nn.Module = model
    for part in name.split("."):
        mod = getattr(mod, part)
    assert isinstance(mod, nn.Linear)
    return mod


def quantize_model(model: CharGPT, bits: int, method: str, hessians: dict[str, Tensor]) -> None:
    """model の全 nn.Linear を RTN または GPTQ で量子化(in-place・dequant 済み重みを書込)。"""
    with torch.no_grad():
        for name in _linear_names(model):
            lin = _get_linear(model, name)
            if method == "gptq":
                lin.weight.data = quantize_gptq(lin.weight.data, hessians[name], bits)
            else:
                lin.weight.data = quantize_rtn(lin.weight.data, bits)


def _load_checkpoint(path: Path) -> tuple[CharGPT, CharTokenizer]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CharGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])


def _parse_bits(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",")]
    if not raw.strip() or any(not p for p in parts):
        raise ValueError("--bits must be a non-empty comma list")
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--bits must contain only integers") from exc
    if any(b < 2 or b > 16 for b in values):
        raise ValueError("--bits must be in [2,16]")
    return sorted(dict.fromkeys(values), reverse=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GPTQ vs RTN quantization at low bits")
    ap.add_argument("--checkpoint", default="out/lm_aozora_realp1/model.pt")
    ap.add_argument("--corpus-file", default="out/corpus_aozora.txt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--bits", default="3,2")
    ap.add_argument("--calib-windows", type=int, default=32, help="calibration windows for the Hessian")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--json", default="out/gptq_compare.json")
    args = ap.parse_args(argv)

    try:
        bits_list = _parse_bits(args.bits)
        if not 0.0 < args.val_frac < 1.0:
            raise ValueError("--val-frac must be in (0, 1)")
        if args.calib_windows <= 0:
            raise ValueError("--calib-windows must be positive")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ckpt_path = Path(args.checkpoint)
    corpus_path = Path(args.corpus_file)
    if not ckpt_path.exists() or not corpus_path.exists():
        print(f"error: missing checkpoint or corpus: {ckpt_path} / {corpus_path}", file=sys.stderr)
        return 2

    model, tok = _load_checkpoint(ckpt_path)
    text = corpus_path.read_text(encoding="utf-8")
    ids = torch.tensor(tok.encode_safe(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    block = model.config.block_size

    # 校正窓: train split の先頭から非重複 block 窓を calib_windows 個。
    n_train = train_ids.size(0)
    starts = list(range(0, n_train - block, block))[: args.calib_windows]
    if not starts:
        print("error: train split too small for calibration windows", file=sys.stderr)
        return 2
    calib = torch.stack([train_ids[i : i + block] for i in starts])
    print(f"calibrating Hessians on {calib.size(0)} windows ({calib.numel():,} tokens)...")
    hessians = capture_hessians(model, calib)

    fp32_rep = held_out_report_any(model, train_ids, val_ids, tok.vocab_size, block, args.batch_size)
    fp32_ppl = fp32_rep["model_ppl"]
    unigram_ppl = fp32_rep["unigram_ppl"]
    fp32_acc = held_out_top1_report(model, val_ids, block, args.batch_size)
    print(
        f"model: vocab={tok.vocab_size} block={block} linears={len(_linear_names(model))}  "
        f"fp32 PPL={fp32_ppl:.3f} top1={fp32_acc['top1_acc'] * 100:.2f}%"
    )

    records: list[dict[str, Any]] = []
    for bits in bits_list:
        for method in ("rtn", "gptq"):
            qmodel = copy.deepcopy(model)
            quantize_model(qmodel, bits, method, hessians)
            rep = held_out_report_any(qmodel, train_ids, val_ids, tok.vocab_size, block, args.batch_size)
            acc = held_out_top1_report(qmodel, val_ids, block, args.batch_size)
            records.append({
                "bits": bits,
                "method": method,
                "model_ppl": round(rep["model_ppl"], 4),
                "delta_ppl_pct": round((rep["model_ppl"] / fp32_ppl - 1.0) * 100.0, 2),
                "top1_acc": round(acc["top1_acc"], 6),
                "delta_top1_pp": round((acc["top1_acc"] - fp32_acc["top1_acc"]) * 100.0, 2),
                "ppl_gate_pass": passes_gate(rep["model_ppl"], unigram_ppl),
                "capability_gate_pass": passes_capability_gate(acc["top1_acc"], fp32_acc["top1_acc"]),
            })

    print("\n| bits | method | PPL | ΔPPL% | top1% | Δtop1(pp) | ppl-gate | cap-gate |")
    print("|" + "---|" * 8)
    for r in records:
        print(
            f"| {r['bits']} | {r['method']} | {r['model_ppl']:.2f} | {r['delta_ppl_pct']:+.1f}% | "
            f"{r['top1_acc'] * 100:.2f} | {r['delta_top1_pp']:+.2f} | "
            f"{'PASS' if r['ppl_gate_pass'] else 'FAIL'} | "
            f"{'PASS' if r['capability_gate_pass'] else 'FAIL'} |"
        )

    # bits ごとに RTN→GPTQ で cap-gate が FAIL→PASS に変わったか(=GPTQ が床を越えたか)。
    crossed = []
    for bits in bits_list:
        rtn = next(r for r in records if r["bits"] == bits and r["method"] == "rtn")
        gptq = next(r for r in records if r["bits"] == bits and r["method"] == "gptq")
        if (not rtn["capability_gate_pass"]) and gptq["capability_gate_pass"]:
            crossed.append(bits)
    print(
        f"\n[headline] GPTQ が RTN を超えて cap-gate を越えたビット: {crossed or '無し'}。"
        " GPTQ は全 bit で RTN より PPL/top1 を改善する想定(出力誤差最小化)。"
    )
    print(
        "[honest] GPTQ は weight 誤差を犠牲にして output 誤差を下げる(‖(W−Ŵ)X‖² 最小化)。"
        " RTN 超でも 2bit が strict gate を越えるとは限らない(QAT 領域)。Linear のみ・simulated quant。"
    )

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {"checkpoint": str(ckpt_path), "corpus_file": str(corpus_path),
                   "bits": bits_list, "calib_windows": args.calib_windows},
        "fp32": {"model_ppl": round(fp32_ppl, 4), "unigram_ppl": round(unigram_ppl, 4),
                 "top1_acc": round(fp32_acc["top1_acc"], 6)},
        "records": records,
        "gptq_crossed_capgate_bits": crossed,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
