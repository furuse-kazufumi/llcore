# SPDX-License-Identifier: Apache-2.0
"""per-group 量子化比較 — 低ビットの「床」を per-channel より下げられるか。

#3(GPU scale-up)の **CPU で実測可能なスライス**。`quant_bitwidth_sweep.py` は per-channel RTN が
3bit で劣化・2bit で破綻することを示した。本スクリプトは **per-group 量子化**(各行を ``group_size``
列ごとに区切り、群ごとに scale を持つ)が、その床を下げられるかを held-out PPL + top-1 で検証する。
群を小さくすると scale が増えて footprint は増すが、量子化誤差(=cliff の原因)は下がる。

honest 留保
-----------
- per-channel は「群 = 行全体」の特殊形。per_group(G<in) は scale 数が `out×ceil(in/G)` に増える。
- weights-only / dequant fp32 forward の simulated quant(速度は測らない)。RTN(round-to-nearest)で
  GPTQ のような誤差補償はしない(per-group の効果のみを切り出すため)。footprint は int 本体 + scale +
  非量子化 1-D の実合計。

使い方::

    py -3.11 scripts/quant_group_compare.py \
        --checkpoint out/lm_aozora_multi_smoke/model.pt \
        --corpus-file out/corpus_aozora_multi.txt --bits 3,2 --groups full,128,64,32
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

from llcore.lm.data import train_val_split
from llcore.lm.eval import (
    held_out_report_any,
    held_out_top1_report,
    passes_capability_gate,
    passes_gate,
)
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.tokenizer import CharTokenizer


def quantize_grouped(w: Tensor, bits: int, group_size: int) -> tuple[Tensor, int]:
    """per-group 対称量子化。``(dequantized w_hat, n_scales)`` を返す。

    行(dim0=出力)を保ったまま列(dim1=入力)を ``group_size`` ごとに区切り、群ごとに
    amax→scale を取る。``group_size<=0`` または ``>=in_features`` は per-channel(群=行全体)。
    """
    if w.dim() != 2:
        raise ValueError(f"quantize_grouped expects a 2-D weight, got {w.dim()}-D")
    qmax = (1 << (bits - 1)) - 1
    out_f, in_f = w.shape
    g = in_f if group_size <= 0 or group_size >= in_f else group_size
    w_hat = torch.empty_like(w)
    n_groups = 0
    for start in range(0, in_f, g):
        sl = w[:, start : start + g]
        amax = sl.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = amax / qmax
        q = torch.clamp(torch.round(sl / scale), -qmax, qmax)
        w_hat[:, start : start + g] = q * scale
        n_groups += 1
    return w_hat, out_f * n_groups


def _unique_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    out: list[tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((name, p))
    return out


def apply_grouped_quant(model: nn.Module, bits: int, group_size: int) -> int:
    """2-D 重みを per-group quant→dequant で書換え、総 scale 数を返す(in-place)。"""
    total_scales = 0
    with torch.no_grad():
        for _, p in _unique_named_params(model):
            if p.dim() != 2:
                continue
            w_hat, n_scales = quantize_grouped(p.data, bits, group_size)
            p.data = w_hat
            total_scales += n_scales
    return total_scales


def footprint_ratio(model: nn.Module, bits: int, total_scales: int) -> float:
    """int 本体(bits)+ scale(fp32)+ 非量子化 1-D(fp32)/ fp32 全体。"""
    fp32 = 0
    quant = 0.0
    for _, p in _unique_named_params(model):
        numel = p.numel()
        fp32 += numel * 4
        quant += numel * bits / 8.0 if p.dim() == 2 else numel * 4
    quant += total_scales * 4
    return quant / fp32


def _load_checkpoint(path: Path) -> tuple[CharGPT, CharTokenizer]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CharGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])


def _parse_int_list(raw: str, name: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",")]
    if not raw.strip() or any(not p for p in parts):
        raise ValueError(f"--{name} must be a non-empty comma list")
    try:
        return list(dict.fromkeys(int(p) for p in parts))
    except ValueError as exc:
        raise ValueError(f"--{name} must contain only integers") from exc


def _parse_groups(raw: str) -> list[int]:
    """'full' を 0(=per-channel)へ、それ以外は正整数の群サイズへ。"""
    parts = [p.strip() for p in raw.split(",")]
    if not raw.strip() or any(not p for p in parts):
        raise ValueError("--groups must be a non-empty comma list")
    out: list[int] = []
    for p in parts:
        if p == "full":
            out.append(0)
        else:
            try:
                v = int(p)
            except ValueError as exc:
                raise ValueError("--groups items must be 'full' or a positive integer") from exc
            if v <= 0:
                raise ValueError("--groups integer items must be positive")
            out.append(v)
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="per-group vs per-channel quantization at low bits")
    ap.add_argument("--checkpoint", default="out/lm_aozora_multi_smoke/model.pt")
    ap.add_argument("--corpus-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--bits", default="3,2")
    ap.add_argument("--groups", default="full,128,64,32")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--json", default="out/quant_group_compare.json")
    args = ap.parse_args(argv)

    try:
        bits_list = sorted(_parse_int_list(args.bits, "bits"), reverse=True)
        groups = _parse_groups(args.groups)
        if any(b < 2 or b > 16 for b in bits_list):
            raise ValueError("--bits must be in [2,16]")
        if not 0.0 < args.val_frac < 1.0:
            raise ValueError("--val-frac must be in (0, 1)")
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

    fp32_rep = held_out_report_any(model, train_ids, val_ids, tok.vocab_size, block, args.batch_size)
    fp32_ppl = fp32_rep["model_ppl"]
    unigram_ppl = fp32_rep["unigram_ppl"]
    fp32_acc = held_out_top1_report(model, val_ids, block, args.batch_size)
    print(
        f"model: vocab={tok.vocab_size} block={block}  fp32 PPL={fp32_ppl:.3f} "
        f"top1={fp32_acc['top1_acc'] * 100:.2f}%  (unigram {unigram_ppl:.1f})"
    )

    records: list[dict[str, Any]] = []
    for bits in bits_list:
        for g in groups:
            qmodel = copy.deepcopy(model)
            total_scales = apply_grouped_quant(qmodel, bits, g)
            ratio = footprint_ratio(model, bits, total_scales)
            rep = held_out_report_any(qmodel, train_ids, val_ids, tok.vocab_size, block, args.batch_size)
            acc = held_out_top1_report(qmodel, val_ids, block, args.batch_size)
            records.append({
                "bits": bits,
                "group": "full" if g == 0 else g,
                "compression_ratio": round(ratio, 4),
                "model_ppl": round(rep["model_ppl"], 4),
                "delta_ppl_pct": round((rep["model_ppl"] / fp32_ppl - 1.0) * 100.0, 2),
                "top1_acc": round(acc["top1_acc"], 6),
                "delta_top1_pp": round((acc["top1_acc"] - fp32_acc["top1_acc"]) * 100.0, 2),
                "ppl_gate_pass": passes_gate(rep["model_ppl"], unigram_ppl),
                "capability_gate_pass": passes_capability_gate(acc["top1_acc"], fp32_acc["top1_acc"]),
            })

    print("\n| bits | group | ratio | PPL | ΔPPL% | top1% | Δtop1(pp) | ppl-gate | cap-gate |")
    print("|" + "---|" * 9)
    for r in records:
        print(
            f"| {r['bits']} | {r['group']} | {r['compression_ratio']:.3f} | {r['model_ppl']:.2f} | "
            f"{r['delta_ppl_pct']:+.1f}% | {r['top1_acc'] * 100:.2f} | {r['delta_top1_pp']:+.2f} | "
            f"{'PASS' if r['ppl_gate_pass'] else 'FAIL'} | "
            f"{'PASS' if r['capability_gate_pass'] else 'FAIL'} |"
        )

    # 各 bits で cap-gate を最初に通す群サイズ(= per-channel が落ちる床を救えたか)。
    rescued: dict[int, Any] = {}
    for bits in bits_list:
        for r in records:
            if r["bits"] == bits and r["capability_gate_pass"]:
                rescued[bits] = r["group"]
                break
    print(
        "\n[headline] cap-gate を通す最小ビット床を per-group が救うか: "
        + " / ".join(f"{b}bit→{rescued.get(b, '救えず')}" for b in bits_list)
        + "(群を小さくするほど誤差↓・footprint↑)。"
    )
    print("[honest] RTN per-group(誤差補償なし)/ weights-only / simulated quant(速度未測)。")

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {"checkpoint": str(ckpt_path), "corpus_file": str(corpus_path),
                   "bits": bits_list, "groups": ["full" if g == 0 else g for g in groups]},
        "fp32": {"model_ppl": round(fp32_ppl, 4), "unigram_ppl": round(unigram_ppl, 4),
                 "top1_acc": round(fp32_acc["top1_acc"], 6)},
        "records": records,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
