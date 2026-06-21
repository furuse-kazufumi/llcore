# SPDX-License-Identifier: Apache-2.0
"""量子化ビット幅スイープ — cliff_then_flat を反証可能に実測する。

メモリ効率 pivot の検証実験(2026-06-17, memory-scaling ワークフローの完全性批評が推奨)。
`scripts/int8_quant_footprint.py` の対称量子化を **任意ビット幅**へ一般化し、
{8,6,5,4,3,2}-bit の per-channel weight-only PTQ で held-out PPL を測る。

検証する反証可能予測(literature: Dettmers & Zettlemoyer 2023 / cliff_then_flat)::

    PPL は ~4bit まで fp32 baseline 近傍で平坦 → 3bit で急落(cliff)→ 2bit で破綻(QAT 無し)。

honest hook(同批評の警告)
--------------------------
PPL だけ見ると **capability cliff は PPL cliff より低く見える**(PPL 無傷でも厳密 recall は先に崩れる)。
そこで **hard-capability proxy = held-out 次トークン top-1 accuracy** を PPL と併記する。top-1 が PPL より
早く/大きく劣化すれば、「PPL だけの gate は危険」を自前データで実証したことになる。

留保(int8 script と共通)
- weights-only / dequant fp32 forward の simulated quantization(速度は測らない)。
- footprint は保存/常駐バイト実合計(int 本体=numel×bits/8 + scale fp32 + 非量子化 1-D fp32)。
- bits は重み精度。activation 量子化(W4A4 等)はより早く劣化するが本実験の対象外。

使い方::

    py -3.11 scripts/quant_bitwidth_sweep.py \
        --checkpoint out/lm_aozora_multi_smoke/model.pt \
        --corpus-file out/corpus_aozora_multi.txt \
        --bits 8,6,5,4,3,2 --json out/quant_bitwidth_sweep.json
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


def qmax_for_bits(bits: int) -> int:
    """対称量子化の最大正整数レベル。bits=8→127, 4→7, 3→3, 2→1。"""
    if bits < 2:
        raise ValueError(f"bits must be >= 2 (got {bits}); 1-bit symmetric has no nonzero level")
    return (1 << (bits - 1)) - 1


def quantize_symmetric(w: Tensor, bits: int, per_channel: bool) -> tuple[Tensor, Tensor]:
    """対称 ``bits``-bit 量子化(int8 script の一般化版)。``(q[int], scale)`` を返す。

    per_channel=True は 2-D 重みの行(出力チャネル)ごとに scale。``amax`` は 0 除算回避で下限 clamp。
    低ビットでは表現レベルが減るので量子化誤差が増える(cliff の原因)。
    """
    if w.dim() != 2:
        raise ValueError(f"quantize_symmetric expects a 2-D weight, got {w.dim()}-D")
    qmax = qmax_for_bits(bits)
    if per_channel:
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    else:
        amax = w.abs().amax().clamp(min=1e-12)
    scale = amax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q, scale.to(torch.float32)


def dequantize(q: Tensor, scale: Tensor) -> Tensor:
    """``q * scale`` で fp32 重みを復元する。"""
    return q.to(torch.float32) * scale


def _unique_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """tied parameter を 1 度だけ含む (name, param) のリスト(id で dedup)。"""
    seen: set[int] = set()
    out: list[tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((name, p))
    return out


def footprint_bytes(model: nn.Module, bits: int, per_channel: bool) -> dict[str, float]:
    """``bits``-bit 量子化時の保存バイト数(int 本体 + scale + 非量子化 1-D)。"""
    fp32_bytes = 0.0
    int_body = 0.0
    scale_bytes = 0.0
    unquant_bytes = 0.0
    for _, p in _unique_named_params(model):
        numel = p.numel()
        fp32_bytes += numel * 4
        if p.dim() == 2:
            int_body += numel * bits / 8.0  # bits/weight に厳密線形(唯一の真の線形軸)
            scale_bytes += (p.size(0) if per_channel else 1) * 4
        else:
            unquant_bytes += numel * 4  # 1-D(bias/LayerNorm)は fp32 据え置き
    int8_total = int_body + scale_bytes + unquant_bytes
    return {
        "fp32_bytes": fp32_bytes,
        "quant_bytes": int8_total,
        "compression_ratio": round(int8_total / fp32_bytes, 4),
        "savings_pct": round((1.0 - int8_total / fp32_bytes) * 100.0, 2),
    }


def apply_quantization(model: nn.Module, bits: int, per_channel: bool) -> float:
    """2-D 重みを quant→dequant で書き換え、重み rel-RMSE を返す(in-place)。"""
    total_sq_err = 0.0
    total_sq_val = 0.0
    with torch.no_grad():
        for _, p in _unique_named_params(model):
            if p.dim() != 2:
                continue
            w = p.data
            w_hat = dequantize(*quantize_symmetric(w, bits, per_channel))
            total_sq_err += float(((w_hat - w) ** 2).sum().item())
            total_sq_val += float((w * w).sum().item())
            p.data = w_hat
    return (total_sq_err / total_sq_val) ** 0.5 if total_sq_val > 0 else 0.0


# hard-capability proxy(top-1/top-5)は llcore の first-class eval 指標へ昇格したので再利用する
# (DRY)。エイリアスにすることで本スクリプト内の既存呼び出し・テストの参照名はそのまま使える。
held_out_top1 = held_out_top1_report


def _load_checkpoint(path: Path) -> tuple[CharGPT, CharTokenizer]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CharGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])


def _parse_bits(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",")]
    if not raw.strip() or any(not part for part in parts):
        raise ValueError("--bits must be a non-empty comma list")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--bits must contain only integers") from exc
    invalid = [v for v in values if v < 2 or v > 16]
    if invalid:
        raise ValueError(f"--bits must be in [2,16]; got {', '.join(map(str, invalid))}")
    # 大きいビット幅から並べると cliff が読みやすい(降順 unique)。
    return sorted(dict.fromkeys(values), reverse=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="quantization bit-width sweep: PPL + capability cliff")
    ap.add_argument("--checkpoint", default="out/lm_aozora_multi_smoke/model.pt")
    ap.add_argument("--corpus-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--bits", default="8,6,5,4,3,2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--json", default="out/quant_bitwidth_sweep.json")
    args = ap.parse_args(argv)

    try:
        bits_list = _parse_bits(args.bits)
        if not 0.0 < args.val_frac < 1.0:
            raise ValueError("--val-frac must be in (0, 1)")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ckpt_path = Path(args.checkpoint)
    corpus_path = Path(args.corpus_file)
    if not ckpt_path.exists():
        print(f"error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    if not corpus_path.exists():
        print(f"error: corpus not found: {corpus_path}", file=sys.stderr)
        return 2

    model, tok = _load_checkpoint(ckpt_path)
    text = corpus_path.read_text(encoding="utf-8")
    ids = torch.tensor(tok.encode_safe(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    block_size = model.config.block_size

    # fp32 baseline: PPL(尤度)と top-1/top-5(hard-capability proxy)。
    fp32_report = held_out_report_any(
        model, train_ids, val_ids, tok.vocab_size, block_size, args.batch_size
    )
    fp32_ppl = fp32_report["model_ppl"]
    unigram_ppl = fp32_report["unigram_ppl"]
    fp32_acc = held_out_top1(model, val_ids, block_size, args.batch_size)
    n_tokens = int(fp32_report["n_tokens"])

    print(
        f"model: {sum(p.numel() for _, p in _unique_named_params(model)):,} params  "
        f"vocab={tok.vocab_size}  block={block_size}  eval_tokens={n_tokens:,}"
    )
    print(
        f"fp32 baseline: PPL={fp32_ppl:.4f}  top1={fp32_acc['top1_acc'] * 100:.2f}%  "
        f"top5={fp32_acc['top5_acc'] * 100:.2f}%  (unigram PPL={unigram_ppl:.2f})"
    )

    records: list[dict[str, Any]] = []
    for bits in bits_list:
        quant_model = copy.deepcopy(model)
        rel_rmse = apply_quantization(quant_model, bits, per_channel=True)
        rep = held_out_report_any(
            quant_model, train_ids, val_ids, tok.vocab_size, block_size, args.batch_size
        )
        acc = held_out_top1(quant_model, val_ids, block_size, args.batch_size)
        fp = footprint_bytes(model, bits, per_channel=True)
        records.append({
            "bits": bits,
            "compression_ratio": fp["compression_ratio"],
            "savings_pct": fp["savings_pct"],
            "model_ppl": round(rep["model_ppl"], 4),
            "delta_ppl_pct": round((rep["model_ppl"] / fp32_ppl - 1.0) * 100.0, 3),
            "top1_acc": round(acc["top1_acc"], 6),
            "delta_top1_pp": round((acc["top1_acc"] - fp32_acc["top1_acc"]) * 100.0, 3),
            "top5_acc": round(acc["top5_acc"], 6),
            "weight_rel_rmse": round(rel_rmse, 6),
            "ppl_gate_pass": passes_gate(rep["model_ppl"], unigram_ppl),
            # capability gate: top-1 を fp32 比 97% 以上保てているか。ppl_gate が見逃す
            # 低ビットの recall 喪失を捕まえる(両者の差が本実験の眼目)。
            "capability_gate_pass": passes_capability_gate(acc["top1_acc"], fp32_acc["top1_acc"]),
        })

    print("\n| bits | ratio | savings | PPL | ΔPPL% | top1% | Δtop1(pp) | ppl-gate | cap-gate |")
    print("|" + "---|" * 9)
    for r in records:
        print(
            f"| {r['bits']} | {r['compression_ratio']:.3f} | {r['savings_pct']:.1f}% | "
            f"{r['model_ppl']:.3f} | {r['delta_ppl_pct']:+.2f}% | {r['top1_acc'] * 100:.2f} | "
            f"{r['delta_top1_pp']:+.2f} | "
            f"{'PASS' if r['ppl_gate_pass'] else 'FAIL'} | "
            f"{'PASS' if r['capability_gate_pass'] else 'FAIL'} |"
        )

    # cliff = 明確な急落の最初の点(ΔPPL% > 10 か unigram gate 失敗)。降順(8→2)で下げながら見る。
    # +1.66%@4bit のような「軽い膝」を cliff と呼ばないよう、しきい値は保守的に 10% とする。
    cliff_bits = None
    knee_bits = None  # 膝 = ΔPPL% が初めて 1% を超える点(cliff より高ビット側に来やすい)
    for r in records:
        if knee_bits is None and r["delta_ppl_pct"] > 1.0:
            knee_bits = r["bits"]
        if r["delta_ppl_pct"] > 10.0 or not r["ppl_gate_pass"]:
            cliff_bits = r["bits"]
            break
    worst = records[-1]
    # ppl-gate は通すのに cap-gate が止める最初のビット幅 = 「PPL だけでは見逃す」実証点。
    ppl_pass_cap_fail = next(
        (r["bits"] for r in records if r["ppl_gate_pass"] and not r["capability_gate_pass"]), None
    )
    print(
        f"\n[headline] 膝(ΔPPL>1%開始)= {knee_bits or '無'} bit / cliff(ΔPPL>10% or gate fail)= "
        f"{cliff_bits or '無'} bit。最低 {worst['bits']}bit で PPL {worst['delta_ppl_pct']:+.1f}% / "
        f"top1 {worst['delta_top1_pp']:+.2f}pp / ppl-gate {'PASS' if worst['ppl_gate_pass'] else 'FAIL'} / "
        f"cap-gate {'PASS' if worst['capability_gate_pass'] else 'FAIL'}。"
    )
    print(
        f"[honest] ppl-gate は通すが cap-gate が止めるビット幅 = {ppl_pass_cap_fail or '無し'}"
        "(= PPL だけの合否では低ビットの capability 喪失を見逃す実証)。"
        " weights-only / simulated quant(速度未測)。"
    )

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {
            "checkpoint": str(ckpt_path),
            "corpus_file": str(corpus_path),
            "val_frac": args.val_frac,
            "bits": bits_list,
            "batch_size": args.batch_size,
        },
        "fp32": {
            "model_ppl": round(fp32_ppl, 4),
            "unigram_ppl": round(unigram_ppl, 4),
            "top1_acc": round(fp32_acc["top1_acc"], 6),
            "top5_acc": round(fp32_acc["top5_acc"], 6),
            "n_eval_tokens": n_tokens,
        },
        "ppl_knee_bits": knee_bits,
        "ppl_cliff_bits": cliff_bits,
        "records": records,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
