# SPDX-License-Identifier: Apache-2.0
"""int8 量子化 footprint vs PPL — メモリ効率 pivot 第二歩 (b)。

llcore の北極星「メモリ使用効率」(2026-06-16 pivot,
memory:project_llcore_memory_efficiency_pivot) の (b)。訓練済み :class:`CharGPT`
の 2-D 重み行列を対称 int8 量子化し、

  - **footprint**: fp32 bytes vs int8 bytes(scale と非量子化 1-D params を含む実合計)
  - **PPL コスト**: 同一 held-out split で fp32 model_ppl vs int8 model_ppl

を測る。量子化方式は **per-tensor**(テンソル 1 scale)と **per-channel**(出力行ごと
の scale)の 2 種を比較し、footprint/品質トレードオフを示す。

honest 留保
-----------
- これは **weights-only** の量子化(activation は fp32)。推論時の activation/KV メモリは
  別問題で、ここでは測らない。北極星の「常駐・保存に必要な重みバイト数」を対象にする。
- int8 footprint には **scale(fp32)と非量子化 1-D params(bias / LayerNorm)を必ず計上**
  する。「重みだけ 1/4」ではなく、現実に保存・常駐するバイトの合計を示すため。
- 量子化は dequant して fp32 で forward する **simulated quantization**(真の int8 GEMM では
  ない)。これは「量子化が PPL をどれだけ劣化させるか」を測るためで、速度は測らない。
- tied ``wte`` / ``lm_head`` は同一 Parameter なので **1 度だけ**量子化・計上する。
- 因果マスク buffer(``attn.bias``)は学習パラメータではないので footprint 対象外
  (再生成可能)。

使い方::

    py -3.11 scripts/int8_quant_footprint.py \
        --checkpoint out/lm_aozora_multi_smoke/model.pt \
        --corpus-file out/corpus_aozora_multi.txt \
        --json out/int8_quant_footprint.json
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
from llcore.lm.eval import held_out_report_any, passes_gate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.tokenizer import CharTokenizer

INT8_MAX = 127
SCHEMES = ("per_tensor", "per_channel")


def quantize_symmetric(w: Tensor, per_channel: bool) -> tuple[Tensor, Tensor]:
    """対称 int8 量子化。``(q[int8], scale[float32])`` を返す。

    per_channel=True は 2-D 重みの **行(dim 0 = 出力チャネル)ごと**に scale を持つ
    (Linear の out_features 単位 = 標準的な weight-only 量子化)。per_channel=False は
    テンソル全体で 1 つの scale。``amax`` は 0 除算回避のため下限 clamp する。
    """
    if w.dim() != 2:
        raise ValueError(f"quantize_symmetric expects a 2-D weight, got {w.dim()}-D")
    if per_channel:
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    else:
        amax = w.abs().amax().clamp(min=1e-12)
    scale = amax / INT8_MAX
    q = torch.clamp(torch.round(w / scale), -INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale.to(torch.float32)


def dequantize(q: Tensor, scale: Tensor) -> Tensor:
    """``q.float() * scale`` で fp32 重みを復元する。"""
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


def is_quantizable(p: nn.Parameter) -> bool:
    """2-D 重みのみ量子化対象(bias / LayerNorm など 1-D は fp32 据え置き)。"""
    return p.dim() == 2


def footprint_bytes(model: nn.Module, per_channel: bool) -> dict[str, int]:
    """fp32 / int8 の保存バイト数を計上する(scale・非量子化 params を含む実合計)。

    Returns
    -------
    dict with::

        fp32_bytes, int8_bytes, quantized_param_bytes(int8 本体のみ),
        scale_bytes, unquantized_param_bytes, ideal_int8_bytes(全 params を 1B 換算),
        n_quantized_params, n_unquantized_params
    """
    fp32_bytes = 0
    int8_body = 0
    scale_bytes = 0
    unquant_bytes = 0
    ideal_int8 = 0
    n_q = 0
    n_u = 0
    for _, p in _unique_named_params(model):
        numel = p.numel()
        fp32_bytes += numel * 4
        ideal_int8 += numel * 1
        if is_quantizable(p):
            int8_body += numel * 1
            scale_bytes += (p.size(0) if per_channel else 1) * 4
            n_q += 1
        else:
            unquant_bytes += numel * 4
            n_u += 1
    return {
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_body + scale_bytes + unquant_bytes,
        "quantized_param_bytes": int8_body,
        "scale_bytes": scale_bytes,
        "unquantized_param_bytes": unquant_bytes,
        "ideal_int8_bytes": ideal_int8,
        "n_quantized_params": n_q,
        "n_unquantized_params": n_u,
    }


def apply_quantization(model: nn.Module, per_channel: bool) -> dict[str, float]:
    """``model`` の 2-D 重みを quant→dequant で書き換え、重み誤差統計を返す(in-place)。

    tied parameter は同一オブジェクトなので、片側の ``.data`` 書換えで両側が更新される。
    """
    total_sq_err = 0.0
    total_sq_val = 0.0
    total_abs_err = 0.0
    total_numel = 0
    max_abs_err = 0.0
    with torch.no_grad():
        for _, p in _unique_named_params(model):
            if not is_quantizable(p):
                continue
            w = p.data
            q, scale = quantize_symmetric(w, per_channel)
            w_hat = dequantize(q, scale)
            err = (w_hat - w).abs()
            total_sq_err += float((err * err).sum().item())
            total_sq_val += float((w * w).sum().item())
            total_abs_err += float(err.sum().item())
            max_abs_err = max(max_abs_err, float(err.max().item()))
            total_numel += w.numel()
            p.data = w_hat
    rel_rmse = (total_sq_err / total_sq_val) ** 0.5 if total_sq_val > 0 else 0.0
    return {
        "weight_rel_rmse": rel_rmse,
        "weight_mean_abs_err": total_abs_err / max(1, total_numel),
        "weight_max_abs_err": max_abs_err,
    }


def _load_checkpoint(path: Path) -> tuple[CharGPT, CharTokenizer]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CharGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])


def _parse_schemes(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",")]
    if not raw.strip() or any(not part for part in parts):
        raise ValueError("--schemes must be a non-empty comma list")
    invalid = [part for part in parts if part not in SCHEMES]
    if invalid:
        raise ValueError(f"--schemes must be in {SCHEMES}; got {', '.join(invalid)}")
    return list(dict.fromkeys(parts))


def evaluate_scheme(
    base_model: CharGPT,
    scheme: str,
    train_ids: Tensor,
    val_ids: Tensor,
    vocab_size: int,
    block_size: int,
    batch_size: int,
) -> dict[str, Any]:
    """1 量子化方式の footprint + PPL を測る(base_model は不変、複製を量子化する)。"""
    per_channel = scheme == "per_channel"
    quant_model = copy.deepcopy(base_model)
    werr = apply_quantization(quant_model, per_channel)
    fp = footprint_bytes(base_model, per_channel)
    report = held_out_report_any(quant_model, train_ids, val_ids, vocab_size, block_size, batch_size)
    return {
        "scheme": scheme,
        "footprint": fp,
        "compression_ratio": round(fp["int8_bytes"] / fp["fp32_bytes"], 4),
        "savings_pct": round((1.0 - fp["int8_bytes"] / fp["fp32_bytes"]) * 100.0, 2),
        "weight_error": {k: round(v, 6) for k, v in werr.items()},
        "model_ppl": round(report["model_ppl"], 4),
        "model_nll": round(report["model_nll"], 6),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 weight-only quantization: footprint vs PPL")
    ap.add_argument("--checkpoint", default="out/lm_aozora_multi_smoke/model.pt")
    ap.add_argument("--corpus-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--schemes", default="per_tensor,per_channel")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--json", default="out/int8_quant_footprint.json")
    args = ap.parse_args(argv)

    try:
        schemes = _parse_schemes(args.schemes)
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

    fp32_report = held_out_report_any(
        model, train_ids, val_ids, tok.vocab_size, block_size, args.batch_size
    )
    fp32_ppl = fp32_report["model_ppl"]
    unigram_ppl = fp32_report["unigram_ppl"]
    n_tokens = int(fp32_report["n_tokens"])
    n_params = sum(p.numel() for _, p in _unique_named_params(model))

    print(
        f"model: {n_params:,} params  vocab={tok.vocab_size}  block={block_size}  "
        f"eval_tokens={n_tokens:,}"
    )
    print(
        f"fp32 baseline: model_ppl={fp32_ppl:.4f}  "
        f"unigram_ppl={unigram_ppl:.4f}  ratio={fp32_ppl / unigram_ppl:.4f}"
    )

    scheme_records: list[dict[str, Any]] = []
    for scheme in schemes:
        rec = evaluate_scheme(
            model, scheme, train_ids, val_ids, tok.vocab_size, block_size, args.batch_size
        )
        rec["delta_ppl_abs"] = round(rec["model_ppl"] - fp32_ppl, 4)
        rec["delta_ppl_pct"] = round((rec["model_ppl"] / fp32_ppl - 1.0) * 100.0, 3)
        rec["ratio_over_unigram"] = round(rec["model_ppl"] / unigram_ppl, 4)
        rec["ppl_gate_pass"] = passes_gate(rec["model_ppl"], unigram_ppl)
        scheme_records.append(rec)

    print(
        "\n| scheme | fp32 KB | int8 KB | ratio | savings | "
        "model PPL | ΔPPL | Δ% | gate | w-rel-RMSE |"
    )
    print("|" + "---|" * 10)
    for rec in scheme_records:
        fp = rec["footprint"]
        print(
            f"| {rec['scheme']} | {fp['fp32_bytes'] / 1024:,.1f} | "
            f"{fp['int8_bytes'] / 1024:,.1f} | {rec['compression_ratio']:.3f} | "
            f"{rec['savings_pct']:.1f}% | {rec['model_ppl']:.3f} | "
            f"{rec['delta_ppl_abs']:+.3f} | {rec['delta_ppl_pct']:+.2f}% | "
            f"{'PASS' if rec['ppl_gate_pass'] else 'FAIL'} | "
            f"{rec['weight_error']['weight_rel_rmse']:.4f} |"
        )

    print(
        "\n[headline] int8 weight-only は重み常駐を ~4x 圧縮しつつ "
        f"PPL コストは小さい(fp32 {fp32_ppl:.2f} 基準)。"
        "per_channel は per_tensor より誤差が小さい(scale 行ごと)。"
    )
    print(
        "[honest] footprint=保存/常駐バイト(scale+非量子化 1-D 込み)。"
        "activation/KV は別。dequant fp32 forward の simulated quant=速度は未測定。"
    )

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {
            "checkpoint": str(ckpt_path),
            "corpus_file": str(corpus_path),
            "val_frac": args.val_frac,
            "batch_size": args.batch_size,
            "schemes": schemes,
        },
        "model": {"n_params": n_params, "vocab_size": tok.vocab_size, "block_size": block_size},
        "fp32": {
            "model_ppl": round(fp32_ppl, 4),
            "unigram_ppl": round(unigram_ppl, 4),
            "ratio_over_unigram": round(fp32_ppl / unigram_ppl, 4),
            "n_eval_tokens": n_tokens,
        },
        "schemes": scheme_records,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
