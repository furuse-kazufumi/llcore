# SPDX-License-Identifier: Apache-2.0
"""QAT(量子化を意識した学習)— アークの定説「2bit は QAT 領域」を実証する capstone。

`gptq_compare.py` / `quant_group_compare.py` の結論は「PTQ(RTN/per-group/GPTQ)は低ビット品質を
改善するが、strict cap-gate(top1 fp32 比 97% 保持)を **2bit では越えられない**=QAT(学習時量子化)が
必要」。本スクリプトはそれを実証する: **fake-quant + STE**(straight-through estimator)で重みを
量子化したまま学習し、PTQ が落ちた 2bit cap-gate を QAT が越えるかを held-out PPL + top-1 で判定する。

fake-quant + STE
----------------
forward は量子化重み ``wq`` を使い、backward は恒等(``w + (wq - w).detach()``)で fp32 重みへ勾配を流す。
これにより「量子化に頑健な重み」を学習できる(PTQ は学習後に量子化するので量子化を見越せない)。

比較
----
同一 corpus / config / 学習 iters で:
- **fp32 reference**: 既存の fp32 学習済 checkpoint(`--fp32-checkpoint`)の held-out top1 を cap-gate 基準に。
- **QAT(本スクリプト)**: 同 bits で fake-quant 学習 → deploy 時(量子化重み)の top1。
- **PTQ(参考)**: `out/gptq_compare*.json` があれば同 bits の RTN/GPTQ top1 を並記。

honest 留保
-----------
- 量子化対象は `nn.Linear` のみ(Embedding/LN は fp32)= PTQ 比較と apples-to-apples。
- CPU 学習なので smoke config・限定 iters。fp32 reference と同 iters で公平比較する。
- per-channel 対称・weights-only・simulated quant(速度未測)。学習は乱数初期化からで、fp32 reference とは
  別の重みに収束する(QAT vs PTQ は「同じ学習予算で 2bit deploy するならどちらが良いか」の比較)。

使い方::

    py -3.11 scripts/qat_train.py --bits 2 --corpus-file out/corpus_aozora_multi.txt \
        --fp32-checkpoint out/lm_aozora_multi_smoke/model.pt --max-iters 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llcore.lm.data import encode_corpus, train_val_split  # type: ignore[import-untyped]
from llcore.lm.eval import (  # type: ignore[import-untyped]
    held_out_report_any,
    held_out_top1_report,
    passes_capability_gate,
    passes_gate,
)
from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]
from llcore.lm.tokenizer import CharTokenizer  # type: ignore[import-untyped]
from llcore.lm.trainer import Trainer, TrainConfig  # type: ignore[import-untyped]

MODEL_PRESETS = {
    "smoke": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 64},
    "p1": {"n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256},
}


def fake_quant_ste(w: Tensor, bits: int) -> Tensor:
    """per-channel 対称 fake-quant(STE)。forward=量子化重み / backward=恒等で fp32 へ。"""
    qmax = (1 << (bits - 1)) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    wq = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (wq - w).detach()  # STE: 出力は wq、勾配は w へ恒等で流れる


class FakeQuantLinear(nn.Linear):
    """nn.Linear のサブクラス。forward で重みを fake-quant する(QAT 用)。"""

    bits: int

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, F.linear(x, fake_quant_ste(self.weight, self.bits), self.bias))


def convert_to_fake_quant(model: nn.Module, bits: int) -> nn.Module:
    """model 内の全 nn.Linear を FakeQuantLinear へ置換(重み/バイアスを引継ぎ)。"""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, FakeQuantLinear):
            fq = FakeQuantLinear(child.in_features, child.out_features, bias=child.bias is not None)
            fq.bits = bits
            with torch.no_grad():
                fq.weight.copy_(child.weight)
                if child.bias is not None and fq.bias is not None:
                    fq.bias.copy_(child.bias)
            setattr(model, name, fq)
        else:
            convert_to_fake_quant(child, bits)
    return model


def _eval_model(
    model: CharGPT, train_ids: Tensor, val_ids: Tensor, vocab: int, block: int, bs: int
) -> dict[str, float]:
    rep = held_out_report_any(model, train_ids, val_ids, vocab, block, bs)
    acc = held_out_top1_report(model, val_ids, block, bs)
    return {"model_ppl": rep["model_ppl"], "unigram_ppl": rep["unigram_ppl"],
            "top1_acc": acc["top1_acc"]}


def _ptq_reference(bits: int) -> dict[str, float] | None:
    """out/gptq_compare.json があれば同 bits の RTN/GPTQ top1 を拾う(参考並記用)。"""
    path = Path("out/gptq_compare.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 参考情報、壊れていても本体続行
        return None
    out: dict[str, float] = {}
    for rec in data.get("records", []):
        if rec.get("bits") == bits:
            out[f"ptq_{rec['method']}_top1"] = float(rec["top1_acc"])
            out[f"ptq_{rec['method']}_delta_ppl_pct"] = float(rec["delta_ppl_pct"])
    return out or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="QAT (fake-quant + STE): can it cross the 2-bit cap-gate?")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--corpus-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--fp32-checkpoint", default="out/lm_aozora_multi_smoke/model.pt",
                    help="fp32 reference for the capability-retention baseline")
    ap.add_argument("--config", choices=list(MODEL_PRESETS), default="smoke")
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json", default="out/qat_train.json")
    args = ap.parse_args(argv)

    if args.bits < 2 or args.bits > 16:
        print("error: --bits must be in [2,16]", file=sys.stderr)
        return 2
    corpus_path = Path(args.corpus_file)
    if not corpus_path.exists():
        print(f"error: corpus not found: {corpus_path}", file=sys.stderr)
        return 2

    text = corpus_path.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    preset = MODEL_PRESETS[args.config]
    block = int(preset["block_size"])
    bs = args.batch_size

    # fp32 reference(cap-gate の基準 top1)。checkpoint があれば使い、無ければ unigram のみ。
    fp32_ref: dict[str, float] | None = None
    fp32_path = Path(args.fp32_checkpoint)
    if fp32_path.exists():
        ck = torch.load(fp32_path, map_location="cpu", weights_only=True)
        ref_model = CharGPT(GPTConfig(**ck["config"]))
        ref_model.load_state_dict(ck["model_state"])
        ref_model.eval()
        ref_tok = CharTokenizer(ck["itos"])
        # tokenizer 一致が必要(同 vocab)。一致しなければ reference を諦める。
        if ref_tok.itos == tok.itos:
            fp32_ref = _eval_model(ref_model, train_ids, val_ids, tok.vocab_size, block, bs)
        del ref_model

    torch.manual_seed(args.seed)
    model = CharGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=block,
                              n_layer=int(preset["n_layer"]), n_head=int(preset["n_head"]),
                              n_embd=int(preset["n_embd"]), dropout=0.0))
    convert_to_fake_quant(model, args.bits)
    n_lin = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    print(
        f"QAT: {args.bits}-bit  config={args.config}  vocab={tok.vocab_size}  block={block}  "
        f"fake-quant Linears={n_lin}  iters={args.max_iters}  train={train_ids.numel():,}"
    )
    print("training (fake-quant active)...")
    trainer = Trainer(model, TrainConfig(
        max_iters=args.max_iters, lr_decay_iters=args.max_iters,
        warmup_iters=min(100, args.max_iters // 10), batch_size=bs,
        eval_interval=max(1, args.max_iters // 4), eval_iters=20, seed=args.seed))
    trainer.train(train_ids, val_ids,
                  on_eval=lambda it, tr, va: print(f"  iter {it:>5} train {tr:.4f} val {va:.4f}"))

    qat = _eval_model(model, train_ids, val_ids, tok.vocab_size, block, bs)
    unigram_ppl = qat["unigram_ppl"]
    ref_top1 = fp32_ref["top1_acc"] if fp32_ref else 0.0
    cap_pass = passes_capability_gate(qat["top1_acc"], ref_top1) if fp32_ref else None
    ptq = _ptq_reference(args.bits)

    print(f"\n=== QAT {args.bits}-bit VERDICT ===")
    if fp32_ref:
        print(f"  fp32 ref : PPL {fp32_ref['model_ppl']:.2f}  top1 {fp32_ref['top1_acc'] * 100:.2f}%")
    print(f"  QAT      : PPL {qat['model_ppl']:.2f}  top1 {qat['top1_acc'] * 100:.2f}%  "
          f"(unigram {unigram_ppl:.1f})")
    if fp32_ref:
        ret = qat["top1_acc"] / ref_top1 if ref_top1 > 0 else float("nan")
        print(f"  retention: top1 {ret * 100:.1f}% of fp32  "
              f"ppl-gate {'PASS' if passes_gate(qat['model_ppl'], unigram_ppl) else 'FAIL'}  "
              f"cap-gate {'PASS' if cap_pass else 'FAIL'}")
    if ptq:
        for method in ("rtn", "gptq"):
            k = f"ptq_{method}_top1"
            if k in ptq:
                print(f"  PTQ {method:<4}: top1 {ptq[k] * 100:.2f}%  (QAT vs PTQ top1 "
                      f"{(qat['top1_acc'] - ptq[k]) * 100:+.2f}pp)")
    print(
        f"\n[headline] QAT {args.bits}-bit は PTQ が越えられなかった cap-gate を "
        f"{'**越えた** ✓' if cap_pass else '越えられず(QAT でも 2bit は厳しい=honest)'}。"
    )
    print("[honest] CPU smoke / weights-only / Linear のみ / fp32 ref と同 iters で公平比較。")

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {"bits": args.bits, "corpus_file": str(corpus_path), "preset": args.config,
                   "max_iters": args.max_iters, "vocab": tok.vocab_size, "block": block},
        "fp32_reference": fp32_ref,
        "qat": qat,
        "qat_top1_retention": (qat["top1_acc"] / ref_top1) if (fp32_ref and ref_top1 > 0) else None,
        "ppl_gate_pass": passes_gate(qat["model_ppl"], unigram_ppl),
        "capability_gate_pass": cap_pass,
        "ptq_reference": ptq,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
