# SPDX-License-Identifier: Apache-2.0
"""Per-layer linearization-tolerance profile of a real pretrained Qwen2 model.

Deep-internal research output: how much does replacing each layer's softmax attention with a
constant-state linear attention (reusing that layer's pretrained projections) hurt language-model
quality, and how many layers can be linearized before the model breaks? Each linearized layer
trades an O(T) KV cache for an O(d²) running state — constant in conversation length — so the
profile tells us how much bounded-memory we can buy and at what quality cost, before any recovery
distillation. Memory is kept ~flat by swapping a layer's attention in place and restoring it (the
linear attention REUSES the original projection weights, so no model copy / no RAM doubling).

Honest scope: tiny CPU model, perplexity on one held-out Japanese corpus; this measures
*zero-shot* linearization (no distillation). Recovery via llcore's constant-state distillation is
the follow-up. Prior art: SUPRA / LoLCATs / Mamba-in-Llama; contribution = on-prem internal
surgery + rigorous per-layer measurement in llcore's own code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from llcore.runtime.linearize import LinearAttention
from llcore.runtime.qwen2 import Qwen2Attention, Qwen2LM, Qwen2Params


def _load_model(model_dir: str) -> tuple[Qwen2LM, object]:
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    model = Qwen2LM(Qwen2Params.from_hf_config(cfg)).eval()
    model.load_hf_state_dict(load_file(str(Path(model_dir) / "model.safetensors")))
    tok = AutoTokenizer.from_pretrained(model_dir)
    return model, tok


def _set_linear(model: Qwen2LM, idxs: list[int]) -> dict[int, Qwen2Attention]:
    """Swap the given layers to linear attention IN PLACE; return the saved originals to restore."""
    saved: dict[int, Qwen2Attention] = {}
    for i in idxs:
        attn = model.model.layers[i].self_attn
        if isinstance(attn, Qwen2Attention):
            saved[i] = attn
            model.model.layers[i].self_attn = LinearAttention.from_attention(attn, model.params)
    return saved


def _restore(model: Qwen2LM, saved: dict[int, Qwen2Attention]) -> None:
    for i, attn in saved.items():
        model.model.layers[i].self_attn = attn


@torch.no_grad()
def _nll(model: Qwen2LM, ids: torch.Tensor) -> float:
    logits = model(ids)  # [1,T,V]
    assert isinstance(logits, torch.Tensor)
    pred = logits[:, :-1].reshape(-1, logits.size(-1))
    tgt = ids[:, 1:].reshape(-1)
    return float(torch.nn.functional.cross_entropy(pred, tgt).item())


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="per-layer linearization tolerance of a Qwen2 model")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--text-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--n-tokens", type=int, default=1024)
    ap.add_argument("--text-skip-chars", type=int, default=50000)
    ap.add_argument("--out", default="out/linearize_tolerance")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, tok = _load_model(args.model_dir)
    n_layer = model.params.n_layer
    print(f"[load] {sum(p.numel() for p in model.parameters())/1e6:.0f}M params in {time.perf_counter()-t0:.1f}s", flush=True)

    text = Path(args.text_file).read_text(encoding="utf-8")[args.text_skip_chars:]
    ids = tok(text, return_tensors="pt").input_ids[:, : args.n_tokens]  # type: ignore[operator]
    print(f"[text] {args.text_file}  eval_tokens={ids.size(1)}", flush=True)

    base_nll = _nll(model, ids)
    base_ppl = float(torch.tensor(base_nll).exp())
    print(f"[baseline] softmax all-layers  nll={base_nll:.4f}  ppl={base_ppl:.2f}", flush=True)

    # 1) per-layer: linearize ONE layer at a time
    per_layer: list[dict[str, Any]] = []
    for i in range(n_layer):
        saved = _set_linear(model, [i])
        nll = _nll(model, ids)
        _restore(model, saved)
        per_layer.append({"layer": i, "nll": nll, "ppl": float(torch.tensor(nll).exp()), "delta_nll": nll - base_nll})
        print(f"  layer {i:>2} linearized: ppl={per_layer[-1]['ppl']:.2f}  Δnll={per_layer[-1]['delta_nll']:+.4f}", flush=True)

    # 2) cumulative: linearize the most-tolerant K layers (greedy by single-layer Δ)
    order = [int(d["layer"]) for d in sorted(per_layer, key=lambda d: d["delta_nll"])]
    cumulative = []
    for k in (1, 2, 4, 6, 8, 12, n_layer):
        if k > n_layer:
            continue
        idxs = order[:k]
        saved = _set_linear(model, idxs)
        nll = _nll(model, ids)
        _restore(model, saved)
        cumulative.append({"k": k, "layers": sorted(idxs), "nll": nll, "ppl": float(torch.tensor(nll).exp()), "delta_nll": nll - base_nll})
        print(f"  top-{k:>2} tolerant linearized: ppl={cumulative[-1]['ppl']:.2f}  Δnll={cumulative[-1]['delta_nll']:+.4f}", flush=True)

    # 3) memory accounting: softmax KV O(T) vs linear state O(d^2) (constant)
    p = model.params
    softmax_kv_per_layer: Callable[[int], int] = lambda T: 2 * p.n_kv_head * T * p.head_dim * 4  # noqa: E731
    linear_state_per_layer = p.n_head * p.head_dim * p.head_dim * 4 + p.n_head * p.head_dim * 4
    crossover_T = linear_state_per_layer // (2 * p.n_kv_head * p.head_dim * 4)
    mem = {
        "softmax_kv_bytes_per_layer_at": {str(T): softmax_kv_per_layer(T) for T in (512, 2048, 8192, 32768)},
        "linear_state_bytes_per_layer": linear_state_per_layer,
        "linear_is_constant_in_T": True,
        "crossover_tokens_where_linear_cheaper": int(crossover_T),
        "note": "softmax KV grows linearly with conversation length T; linear-attention state is "
        "fixed. Beyond ~crossover_T tokens of context, a linearized layer uses less memory, and "
        "it never OOMs as the conversation grows.",
    }

    report = {
        "model_dir": args.model_dir,
        "n_layer": n_layer,
        "eval_tokens": int(ids.size(1)),
        "baseline_nll": base_nll,
        "baseline_ppl": base_ppl,
        "per_layer": per_layer,
        "tolerance_order_best_first": order,
        "cumulative_topk": cumulative,
        "memory": mem,
    }
    (out / "linearize_tolerance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    most_tol = min(per_layer, key=lambda d: d["delta_nll"])
    least_tol = max(per_layer, key=lambda d: d["delta_nll"])
    print(
        f"\n[profile] most tolerant layer={most_tol['layer']} (Δnll={most_tol['delta_nll']:+.4f}); "
        f"least tolerant layer={least_tol['layer']} (Δnll={least_tol['delta_nll']:+.4f})",
        flush=True,
    )
    print(f"[memory] linear state/layer={linear_state_per_layer}B (const); softmax KV/layer@8192={softmax_kv_per_layer(8192)}B; crossover≈{int(crossover_T)} tok", flush=True)
    print(f"[done] wrote {out}/linearize_tolerance.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
