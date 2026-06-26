# SPDX-License-Identifier: Apache-2.0
"""L7 (Hedgehog ablation): does llcore's linear-attention feature map reproduce softmax's
spiky (low-entropy) + dot-product-monotonic attention, or is the elu+1 / affine map too flat?

Hedgehog (Zhang et al. 2024, arXiv:2402.04347) identified that ``elu+1`` linear attention produces
high-entropy (flat) attention weights and loses softmax's spikiness and its monotonic key ranking
— *the* reason naive linearization underperforms. llcore's ``LinearAttention`` uses
``phi(x)=elu(x)+1`` with an optional identity-init per-head affine on q/k (LoLCATs-style), so a
freshly-learnable linear attention starts exactly at the fixed ``phi``. This script measures, per
layer (averaged over heads/query positions) on a real pretrained Qwen2.5 model:

  * spikiness gap  — mean row entropy and mean top-1 mass, softmax vs linear;
  * monotonicity   — Spearman rank correlation between the per-row key weights of softmax and
                     linear attention (does linear keep the *order* of which keys matter?);

and correlates the spikiness gap with each layer's linearization tolerance (``delta_nll`` from
``linearize_tolerance.py``) — the prediction being that non-tolerant layers have the largest gap.

Honest scope: ZERO-SHOT feature map (no distillation). Identity-init affine == fixed ``phi``, so
this is the *starting point* distillation would have to fix, not the post-recovery state. Tiny CPU
model, one held-out Japanese sample. It does not by itself prove the map is too weak — it
quantifies how far from softmax the starting point is, per layer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

from llcore.runtime.linearize import _phi
from llcore.runtime.qwen2 import (
    Qwen2DecoderLayer,
    Qwen2LM,
    Qwen2Params,
    _apply_rope,
    _repeat_kv,
    _rope_cos_sin,
)


def softmax_weights(q: torch.Tensor, kf: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Causal softmax attention weights ``[H,T,T]`` (rows sum to 1 over keys j<=i)."""
    scores = torch.matmul(q, kf.transpose(-2, -1)) / (head_dim**0.5)  # [H,T,T]
    t = scores.size(-1)
    idx = torch.arange(t)
    mask = torch.where(idx[None, :] <= idx[:, None], 0.0, float("-inf"))
    return torch.softmax(scores + mask, dim=-1)


def linear_weights(q: torch.Tensor, kf: torch.Tensor) -> torch.Tensor:
    """Causal linear-attention implied weights ``[H,T,T]`` for phi(q)·phi(k), rows normalized
    over keys j<=i — the per-key weight that ``_causal_linear_attn`` applies to each value."""
    raw = torch.matmul(_phi(q), _phi(kf).transpose(-2, -1))  # [H,T,T], >0
    t = raw.size(-1)
    idx = torch.arange(t)
    causal = (idx[None, :] <= idx[:, None]).float()  # [T,T]
    raw = raw * causal
    denom = raw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return raw / denom


def mean_row_entropy(weights: torch.Tensor, min_keys: int) -> float:
    """Mean Shannon entropy (nats) of each causal weight row with >= ``min_keys`` valid keys."""
    h, t, _ = weights.shape
    vals: list[float] = []
    for i in range(t):
        if i + 1 < min_keys:
            continue
        row = weights[:, i, : i + 1].clamp_min(1e-12)  # [H, i+1]
        ent = -(row * row.log()).sum(dim=-1)  # [H]
        vals.append(float(ent.mean()))
    return float(sum(vals) / len(vals)) if vals else 0.0


def mean_top1_mass(weights: torch.Tensor, min_keys: int) -> float:
    """Mean over heads/rows of the largest single-key weight (1.0 = perfectly spiky)."""
    h, t, _ = weights.shape
    vals: list[float] = []
    for i in range(t):
        if i + 1 < min_keys:
            continue
        vals.append(float(weights[:, i, : i + 1].max(dim=-1).values.mean()))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors (Pearson on ranks)."""
    if a.numel() < 2:
        return 1.0
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / denom)


def mean_rank_corr(a_sm: torch.Tensor, a_lin: torch.Tensor, min_keys: int) -> float:
    """Mean Spearman corr between softmax & linear key-weights per row (monotonicity of ranking)."""
    h, t, _ = a_sm.shape
    vals: list[float] = []
    for i in range(t):
        if i + 1 < min_keys:
            continue
        for hd in range(h):
            vals.append(_spearman(a_sm[hd, i, : i + 1], a_lin[hd, i, : i + 1]))
    return float(sum(vals) / len(vals)) if vals else 1.0


def _load_model(model_dir: str) -> tuple[Qwen2LM, object]:
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    model = Qwen2LM(Qwen2Params.from_hf_config(cfg)).eval()
    model.load_hf_state_dict(load_file(str(Path(model_dir) / "model.safetensors")))
    tok = AutoTokenizer.from_pretrained(model_dir)
    return model, tok


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Hedgehog ablation: feature-map spikiness/monotonicity")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--text-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--text-skip-chars", type=int, default=50000)
    ap.add_argument("--min-keys", type=int, default=8)
    ap.add_argument("--tolerance-json", default="out/linearize_tolerance/linearize_tolerance.json")
    ap.add_argument("--out", default="out/feature_map_spikiness")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, tok = _load_model(args.model_dir)
    p = model.params
    print(f"[load] {sum(x.numel() for x in model.parameters())/1e6:.0f}M in {time.perf_counter()-t0:.1f}s", flush=True)

    text = Path(args.text_file).read_text(encoding="utf-8")[args.text_skip_chars:]
    ids = tok(text, return_tensors="pt").input_ids[:, : args.n_tokens]  # type: ignore[operator]
    n_tok = int(ids.size(1))
    print(f"[text] eval_tokens={n_tok}", flush=True)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(lambda _m, a, i=i: captured.__setitem__(i, a[0].detach())))
    model(ids)
    for hd in handles:
        hd.remove()

    positions = torch.arange(n_tok)
    cos, sin = _rope_cos_sin(positions, p.head_dim, p.rope_theta)
    n_rep = p.n_head // p.n_kv_head

    # optional: tolerance delta_nll per layer for correlation
    tol: dict[int, float] = {}
    tpath = Path(args.tolerance_json)
    if tpath.exists():
        td = json.loads(tpath.read_text(encoding="utf-8"))
        if int(td.get("n_layer", -1)) == p.n_layer:
            tol = {int(d["layer"]): float(d["delta_nll"]) for d in td["per_layer"]}

    per_layer: list[dict[str, Any]] = []
    for i in range(p.n_layer):
        x = captured[i]
        layer = cast(Qwen2DecoderLayer, model.model.layers[i])
        attn = layer.self_attn
        xn = layer.input_layernorm(x)
        q = attn.q_proj(xn).view(1, n_tok, p.n_head, p.head_dim).transpose(1, 2)
        k = attn.k_proj(xn).view(1, n_tok, p.n_kv_head, p.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        kf = _repeat_kv(k, n_rep)
        qh, kfh = q[0], kf[0]  # [H,T,Dk]
        a_sm = softmax_weights(qh, kfh, p.head_dim)
        a_lin = linear_weights(qh, kfh)
        rec = {
            "layer": i,
            "entropy_softmax": mean_row_entropy(a_sm, args.min_keys),
            "entropy_linear": mean_row_entropy(a_lin, args.min_keys),
            "top1_softmax": mean_top1_mass(a_sm, args.min_keys),
            "top1_linear": mean_top1_mass(a_lin, args.min_keys),
            "rank_corr": mean_rank_corr(a_sm, a_lin, args.min_keys),
        }
        rec["entropy_gap"] = rec["entropy_linear"] - rec["entropy_softmax"]
        if i in tol:
            rec["tolerance_delta_nll"] = tol[i]
        per_layer.append(rec)
        print(
            f"  L{i:>2} H_sm={rec['entropy_softmax']:.3f} H_lin={rec['entropy_linear']:.3f} "
            f"gap={rec['entropy_gap']:+.3f} top1_sm={rec['top1_softmax']:.3f} "
            f"top1_lin={rec['top1_linear']:.3f} rankρ={rec['rank_corr']:+.3f}",
            flush=True,
        )

    # spikiness-gap vs tolerance correlation (the Hedgehog prediction)
    corr_gap_tol: float | None = None
    if tol:
        g = torch.tensor([r["entropy_gap"] for r in per_layer if "tolerance_delta_nll" in r])
        d = torch.tensor([r["tolerance_delta_nll"] for r in per_layer if "tolerance_delta_nll" in r])
        gc = g - g.mean()
        dc = d - d.mean()
        corr_gap_tol = float((gc * dc).sum() / (gc.norm() * dc.norm()).clamp_min(1e-12))

    report = {
        "model_dir": args.model_dir,
        "n_layer": p.n_layer,
        "eval_tokens": n_tok,
        "feature_map": "elu+1 (identity-init affine == fixed phi)",
        "per_layer": per_layer,
        "mean_entropy_gap": float(sum(r["entropy_gap"] for r in per_layer) / len(per_layer)),
        "mean_rank_corr": float(sum(r["rank_corr"] for r in per_layer) / len(per_layer)),
        "corr_entropy_gap_vs_tolerance": corr_gap_tol,
        "note": "entropy_gap>0 means linear attention is FLATTER than softmax (Hedgehog's "
        "spikiness loss). rank_corr<1 means the linear map reorders which keys matter. A positive "
        "corr_entropy_gap_vs_tolerance supports 'non-tolerant layers are the spiky ones'.",
    }
    (out / "feature_map_spikiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n[summary] mean entropy gap (lin-sm)={report['mean_entropy_gap']:+.3f} nats  "
        f"mean rankρ={report['mean_rank_corr']:+.3f}  "
        f"corr(gap, tolerance Δnll)={corr_gap_tol if corr_gap_tol is None else round(corr_gap_tol, 3)}",
        flush=True,
    )
    print(f"[done] wrote {out}/feature_map_spikiness.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
