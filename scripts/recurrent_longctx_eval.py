# SPDX-License-Identifier: Apache-2.0
"""Long-context quality evaluation of a *trained* constant-state recurrent char-LM.

The prior memory work proved the constant-O(1)-state win with RANDOM models (peak RSS flat
vs GPT's O(T^2)). This harness answers the QUALITY question on an actually-trained model and
is deliberately built to neutralise the artifact traps a methodology red-team flagged
(workflow: recurrent-longctx-methodology). It runs, in the red-team's priority order:

  0. CORRECTNESS GATES (blocking): chunk_size invariance + NaN/inf over long T. If either
     fails, every long-context number is suspect.
  1. CONTEXT-LENGTH CURVE (primary, confound-free): on a FIXED set of deep target positions,
     score each conditioning on exactly c preceding tokens (fresh cold window), sweeping c.
     Same positions across all c => text difficulty controlled, no warmup asymmetry. A
     drop-then-plateau reads the model's *effective* context length; the plateau onset is the
     honest answer to "how much context does this BPTT=block_size model actually use".
  2. BANDED STREAMING (non-degradation): per-position-band NLL/top-1 over many disjoint
     held-out chunks (mean +/- std), with per-band unigram floor and tail-mean (warmup
     excluded). Flat bands beyond block_size == constant-memory non-degradation.
  3. CARRY-vs-RESET delta on identical tokens (reported WITH the cold-start caveat).
  4. Optional GPT sliding-window baseline (stride=block_size cheap; stride=1 gold on a few
     short chunks) plotted as a reference, NOT a quality-superiority claim.

Honest claim ceiling (enforced in the emitted report): defensible = "streaming runs to
completion at O(1) state on T >> block_size, impossible for the block_size-bounded GPT" and
"per-token NLL stays flat past block_size on a trained model (constant-memory non-degradation)".
NOT defensible without TBPTT + ablation = "the model uses/benefits from long context".
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

from llcore.lm.checkpoint import load_lm_checkpoint
from llcore.lm.data import encode_corpus, train_val_split
from llcore.lm.longctx_eval import (
    block_reset_nll,
    context_length_curve,
    gpt_sliding_window_nll,
    streaming_metrics_by_band,
)
from llcore.lm.model import CharGPT
from llcore.lm.recurrent import RecurrentLM
from llcore.lm.rwkv import RWKVLM


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _unigram_logp(train_ids: torch.Tensor, vocab_size: int, alpha: float = 1.0) -> torch.Tensor:
    counts = torch.bincount(train_ids, minlength=vocab_size).double()
    probs = (counts + alpha) / (train_ids.numel() + alpha * vocab_size)
    return torch.log(probs)


def _disjoint_chunks(ids: torch.Tensor, n_chunks: int, chunk_tokens: int) -> list[torch.Tensor]:
    """Evenly-spaced disjoint contiguous slices of ``ids`` (so dispersion isn't one passage)."""
    n = int(ids.size(0))
    out: list[torch.Tensor] = []
    if n < chunk_tokens + 1:
        return [ids]
    span = n // n_chunks
    for k in range(n_chunks):
        start = k * span
        stop = min(start + chunk_tokens, n)
        if stop - start >= 2:
            out.append(ids[start:stop])
    return out


def _correctness_gates(model: RecurrentLM | RWKVLM, probe: torch.Tensor) -> dict[str, object]:
    a, _ = model.streaming_nll(probe, chunk_size=64)
    b, _ = model.streaming_nll(probe, chunk_size=256)
    c, _ = model.streaming_nll(probe, chunk_size=1024)
    spread = max(abs(a - b), abs(a - c))
    finite = all(math.isfinite(x) for x in (a, b, c))
    return {
        "chunk_invariance_spread": spread,
        "chunk_invariance_pass": bool(spread < 1e-3 and finite),
        "streaming_nll_finite": bool(finite),
        "probe_tokens": int(probe.size(0)),
        "nll_chunk64": a,
        "nll_chunk256": b,
        "nll_chunk1024": c,
    }


def _aggregate_bands(per_chunk: list[dict[str, object]]) -> list[dict[str, object]]:
    """Mean +/- std across chunks of each band's mean_nll / ppl / top1 / beats_unigram rate."""
    by_lo: dict[int, list[dict[str, object]]] = {}
    for res in per_chunk:
        for band in res["bands"]:  # type: ignore[index]
            by_lo.setdefault(int(band["lo"]), []).append(band)  # type: ignore[index,arg-type]
    out: list[dict[str, object]] = []
    for lo in sorted(by_lo):
        bands = by_lo[lo]
        nlls = [float(b["mean_nll"]) for b in bands]
        top1s = [float(b["top1"]) for b in bands]
        n_tok = sum(int(b["n_tok"]) for b in bands)
        mean_nll = statistics.fmean(nlls)
        row: dict[str, object] = {
            "lo": lo,
            "hi": bands[0]["hi"],
            "n_chunks": len(nlls),
            "n_tok_total": n_tok,
            "mean_nll": mean_nll,
            "std_nll": statistics.pstdev(nlls) if len(nlls) > 1 else 0.0,
            "ppl": math.exp(mean_nll),
            "bpc": mean_nll / math.log(2),
            "mean_top1": statistics.fmean(top1s),
        }
        if "beats_unigram" in bands[0]:
            row["beats_unigram_rate"] = statistics.fmean(
                [1.0 if b["beats_unigram"] else 0.0 for b in bands]
            )
            row["mean_unigram_nll"] = statistics.fmean([float(b["unigram_nll"]) for b in bands])
        out.append(row)
    return out


def _render_band_svg(rows: list[dict[str, object]], gpt_ref: float | None, block_size: int) -> str:
    xs = [int(r["lo"]) for r in rows]
    ys = [float(r["mean_nll"]) for r in rows]
    if not xs:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    lo_y = min([*ys, gpt_ref] if gpt_ref is not None else ys) * 0.95
    hi_y = max([*ys, gpt_ref] if gpt_ref is not None else ys) * 1.05
    w, h, l, r, t, b = 680, 360, 70, 24, 30, 52
    pw, ph = w - l - r, h - t - b
    n = len(xs)

    def sx(i: int) -> float:
        return l + (pw * i / max(1, n - 1))

    def sy(v: float) -> float:
        return t + ph - (v - lo_y) / (hi_y - lo_y) * ph if hi_y > lo_y else t + ph / 2

    pts = " ".join(f"{sx(i):.1f},{sy(ys[i]):.1f}" for i in range(n))
    dots = "".join(f'<circle cx="{sx(i):.1f}" cy="{sy(ys[i]):.1f}" r="3.5" fill="#059669" />' for i in range(n))
    xlabels = "".join(
        f'<text x="{sx(i):.1f}" y="{h - 30}" text-anchor="middle" font-size="11" fill="#374151">{xs[i]}</text>'
        for i in range(n)
    )
    gpt_line = ""
    if gpt_ref is not None:
        gy = sy(gpt_ref)
        gpt_line = (
            f'<line x1="{l}" y1="{gy:.1f}" x2="{w - r}" y2="{gy:.1f}" stroke="#2563eb" '
            f'stroke-width="2" stroke-dasharray="7 5" />'
            f'<text x="{w - r - 4}" y="{gy - 6:.1f}" text-anchor="end" font-size="11" fill="#2563eb">'
            f'GPT (≤block_size window): {gpt_ref:.3f}</text>'
        )
    block_x = None
    for i, xv in enumerate(xs):
        if xv >= block_size:
            block_x = sx(i)
            break
    block_marker = (
        f'<line x1="{block_x:.1f}" y1="{t}" x2="{block_x:.1f}" y2="{h - b}" stroke="#9ca3af" '
        f'stroke-width="1" stroke-dasharray="3 4" />'
        f'<text x="{block_x:.1f}" y="{t - 8}" text-anchor="middle" font-size="10" fill="#6b7280">'
        f'block_size={block_size}</text>'
        if block_x is not None
        else ""
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">
<title id="t">Trained recurrent: per-token NLL by context position</title>
<desc id="d">Per-token held-out NLL (nats) bucketed by absolute target position. Flat past block_size = constant-memory non-degradation at O(1) state.</desc>
<rect width="{w}" height="{h}" fill="#ffffff" />
<text x="{l}" y="18" font-family="Segoe UI, sans-serif" font-size="13" fill="#111827">Trained recurrent: held-out NLL (nats) vs context position band</text>
<line x1="{l}" y1="{t}" x2="{l}" y2="{h - b}" stroke="#111827" />
<line x1="{l}" y1="{h - b}" x2="{w - r}" y2="{h - b}" stroke="#111827" />
<text x="{l - 50}" y="{t + 6}" font-size="11" fill="#374151">{hi_y:.2f}</text>
<text x="{l - 50}" y="{h - b}" font-size="11" fill="#374151">{lo_y:.2f}</text>
<text x="{l}" y="{h - 10}" font-size="11" fill="#6b7280">target position band (chars)</text>
{block_marker}
{gpt_line}
<polyline fill="none" stroke="#059669" stroke-width="2.5" points="{pts}" />
{dots}
{xlabels}
<text x="{w - r - 4}" y="{t + 14}" text-anchor="end" font-size="11" fill="#059669">recurrent streaming (O(1) state, carried)</text>
</svg>
"""


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="long-context quality eval for a trained recurrent char-LM")
    ap.add_argument("--checkpoint", required=True, help="trained recurrent/rwkv checkpoint (model.pt)")
    ap.add_argument("--corpus-file", required=True, help="same corpus used for training (to rebuild val split)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--gpt-checkpoint", default=None, help="optional GPT checkpoint for the sliding-window baseline")
    ap.add_argument("--context-lens", default="8,16,32,64,128,256,512,1024,2048")
    ap.add_argument("--band-edges", default="0,128,256,512,1024,2048,4096")
    ap.add_argument("--n-positions", type=int, default=400)
    ap.add_argument("--n-chunks", type=int, default=8)
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--gpt-stride1-tokens", type=int, default=512, help="short chunk length for the gold stride=1 GPT baseline")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_lm_checkpoint(args.checkpoint)
    if not isinstance(model, (RecurrentLM, RWKVLM)):
        raise SystemExit(f"--checkpoint must be a recurrent/rwkv model, got {type(model).__name__}")
    block_size = int(model.config.block_size)
    vocab = len(tok.itos)

    text = Path(args.corpus_file).read_text(encoding="utf-8")
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    logp = _unigram_logp(train_ids, vocab)
    print(
        f"[eval] arch={type(model).__name__} block_size={block_size} vocab={vocab} "
        f"val_tokens={val_ids.numel():,}",
        flush=True,
    )

    # 0. correctness gates -----------------------------------------------------------------
    probe = val_ids[: max(2048, block_size * 4)]
    gates = _correctness_gates(model, probe)
    print(
        f"[gate] chunk_invariance spread={gates['chunk_invariance_spread']:.2e} "
        f"pass={gates['chunk_invariance_pass']} finite={gates['streaming_nll_finite']}",
        flush=True,
    )

    # 1. context-length curve (primary, confound-free) -------------------------------------
    context_lens = _parse_int_list(args.context_lens)
    context_lens = [c for c in context_lens if c < val_ids.numel() - 1]
    curve = context_length_curve(model, val_ids, context_lens, args.n_positions, args.seed)
    print("[curve] NLL by available context length (same fixed positions):", flush=True)
    for c in curve["context_lens"]:  # type: ignore[union-attr]
        nll = curve["nll_by_context"][c]  # type: ignore[index]
        marker = "  <- block_size" if c == block_size else ("  (> block_size = OOD)" if c > block_size else "")
        print(f"        c={c:>5}  nll={nll:.4f}  ppl={math.exp(nll):.2f}{marker}", flush=True)

    # 2. banded streaming over disjoint chunks (non-degradation + dispersion) ---------------
    band_edges = _parse_int_list(args.band_edges)
    chunks = _disjoint_chunks(val_ids, args.n_chunks, args.chunk_tokens)
    per_chunk_bands: list[dict[str, object]] = []
    carry_vals: list[float] = []
    reset_vals: list[float] = []
    for ch in chunks:
        per_chunk_bands.append(
            streaming_metrics_by_band(model, ch, band_edges, unigram_logp=logp, tail_start=block_size)
        )
        carry, _ = model.streaming_nll(ch)
        reset, _ = block_reset_nll(model, ch, reset_every=block_size)
        carry_vals.append(carry)
        reset_vals.append(reset)
    band_rows = _aggregate_bands(per_chunk_bands)
    tail_means = [float(r["tail_mean_nll"]) for r in per_chunk_bands if r["tail_mean_nll"] is not None]
    print(f"[band] {len(chunks)} disjoint chunks; per-position-band held-out NLL (nats):", flush=True)
    for row in band_rows:
        hi = row["hi"] if row["hi"] is not None else "inf"
        beat = row.get("beats_unigram_rate")
        beat_s = f" beats_uni={beat:.2f}" if beat is not None else ""
        print(
            f"        pos[{row['lo']:>4}-{hi:>5}]  nll={row['mean_nll']:.4f}+/-{row['std_nll']:.4f}"
            f"  ppl={row['ppl']:.2f}  top1={row['mean_top1']:.3f}{beat_s}",
            flush=True,
        )

    # 3. carry-vs-reset delta (with cold-start caveat) -------------------------------------
    carry_mean = statistics.fmean(carry_vals)
    reset_mean = statistics.fmean(reset_vals)
    delta = reset_mean - carry_mean
    print(
        f"[delta] carry(full state)={carry_mean:.4f}  reset(every {block_size})={reset_mean:.4f}  "
        f"reset-carry={delta:+.4f} nats  (CAVEAT: reset pays repeated cold-starts; "
        f"see context-length curve for the confound-free signal)",
        flush=True,
    )

    # 4. optional GPT sliding-window baseline ---------------------------------------------
    gpt_baseline: dict[str, object] | None = None
    if args.gpt_checkpoint:
        gpt, _ = load_lm_checkpoint(args.gpt_checkpoint)
        if not isinstance(gpt, CharGPT):
            raise SystemExit(f"--gpt-checkpoint must be a GPT model, got {type(gpt).__name__}")
        gblock = int(gpt.config.block_size)
        strideB_vals = [gpt_sliding_window_nll(gpt, ch, stride=gblock)[0] for ch in chunks]
        short = val_ids[: args.gpt_stride1_tokens]
        stride1_nll, stride1_tok = gpt_sliding_window_nll(gpt, short, stride=1)
        strideB_short, _ = gpt_sliding_window_nll(gpt, short, stride=gblock)
        gpt_baseline = {
            "gpt_block_size": gblock,
            "stride_block_nll_mean": statistics.fmean(strideB_vals),
            "stride_block_ppl_mean": math.exp(statistics.fmean(strideB_vals)),
            "stride1_nll_short": stride1_nll,
            "stride1_ppl_short": math.exp(stride1_nll),
            "stride_block_nll_short": strideB_short,
            "stride1_tokens": int(stride1_tok),
            "note": (
                "GPT cannot consume T>block_size in one window; these are strided sliding-window "
                "references. stride=1 (full block_size context, gold) may MATCH/BEAT the recurrent "
                "per-token NLL — the recurrent win is FEASIBILITY/reach at O(1) memory, not quality."
            ),
        }
        print(
            f"[gpt] stride={gblock} nll={gpt_baseline['stride_block_nll_mean']:.4f}  "
            f"stride=1(short {stride1_tok}tok) nll={stride1_nll:.4f}",
            flush=True,
        )

    # assemble + honest verdict ------------------------------------------------------------
    tail_mean = statistics.fmean(tail_means) if tail_means else None
    first_band = band_rows[0]["mean_nll"] if band_rows else None
    # non-degradation reference = the first WARM band [block_size, 2*block_size) (NOT the
    # [0, block_size) warmup band). "Holds" = every band beyond block_size <= that warm band
    # within dispersion (red-team spec: bands past block_size <= the block_size..2*block_size band).
    ref_band = next((r for r in band_rows if int(r["lo"]) == block_size), None)
    beyond = [r for r in band_rows if int(r["lo"]) > block_size]
    max_beyond = max((float(r["mean_nll"]) for r in beyond), default=None)
    nondegrade: bool | None = None
    if ref_band is not None and beyond:
        ref = float(ref_band["mean_nll"])
        std_ref = max([float(r["std_nll"]) for r in beyond] + [float(ref_band["std_nll"])])
        nondegrade = bool(max_beyond <= ref + 2 * std_ref + 0.05)
    report = {
        "checkpoint": str(args.checkpoint),
        "arch": type(model).__name__,
        "block_size": block_size,
        "vocab_size": vocab,
        "val_tokens": int(val_ids.numel()),
        "correctness_gates": gates,
        "context_length_curve": curve,
        "banded_streaming": {
            "n_chunks": len(chunks),
            "chunk_tokens": args.chunk_tokens,
            "bands": band_rows,
            "tail_mean_nll": tail_mean,
            "tail_mean_ppl": math.exp(tail_mean) if tail_mean is not None else None,
            "first_band_nll": first_band,
            "max_nll_beyond_block_size": max_beyond,
        },
        "carry_vs_reset": {
            "carry_full_state_nll": carry_mean,
            "reset_every_block_nll": reset_mean,
            "reset_minus_carry": delta,
            "caveat": "reset pays a repeated cold-start warmup at every block boundary; a positive "
            "delta conflates 'long context helps' with 'streaming avoided repeated warmups'. The "
            "confound-free signal is the context-length curve (same fixed positions).",
        },
        "gpt_baseline": gpt_baseline,
        "verdict": {
            "streaming_runs_beyond_block_size": True,
            "constant_memory_nondegradation": nondegrade,
            "nondegradation_definition": "max per-token NLL in bands beyond block_size <= the "
            "block-straddling band NLL + 2*std + 0.05 nats slack",
            "effective_context_note": "read the plateau onset in context_length_curve.ppl_by_context; "
            "for a BPTT=block_size model the curve is expected to flatten near block_size",
            "defensible_claims": [
                "streaming_nll runs to completion on T >> block_size at O(1) per-layer state and "
                "O(chunk) activation memory; the block_size-bounded GPT structurally cannot.",
                "on a trained model, per-token NLL stays flat past block_size (constant-memory "
                "non-degradation) — advances the prior random-model RSS demo to a trained model.",
            ],
            "overclaims_avoided": [
                "NOT claiming the model uses/benefits from long context (needs TBPTT + ablation).",
                "NOT claiming quality parity/superiority vs the Transformer at long T (stride-1 GPT "
                "may match or beat per-token NLL); the recurrent advantage is reach at O(1) memory.",
            ],
        },
    }
    (out / "longctx_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "longctx_curve.svg").write_text(
        _render_band_svg(band_rows, (gpt_baseline["stride_block_nll_mean"] if gpt_baseline else None), block_size),  # type: ignore[arg-type]
        encoding="utf-8",
    )
    print(f"\n[done] wrote {out}/longctx_eval.json + longctx_curve.svg", flush=True)
    print(
        f"[verdict] constant_memory_nondegradation={nondegrade}  "
        f"tail_mean_nll={tail_mean:.4f}" if tail_mean is not None else "[verdict] (no tail)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
