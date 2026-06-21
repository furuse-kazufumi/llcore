# SPDX-License-Identifier: Apache-2.0
"""Render an honest-disclosure Markdown report from a ``nas_pareto.py`` result JSON.

`scripts/nas_pareto.py` writes ``out/<run>/nas_pareto.json``: the greedy/evolved Pareto frontiers,
their 2-D hypervolumes, and (with ``--proxy-v2``) a rigorous tier whose headline verdict is computed
on a FRESH disjoint holdout pool. This script turns that JSON into a reviewer-facing Markdown report
that keeps every honest-disclosure guard visible:

* the headline uses **holdout** Δnll, never the optimistically biased selection Δnll;
* ``optimism_gap = selection - heldout`` is shown per frontier point (winner's-curse magnitude);
* the hypervolume-gain claim is quoted with its 95% bootstrap CI and ``p_memetic_wins``;
* ``scope`` is pinned to ``next_token_nll_proxy`` and any conversational claim is explicitly refused;
* the context sweep exposes regime dependence (a win at L=1024 that vanishes at L=2048 is disclosed);
* CI-reliability / few-window caveats and missing optional probes are stated, not hidden.

Read-only: it never imports torch and never touches the running NAS process or its output dir beyond
reading the JSON. Safe to run while the search is still in progress (it just reports "not finished").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _fmt(x: Any, nd: int = 4) -> str:
    """Format a float to ``nd`` decimals; pass through ``None``/non-numerics verbatim."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return "—" if x is None else str(x)


def _frontier_table(points: list[dict[str, Any]]) -> list[str]:
    """Render a ``[{pct_mem_saved, delta_nll}]`` frontier as a Markdown table (sorted by saving)."""
    if not points:
        return ["_(empty frontier)_", ""]
    rows = ["| % mem saved | Δnll (lower=better) |", "|---:|---:|"]
    for p in sorted(points, key=lambda d: d.get("pct_mem_saved", 0.0)):
        dn = p.get("delta_nll")
        dn_s = f"{dn:+.4f}" if isinstance(dn, (int, float)) and not isinstance(dn, bool) else _fmt(dn)
        rows.append(f"| {_fmt(p.get('pct_mem_saved'), 1)} | {dn_s} |")
    rows.append("")
    return rows


def _holdout_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render the rigorous-tier frontier_holdout rows (the headline uses delta_nll_heldout)."""
    if not rows:
        return ["_(no holdout frontier rows)_", ""]
    out = [
        "| % mem saved | Δnll selection | Δnll **holdout** | optimism gap | 95% CI | p_worse | pos_frac | p_sign |",
        "|---:|---:|---:|---:|:--:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda d: d.get("pct", 0.0)):
        ci = f"{_fmt(r.get('ci_lo'))}..{_fmt(r.get('ci_hi'))}"
        out.append(
            f"| {_fmt(r.get('pct'), 1)} | {_fmt(r.get('delta_nll_selection'))} | "
            f"{_fmt(r.get('delta_nll_heldout'))} | {_fmt(r.get('optimism_gap'))} | {ci} | "
            f"{_fmt(r.get('p_worse'), 3)} | {_fmt(r.get('pos_frac'), 3)} | {_fmt(r.get('p_sign'), 3)} |"
        )
    out.append("")
    return out


def _context_sweep_table(sweep: dict[str, Any]) -> list[str]:
    """Render the per-context-length paired-Δnll sweep on the most aggressive genome."""
    if not sweep:
        return ["_(no context sweep)_", ""]
    out = [
        "| context L | mean Δnll | 95% CI | p_worse | pos_frac | p_sign | n_windows |",
        "|---:|---:|:--:|---:|---:|---:|---:|",
    ]
    for length in sorted(sweep, key=lambda s: int(s)):
        v = sweep[length]
        ci = f"{_fmt(v.get('ci_lo'))}..{_fmt(v.get('ci_hi'))}"
        out.append(
            f"| {length} | {_fmt(v.get('mean'))} | {ci} | {_fmt(v.get('p_worse'), 3)} | "
            f"{_fmt(v.get('pos_frac'), 3)} | {_fmt(v.get('p_sign'), 3)} | {_fmt(v.get('n_windows'), 0)} |"
        )
    out.append("")
    return out


def render(report: dict[str, Any]) -> str:
    """Build the full Markdown honest-disclosure report from a nas_pareto.json dict."""
    L: list[str] = []
    L.append("# NAS Pareto — honest-disclosure report")
    L.append("")
    L.append(f"- model: `{report.get('model_dir')}` ({report.get('n_layer')} layers)")
    L.append(f"- mixers: {report.get('mixers')}")
    L.append(f"- base all-softmax nll: {_fmt(report.get('base_nll'))} "
             f"(ppl ≈ {_fmt(__import__('math').exp(report['base_nll']), 2) if isinstance(report.get('base_nll'), (int, float)) else '—'})")
    L.append(f"- real genome evals: {report.get('real_evals')}")
    if "elapsed_s" in report:
        L.append(f"- search wall time: {_fmt(report.get('elapsed_s'), 0)} s")
    L.append("")

    # --- zero-shot frontiers (the v1-compatible scalar layer) ---
    L.append("## Zero-shot frontiers (selection-window scalar)")
    L.append("")
    L.append(f"**verdict (greedy vs memetic):** {report.get('verdict')}")
    L.append("")
    hv = report.get("hypervolume", {})
    L.append(f"hypervolume — greedy {_fmt(hv.get('greedy'))}, evolved {_fmt(hv.get('evolved'))}")
    L.append("")
    L.append("### Greedy frontier")
    L.extend(_frontier_table(report.get("greedy_frontier", [])))
    L.append("### Evolved (memetic NSGA-II) frontier")
    L.extend(_frontier_table(report.get("evolved_frontier", [])))

    # --- distillation right-shift (only with --distill) ---
    if report.get("right_shift"):
        rs = report["right_shift"]
        L.append("## Distillation right-shift")
        L.append("")
        L.append(f"{rs.get('verdict')}  (zero-shot HV {_fmt(rs.get('zero_shot_hv'))} → "
                 f"distilled HV {_fmt(rs.get('distilled_hv'))}, shift {_fmt(rs.get('shift_pct'), 1)}%)")
        L.append("")

    # --- proxy-v2 rigorous tier (the headline) ---
    pv = report.get("proxy_v2")
    if not pv:
        L.append("> proxy-v2 rigorous tier absent (run was v1 or not yet finished).")
        L.append("")
        return "\n".join(L)

    L.append("## proxy-v2 rigorous tier (HEADLINE — holdout numbers)")
    L.append("")
    L.append(f"- **scope:** `{pv.get('scope')}` — conversational claim: "
             f"`{pv.get('conversational_claim')}` (a conversational quality claim must live in a "
             f"separate disclosed generation eval; it is NOT inferred from these perplexity proxies).")
    L.append(f"- inner-loop context: {pv.get('inner_context')} tok; "
             f"holdout offset {pv.get('holdout_offset')} tok, {_fmt(pv.get('holdout_windows'), 0)} windows "
             f"(disjoint from the fast pool's {_fmt(pv.get('fast_windows'), 0)} windows).")
    tau = pv.get("proxy_vs_judge_tau")
    L.append(f"- proxy-vs-judge Kendall τ: {_fmt(tau, 2) if tau is not None else '—'} "
             f"(<0.7 downgrades any positive verdict to 'suggestive').")
    L.append("")

    verdict = pv.get("verdict", {})
    L.append("### Verdict")
    L.append("")
    L.append(f"> **{verdict.get('memetic_vs_greedy')}**")
    L.append(">")
    L.append(f"> confidence: **{verdict.get('confidence')}** — {verdict.get('notes')}")
    if "ci_reliability" in verdict:
        L.append(">")
        L.append(f"> ⚠ {verdict['ci_reliability']}")
    L.append("")

    gain = pv.get("hv_gain_ci", {})
    L.append("### Hypervolume gain (memetic − greedy), holdout")
    L.append("")
    L.append(f"+{_fmt(gain.get('gain_pct_mean'), 1)}% HV  "
             f"(95% CI {_fmt(gain.get('ci_lo'), 1)}..{_fmt(gain.get('ci_hi'), 1)}%, "
             f"p_memetic_wins {_fmt(gain.get('p_memetic_wins'), 3)}). "
             f"The win fires only when CI_lo > 0.")
    L.append("")

    L.append("### Frontier holdout (winner's-curse removed)")
    L.append("")
    L.append("Headline Δnll is the **holdout** column; `optimism_gap = selection − holdout` is the "
             "selection bias. If max gap exceeds the CI noise floor the verdict is suppressed.")
    L.append("")
    L.extend(_holdout_table(pv.get("frontier_holdout", [])))

    L.append("### Context-length sweep (regime dependence)")
    L.append("")
    L.append(f"On the most aggressive genome ({_fmt(pv.get('aggressive_genome_pct'), 1)}% mem saved). "
             "A win at short L that vanishes at long L is the constant-state failure mode (cf. SUPRA).")
    L.append("")
    L.extend(_context_sweep_table(pv.get("context_sweep", {})))

    # --- attention-KL diagnostic ---
    akl = pv.get("attention_kl")
    if akl:
        L.append("### Attention-KL fidelity (diagnostic, ≤256 tok, NOT in fitness)")
        L.append("")
        L.append(f"forward KL(softmax‖student) — mean {_fmt(akl.get('mean'))}, max {_fmt(akl.get('max'))}, "
                 f"sum {_fmt(akl.get('sum'))} over converted layers.")
        pl = akl.get("per_layer") or []
        if pl:
            L.append("")
            L.append("| layer | KL (nats) |")
            L.append("|---:|---:|")
            for e in pl:
                L.append(f"| {_fmt(e.get('layer'), 0)} | {_fmt(e.get('kl'))} |")
        L.append("")

    # --- needle / passkey retrieval horizon ---
    needle = pv.get("needle")
    L.append("### Long-context retrieval (needle / passkey)")
    L.append("")
    if not needle:
        L.append("_not run this study (`--needle` off). The constant-state long-range copy failure "
                 "mode is therefore UNTESTED — disclose this gap rather than implying retrieval is fine._")
    else:
        hz = needle.get("horizon")
        L.append(f"failure horizon: {'never failed in tested range' if hz is None else str(hz) + ' tok'} "
                 "(shortest length where the genome drops retrieval while all-softmax still succeeds).")
        bd = needle.get("by_depth") or {}
        if bd:
            L.append("")
            L.append("| length:depth | argmax_acc | control_acc | mean_logprob |")
            L.append("|:--|---:|---:|---:|")
            for k in sorted(bd):
                v = bd[k]
                L.append(f"| {k} | {_fmt(v.get('argmax_acc'), 3)} | {_fmt(v.get('control_acc'), 3)} | "
                         f"{_fmt(v.get('mean_logprob'))} |")
    L.append("")

    # --- cross-corpus generalization ---
    cc = pv.get("cross_corpus")
    if cc:
        L.append("### Cross-corpus generalization holdout")
        L.append("")
        cg = cc.get("hv_gain_ci", {})
        L.append(f"on `{cc.get('corpus')}` ({_fmt(cc.get('n_windows'), 0)} windows): "
                 f"+{_fmt(cg.get('gain_pct_mean'), 1)}% HV "
                 f"(95% CI {_fmt(cg.get('ci_lo'), 1)}..{_fmt(cg.get('ci_hi'), 1)}%).")
        L.append("")

    # --- prior-art positioning (RAD-grounded differentiation; always rendered) ---
    L.append("### Prior-art positioning (why this is not reinventing the wheel)")
    L.append("")
    L.append("Multi-objective Pareto NAS, novelty/quality-diversity search, and supernet/training-free "
             "fitness proxies are established (e.g. MTF-PDNS, arXiv:2407.20656). That literature already "
             "warns that **training-free / proxy metrics often disagree with actual performance** "
             "(the proxy-noise trade-off).")
    L.append("")
    L.append("The differentiator here is **not a new search operator** but a disclosure layer that "
             "*quantifies the proxy's uncertainty and lets it govern the verdict*:")
    L.append("")
    L.append("1. paired multi-window bootstrap CI on the proxy itself;")
    L.append("2. fresh disjoint holdout to remove winner's-curse, with `optimism_gap` disclosed;")
    L.append("3. proxy-vs-judge Kendall τ < 0.7 downgrades any positive verdict to 'suggestive';")
    L.append("4. the HV-gain win fires **only** when CI_lo > 0 (not on a point estimate);")
    L.append("5. memetic ≈ greedy is reported as an honest negative (separable landscape), not hidden.")
    L.append("")
    L.append("So where prior work concedes 'proxies are noisy', this run *measures the noise and "
             "suppresses the claim accordingly*.")
    L.append("")

    L.append("---")
    L.append("")
    L.append(f"_note: {pv.get('note')}_")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", type=Path, help="path to nas_pareto.json")
    ap.add_argument("-o", "--out", type=Path, default=None, help="write Markdown here (default: stdout)")
    args = ap.parse_args(argv)

    # the report contains Δ / em-dash / ‖; force utf-8 stdout so it survives Windows cp932 consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.json.exists():
        print(f"error: {args.json} does not exist (NAS run not finished?)", file=sys.stderr)
        return 2
    try:
        report = json.loads(args.json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: {args.json} is not valid JSON ({e}) — run may still be writing it", file=sys.stderr)
        return 2
    md = render(report)
    if args.out is not None:
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(md)} chars)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
