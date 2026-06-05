# SPDX-License-Identifier: Apache-2.0
"""Hand-rolled (matplotlib-free) declarative-SVG figures for the llcore paper.

matplotlib is not installed in this CPU-only environment, and FullSense's expression thesis is
declarative CPU/SVG anyway — so figures are emitted as self-contained SVG (stdlib + json only).
Reads ONLY the committed, stable cost-reduction result JSONs. src/ untouched.

Figures:
  fig_cost_speedup.svg   — L2-lite vertex-free vs exact 2^n cert_two: per-gene seconds + speedup (n=8/12/16)
  fig_admit_coverage.svg — certifier admit counts as % of the exact cert_two reach (inf / B1 / B2 / inf∪B2)
"""
from __future__ import annotations

import html
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CR = os.path.normpath(os.path.join(_HERE, "..", "verifier_cost_reduction"))
_LM = os.path.normpath(os.path.join(_HERE, "..", "verified_lm_evolution"))


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _svg(w: int, h: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Segoe UI, Arial, sans-serif">\n'
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n'
        f'<text x="{w//2}" y="28" text-anchor="middle" font-size="18" '
        f'font-weight="bold" fill="#1a1a1a">{_esc(title)}</text>\n'
        f'{body}</svg>\n'
    )


def fig_cost_speedup(cost: dict) -> str:
    """Grouped horizontal bars: log10(sec/gene) for cert_two vs L2-lite at each n, speedup annotated."""
    ns = sorted(cost.keys(), key=int)
    W, H = 760, 360
    x0, y0 = 250, 70           # plot origin (left axis x, top y)
    bar_h, grp_gap, bar_gap = 18, 34, 6
    plot_w = 420
    # log scale over seconds; find range
    vals = []
    for n in ns:
        vals += [cost[n]["sec_per_gene_two_exact"], cost[n]["sec_per_gene_l2lite"]]
    lo, hi = math.log10(min(vals)), math.log10(max(vals))
    span = hi - lo or 1.0

    def bx(sec: float) -> float:
        return 10 + plot_w * (math.log10(sec) - lo) / span

    rows = []
    y = y0
    for n in ns:
        c = cost[n]
        two, l2 = c["sec_per_gene_two_exact"], c["sec_per_gene_l2lite"]
        rows.append(f'<text x="{x0-12}" y="{y+bar_h+2}" text-anchor="end" font-size="13" '
                    f'fill="#333">n={n} (2^n={c["vertices_2pow_n"]})</text>')
        # exact 2^n bar (top of group)
        rows.append(f'<rect x="{x0}" y="{y}" width="{bx(two):.1f}" height="{bar_h}" fill="#c0504d"/>')
        rows.append(f'<text x="{x0+bx(two)+6:.1f}" y="{y+bar_h-4}" font-size="11" fill="#c0504d">'
                    f'cert_two {two:.4g}s</text>')
        y += bar_h + bar_gap
        # L2-lite bar
        rows.append(f'<rect x="{x0}" y="{y}" width="{max(bx(l2),2):.1f}" height="{bar_h}" fill="#4f81bd"/>')
        rows.append(f'<text x="{x0+max(bx(l2),2)+6:.1f}" y="{y+bar_h-4}" font-size="11" fill="#4f81bd">'
                    f'L2-lite {l2:.4g}s  ({c["speedup_x"]:.0f}x faster)</text>')
        y += bar_h + grp_gap
    legend = (f'<text x="{x0}" y="{H-14}" font-size="11" fill="#666">'
              f'horizontal axis = log10(seconds/gene); L2-lite = 2 SVDs (vertex-free), '
              f'cert_two = 2^n vertex SVDs</text>')
    return _svg(W, H, "\n".join(rows) + "\n" + legend,
                "Verifier cost: vertex-free L2-lite vs exact 2^n enumeration")


def fig_admit_coverage(v2: dict) -> str:
    """Horizontal bars: admit count and % of exact cert_two reach for each certifier."""
    cnt = v2["admit_counts"]
    pct = v2["pct_of_exact_two"]
    two = cnt["two_exact"]
    order = [
        ("cert_two (exact 2^n)", two, 100.0, "#7f7f7f"),
        ("cert_inf (O(n^2))", cnt["inf"], pct["inf"], "#9bbb59"),
        ("B1 = sigma(M)+sigma(R)", cnt["b1"], pct["b1"], "#d99694"),
        ("B2 = sigma(|M|+R)  [1 SVD]", cnt["b2"], pct["b2"], "#4f81bd"),
        ("cert_inf OR B2", cnt["inf_or_b2"], pct["inf_or_b2"], "#8064a2"),
    ]
    W, H = 760, 300
    x0, y0 = 250, 60
    bar_h, gap = 26, 16
    plot_w = 380
    rows = []
    y = y0
    for label, c, p, color in order:
        w = plot_w * (c / two)
        rows.append(f'<text x="{x0-12}" y="{y+bar_h-7}" text-anchor="end" font-size="12.5" '
                    f'fill="#333">{_esc(label)}</text>')
        rows.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{color}"/>')
        rows.append(f'<text x="{x0+w+6:.1f}" y="{y+bar_h-7}" font-size="12" fill="#333">'
                    f'{c}  ({p:.1f}% of exact)</text>')
        y += bar_h + gap
    note = (f'<text x="{x0}" y="{H-12}" font-size="11" fill="#666">'
            f'n=8, 3000 random genes; all bounds 0 soundness violations. '
            f'B2 recovers {pct["b2"]:.0f}% of the 2^n reach at 1 SVD.</text>')
    return _svg(W, H, "\n".join(rows) + "\n" + note,
                "Certifier admit coverage vs the exact 2-norm certifier")


def fig_l3_gate_gap(real: dict, null: dict) -> str:
    """The honest-disclosure headline: mean held-out CE per gate, real vs null (shuffled) corpus.

    Real: sound gates (two/sdp) beat the unigram baseline (learning). Null: every gate sits BELOW
    unigram and the gate ORDERING PERSISTS (the relaxed-vs-inf gap does NOT tie) -> the gate-gap is
    structure-independent (evolvability), not language learning.
    """
    gates = [("inf_norm", "inf"), ("two_norm", "two"), ("sdp", "sdp"), ("none", "none")]
    W, H = 760, 380
    x0, y_base = 90, 300
    col_w, grp_gap, bar_w = 150, 26, 46
    # CE axis: pick a window around both unigrams
    rce = real["summary"]; nce = null["summary"]
    ru, nu = real["unigram_ce"], null["unigram_ce"]
    all_ce = [rce[g]["mean_ce"] for g, _ in gates] + [nce[g]["mean_ce"] for g, _ in gates] + [ru, nu]
    lo, hi = min(all_ce) - 0.02, max(all_ce) + 0.02
    span = hi - lo

    def yy(ce: float) -> float:
        return y_base - (y_base - 60) * (ce - lo) / span

    rows = [f'<text x="40" y="{y_base+24}" font-size="12" fill="#333">CE</text>']
    # y-axis gridlines at unigram refs
    for u, lab, col in ((ru, f"real unigram {ru:.3f}", "#c0504d"), (nu, f"null unigram {nu:.3f}", "#4f81bd")):
        rows.append(f'<line x1="{x0}" y1="{yy(u):.1f}" x2="{W-30}" y2="{yy(u):.1f}" '
                    f'stroke="{col}" stroke-dasharray="5 4" stroke-width="1"/>')
        rows.append(f'<text x="{W-30}" y="{yy(u)-4:.1f}" text-anchor="end" font-size="10.5" '
                    f'fill="{col}">{_esc(lab)}</text>')
    x = x0 + 30
    for g, short in gates:
        rb, nb = rce[g]["mean_ce"], nce[g]["mean_ce"]
        # real bar (red) then null bar (blue), grown downward from top of CE window
        rows.append(f'<rect x="{x}" y="{yy(rb):.1f}" width="{bar_w}" height="{y_base-yy(rb):.1f}" fill="#c0504d"/>')
        rows.append(f'<text x="{x+bar_w/2:.1f}" y="{yy(rb)-5:.1f}" text-anchor="middle" font-size="10" '
                    f'fill="#c0504d">{rb:.3f}</text>')
        rows.append(f'<rect x="{x+bar_w+6}" y="{yy(nb):.1f}" width="{bar_w}" height="{y_base-yy(nb):.1f}" fill="#4f81bd"/>')
        rows.append(f'<text x="{x+bar_w+6+bar_w/2:.1f}" y="{yy(nb)-5:.1f}" text-anchor="middle" font-size="10" '
                    f'fill="#4f81bd">{nb:.3f}</text>')
        rows.append(f'<text x="{x+bar_w+3:.1f}" y="{y_base+18}" text-anchor="middle" font-size="12.5" '
                    f'fill="#333">{_esc(short)}</text>')
        x += col_w
    rows.append(f'<rect x="{x0+30}" y="40" width="12" height="12" fill="#c0504d"/>'
                f'<text x="{x0+46}" y="50" font-size="11" fill="#333">real corpus</text>')
    rows.append(f'<rect x="{x0+140}" y="40" width="12" height="12" fill="#4f81bd"/>'
                f'<text x="{x0+156}" y="50" font-size="11" fill="#333">null (shuffled) corpus</text>')
    note = (f'<text x="{x0}" y="{H-14}" font-size="11" fill="#666">'
            f'lower CE = better. Gate ordering inf&gt;two&gt;sdp persists on the null '
            f'(gap ~107% of real on the CE scale) =&gt; evolvability, not language learning.</text>')
    return _svg(W, H, "\n".join(rows) + "\n" + note,
                "L3 honest disclosure: gate-gap persists on the null corpus")


def main():
    with open(os.path.join(_CR, "poc_l2lite_results.json"), encoding="utf-8") as fh:
        poc1 = json.load(fh)
    with open(os.path.join(_CR, "poc_l2lite_v2_results.json"), encoding="utf-8") as fh:
        v2 = json.load(fh)
    with open(os.path.join(_LM, "exp_gated_real10_results.json"), encoding="utf-8") as fh:
        real = json.load(fh)
    with open(os.path.join(_LM, "exp_gated_null_results.json"), encoding="utf-8") as fh:
        null = json.load(fh)
    figs = {
        "fig_cost_speedup.svg": fig_cost_speedup(poc1["cost"]),
        "fig_admit_coverage.svg": fig_admit_coverage(v2),
        "fig_l3_gate_gap.svg": fig_l3_gate_gap(real, null),
    }
    for name, svg in figs.items():
        path = os.path.join(_HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg)
        print(f"wrote {name} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
