# SPDX-License-Identifier: Apache-2.0
"""Phase 2 demo (F11, 動きで魅せる): 「経験的 gate は発散を見逃す / sound certificate は見抜く」。

EVOLVABLE_LLM_PLAN_2026_06_09.md §⑫.2 + PHASE_2_VERDICT.md §1/§5。本研究の核心 = verified-plasticity の
存在意義を **動きで** 見せる honest デモ。

★honest な再設計理由(重要): 計画当初の「無 gate ρ→1.95 出力ノルム発散」アニメは **この基質では物理的に
起きない**。coupled 基質 `s'=decay⊙s+(1-decay)⊙tanh(Ws+x)` は tanh で状態が常時有界で、しかも **単一軌道の
摂動感度すら ρ≥1 で発散しない**(実測: ρ≈3.15 の発散 gene でも実軌道では tanh 飽和+方向ミスアラインで
感度が減衰)。= **状態ノルム監視も有限ホライズン摂動テストも ρ≥1 を見逃す** = まさに Phase 2 の
「STABLE 風経験 gate が発散 gene の 84% を false-admit」現象。不安定を見抜けるのは **box-sup の sound
certificate だけ**。よってデモは「経験 vs certificate の判別力差」を見せるのが唯一 honest な形。

2 つの artifact:
  1. **headline バーチャート SVG** (集団, data-backed): phase2_discriminative_results.json の per-method
     **発散 gene の false-admit 率** + **収縮 gene の過剰棄却率** を可視化。
     none 100% / stable_exp 84% / sound certs 0% 誤許可。cert_sdp は 0% 誤許可かつ 4.6% 過剰棄却のみ
     (=sound かつ最 navigable)。
  2. **単一 gene 感度トレース** (JSON, なぜ経験が騙されるかの supporting evidence): ρ≈3.15 の発散 gene
     でも実軌道の線形化摂動感度 ‖p_t‖ が減衰すること + その gene の **certificate box-sup σ_max > 1**
     (certificate のみが worst-case を見て reject) を併記。

成果物:
  - phase2_demo_gate_discrimination.json
  - phase2_demo_gate_discrimination.svg (ハンドビルド SMIL, matplotlib 非依存, 静止フレーム完成形)

honest 留保:
  - バーチャートは phase2_discriminative の集団実測 (n=400, 95 発散/305 収縮) の再可視化。
  - 単一 gene 感度は from-below 軌道測定 (絶対証明でない)。box-sup σ_max は cert_two 頂点最大。
  - これは可視化であり新規検定ではない。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verified_evolution_sdp_gate")))
import coupled_nd as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DISCRIM_JSON = os.path.join(HERE, "phase2_discriminative_results.json")
N = 6
T = 40
INPUT_SCALE = 0.15
SEED = 20260609

# 表示順と日本語ラベル
METHOD_ORDER = ["none", "stable_exp", "cert_inf", "cert_two", "cert_sdp"]
METHOD_LABEL = {
    "none": "無 gate (負の対照)",
    "stable_exp": "STABLE 風 経験 gate",
    "cert_inf": "cert_inf (sound)",
    "cert_two": "cert_two (sound)",
    "cert_sdp": "cert_sdp (sound)",
}


def _jac0(g):
    return C.jacobian(g, np.zeros(N), np.zeros(N))


def single_gene_sensitivity():
    """ρ≈3 の発散 gene の実軌道感度 ‖p_t‖(減衰=経験が騙される)+ certificate box-sup σ_max(>1=reject)。"""
    rng = np.random.default_rng(SEED)
    X = np.clip(rng.normal(0, INPUT_SCALE, (T, N)), -1, 1)
    gu = None
    for _ in range(40000):
        decay = rng.uniform(0, 1, N); W = rng.normal(0, 1, (N, N))
        du = np.clip(decay * 0.25, 0, 1); Wu = np.clip(W * rng.uniform(1.4, 2.2), -2, 2)
        g = C.CoupledNDGene.make(decay=du, W=Wu)
        if (not C.cert_inf(g)) and (not C.cert_two(g)):
            rho0 = float(np.max(np.abs(np.linalg.eigvals(_jac0(g)))))
            if rho0 > 1.2:
                gu = g; break
    if gu is None:
        return None
    # 実軌道の線形化摂動感度(worst-case 初期方向)
    _, _, Vt = np.linalg.svd(_jac0(gu)); p = Vt[0] / np.linalg.norm(Vt[0])
    s = np.zeros(N); sens = [1.0]
    for t in range(T):
        p = C.jacobian(gu, s, X[t]) @ p
        s = C.step(gu, s, X[t])
        sens.append(float(np.linalg.norm(p)))
    rho_emp = float(C.empirical_rho(gu, n_samples=4000, seed=SEED + 5))
    # certificate box-sup σ_max(cert_two 頂点最大)= certificate が見る worst-case
    tlo = C.t_min_per_coord(gu)
    box_sigma = max(float(np.linalg.svd(C._jac_at_t(gu, v), compute_uv=False)[0]) for v in C._box_vertices(tlo))
    return {
        "empirical_rho": rho_emp,
        "jac0_rho": float(np.max(np.abs(np.linalg.eigvals(_jac0(gu))))),
        "trajectory_sensitivity_final": float(sens[-1]),
        "certificate_box_sup_sigma_max": box_sigma,
        "cert_inf": bool(C.cert_inf(gu)), "cert_two": bool(C.cert_two(gu)),
        "interpretation": (f"発散 gene(ρ≈{rho_emp:.2f}≥1)でも実軌道の摂動感度は 1→{sens[-1]:.1e}(減衰)"
                           f"=経験テストは『安全』と誤認。だが certificate の box-sup σ_max={box_sigma:.2f}>1 ゆえ"
                           "sound cert は reject。状態/軌道は安全に見え、certificate のみが worst-case を見抜く。"),
    }


# --------------------------------------------------------------------------- #
# headline バーチャート SVG(matplotlib 非依存, SMIL, 静止フレーム完成形)
# --------------------------------------------------------------------------- #
def build_svg(per_method, n_div, n_con):
    W_, H_ = 760, 470
    x0, y0, pw, ph = 250, 80, 420, 300   # 棒領域
    bar_h = 22
    row_gap = ph / len(METHOD_ORDER)
    ORANGE, BLUE, GREY = "#ff9f43", "#4ea1ff", "#566075"
    rows = []
    for i, m in enumerate(METHOD_ORDER):
        pm = per_method[m]
        fa = pm["false_admit_rate"]        # 発散の false-admit 率(危険度)
        orj = pm["reject_on_contracting_rate"]  # 収縮の過剰棄却率(コスト)
        cy = y0 + i * row_gap + row_gap / 2
        sound = m.startswith("cert_")
        col = BLUE if sound else ORANGE
        # method ラベル
        rows.append(f'<text x="{x0-12}" y="{cy-2:.1f}" fill="#e8ecf4" font-size="13" text-anchor="end">{METHOD_LABEL[m]}</text>')
        # false-admit 棒(危険) — 上段
        w_fa = pw * fa
        rows.append(f'<rect x="{x0}" y="{cy-bar_h-2:.1f}" width="0" height="{bar_h-6}" rx="3" fill="{col}">'
                    f'<animate attributeName="width" from="0" to="{w_fa:.1f}" dur="1.4s" begin="{0.2+i*0.12:.2f}s" fill="freeze"/></rect>')
        rows.append(f'<text x="{x0+w_fa+8:.1f}" y="{cy-bar_h+9:.1f}" fill="#cfd6e4" font-size="11">{fa*100:.0f}% 発散を見逃し</text>')
        # over-reject 棒(コスト) — 下段(薄)
        w_or = pw * orj
        rows.append(f'<rect x="{x0}" y="{cy+2:.1f}" width="0" height="{bar_h-10}" rx="2" fill="{GREY}">'
                    f'<animate attributeName="width" from="0" to="{w_or:.1f}" dur="1.4s" begin="{0.4+i*0.12:.2f}s" fill="freeze"/></rect>')
        rows.append(f'<text x="{x0+max(w_or,2)+8:.1f}" y="{cy+14:.1f}" fill="#7a8294" font-size="10">{orj*100:.0f}% 収縮を過剰棄却</text>')
    rows = "\n  ".join(rows)
    # 100% 目盛
    ticks = []
    for frac in (0.0, 0.5, 1.0):
        tx = x0 + pw * frac
        ticks.append(f'<line x1="{tx:.1f}" y1="{y0-6}" x2="{tx:.1f}" y2="{y0+ph}" stroke="#2a2f3a" stroke-width="1"/>')
        ticks.append(f'<text x="{tx:.1f}" y="{y0+ph+18}" fill="#7a8294" font-size="11" text-anchor="middle">{frac*100:.0f}%</text>')
    ticks = "\n  ".join(ticks)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_}" font-family="Segoe UI,Helvetica,Arial,sans-serif">
  <rect width="{W_}" height="{H_}" fill="#0f1320"/>
  <text x="{W_/2}" y="34" fill="#e8ecf4" font-size="20" font-weight="700" text-anchor="middle">経験的 gate は発散を見逃す / sound certificate は見抜く</text>
  <text x="{W_/2}" y="55" fill="#9aa3b2" font-size="12" text-anchor="middle">spanning 集団 n={n_div+n_con} ({n_div} 発散 / {n_con} 収縮) の判別力 — 濃い棒=発散の見逃し率(危険)・薄い棒=収縮の過剰棄却率(コスト)</text>
  {ticks}
  {rows}
  <text x="{W_/2}" y="{H_-26}" fill="#ff9f43" font-size="12" text-anchor="middle">STABLE 風経験 gate は発散 gene の 84% を「安全」と誤許可 — tanh 有界で『忘れたように見える』が真 ρ≥1</text>
  <text x="{W_/2}" y="{H_-9}" fill="#4ea1ff" font-size="12" text-anchor="middle">cert_sdp: 発散 0% 誤許可 かつ 収縮 4.6% のみ過剰棄却 = sound かつ最 navigable</text>
</svg>'''


def main():
    if not os.path.exists(DISCRIM_JSON):
        print(f"ERROR: {DISCRIM_JSON} が無い (phase2_discriminative.py を先に実行)", flush=True)
        sys.exit(1)
    d = json.load(open(DISCRIM_JSON, encoding="utf-8"))
    per_method = d["spanning"]["per_method"]
    n_div = d["spanning"]["n_divergent"]; n_con = d["spanning"]["n_contracting"]

    sg = single_gene_sensitivity()

    results = {
        "meta": {"source_population": DISCRIM_JSON, "n_divergent": n_div, "n_contracting": n_con,
                 "honest_note": "norm/forgetting/single-trajectory-sensitivity all FAIL to reveal rho>=1 "
                                "(tanh saturation + direction misalignment); only the box-sup sound certificate "
                                "distinguishes. This demo visualizes that discriminative power."},
        "false_admit_rate": {m: per_method[m]["false_admit_rate"] for m in METHOD_ORDER},
        "over_reject_rate": {m: per_method[m]["reject_on_contracting_rate"] for m in METHOD_ORDER},
        "single_gene_evidence": sg,
        "verdict": ("経験的判別(無 gate 100% / STABLE 風 84% 発散を false-admit)vs sound cert(0% false-admit)。"
                    "cert_sdp は 0% 誤許可かつ 4.6% のみ過剰棄却=sound+最 navigable。"
                    "状態ノルム・有限忘却・単一軌道感度はいずれも ρ≥1 を見逃す → certificate の box-sup のみが見抜く。"),
    }
    out_json = os.path.join(HERE, "phase2_demo_gate_discrimination.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    svg = build_svg(per_method, n_div, n_con)
    out_svg = os.path.join(HERE, "phase2_demo_gate_discrimination.svg")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)

    print("=== gate 判別力 demo (data-backed) ===", flush=True)
    for m in METHOD_ORDER:
        pm = per_method[m]
        print(f"  {METHOD_LABEL[m]:22s} false-admit(発散)={pm['false_admit_rate']*100:5.1f}%  "
              f"over-reject(収縮)={pm['reject_on_contracting_rate']*100:5.1f}%", flush=True)
    if sg:
        print(f"\n単一 gene evidence: {sg['interpretation']}", flush=True)
    print(f"\nJSON: {out_json}\nSVG : {out_svg}", flush=True)


if __name__ == "__main__":
    main()
