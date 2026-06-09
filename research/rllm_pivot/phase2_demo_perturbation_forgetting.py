# SPDX-License-Identifier: Apache-2.0
"""Phase 2 demo (F11, 動きで魅せる): verified gate あり/なしの摂動忘却を可視化。

EVOLVABLE_LLM_PLAN_2026_06_09.md §⑫.2 + PHASE_2_VERDICT.md §5 の「動きで魅せるデモ」。
**honest 物理**: coupled 基質 `s' = decay⊙s + (1-decay)⊙tanh(Ws+x)` は tanh で状態が**常時有界**
ゆえ、不安定性は「出力ノルムの∞発散」ではなく **摂動忘却の失敗(echo-state property の崩壊)** =
近接 2 軌道が収束しない、として現れる。これは Phase 2 の核心知見そのもの:
  「STABLE 風経験 gate は発散 gene の 84% を false-admit — tanh 有界ゆえ有限ホライズンでは
   『忘却したように見える』が真 ρ≥1 = sound certificate でないと見抜けない」(PHASE_2_VERDICT §1)。

∴ 計画の「ρ→1.95 出力ノルム発散」フレーズは tanh 基質では物理的に起きない(=честно修正)。
正しい「動き」= **gate 付き(ρ<1, cert admit)は摂動を指数的に忘れる / gate なし(ρ≥1, cert reject)は
摂動が消えない・増幅する**、を近接 2 軌道の乖離 ‖s_t − s'_t‖ の時系列で見せる。

成果物:
  - phase2_demo_perturbation_forgetting_results.json — 2 gene の (decay,W) / 各 cert / empirical_rho /
    摂動忘却距離の時系列 / 状態ノルム時系列。
  - phase2_demo_perturbation_forgetting.svg — ハンドビルド SMIL アニメ SVG(静止フレーム完成形 +
    progressive draw)。matplotlib 非依存(FullSense 宣言的 SVG 方針 / optional-extras 維持)。

honest 留保:
  - empirical_rho は from-below sampling(0 観測の絶対証明でない consistency oracle)。
  - 入力は |x|≤1 の固定 bounded 列(cert の max_input_abs=1.0 前提と整合)。
  - これは demo(可視化)であり統計的検定ではない。gene は「明確に対照的な 2 例」を seed 探索で選定。
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

N = 6
T = 40           # 軌道長(摂動忘却を見せるのに十分)
EPS_PERT = 1e-2  # 初期摂動の大きさ
SEED = 20260609


def adapter_run_trace(g: C.CoupledNDGene, X, s0):
    """s0 から X を入力に T step 回し、各 step の状態列 (T+1, n) を返す。"""
    s = s0.copy()
    traj = [s.copy()]
    for t in range(X.shape[0]):
        s = C.step(g, s, X[t])
        traj.append(s.copy())
    return np.stack(traj)


def perturbation_forgetting_trace(g, X, rng):
    """近接 2 軌道(s0=0, s0=δ)の乖離 ‖s_t − s'_t‖ 時系列と状態ノルム ‖s_t‖ 時系列を返す。"""
    d = rng.normal(size=N); d = EPS_PERT * d / (np.linalg.norm(d) + 1e-12)
    A = adapter_run_trace(g, X, np.zeros(N))
    B = adapter_run_trace(g, X, d)
    dist = np.linalg.norm(A - B, axis=1)       # (T+1,) 摂動乖離
    snorm = np.linalg.norm(A, axis=1)          # (T+1,) 状態ノルム(tanh 有界)
    return dist, snorm


def find_genes(rng):
    """gate-admit な収縮 gene(cert_inf True, ρ<1)と no-gate な発散 gene(cert reject, ρ≥1)を探索。"""
    stable = None
    unstable = None
    for _ in range(20000):
        decay = rng.uniform(0.0, 1.0, N)
        W = rng.normal(0, 1.0, (N, N))
        # 収縮候補: decay 高め + W 小さめ
        if stable is None:
            ds = np.clip(decay * 0.5 + 0.5, 0, 1)
            Ws = W * rng.uniform(0.1, 0.4)
            gs = C.CoupledNDGene.make(decay=ds, W=Ws)
            if C.cert_inf(gs):
                rho = C.empirical_rho(gs, n_samples=2000, seed=SEED + 1)
                if rho < 0.95:
                    stable = (gs, float(rho))
        # 発散候補: decay 低め + W 大きめ → cert reject かつ ρ≥1
        if unstable is None:
            du = np.clip(decay * 0.3, 0, 1)
            Wu = W * rng.uniform(1.3, 2.0)
            Wu = np.clip(Wu, -2, 2)
            gu = C.CoupledNDGene.make(decay=du, W=Wu)
            if not C.cert_inf(gu) and not C.cert_two(gu):
                rho = C.empirical_rho(gu, n_samples=2000, seed=SEED + 2)
                if rho > 1.25:
                    unstable = (gu, float(rho))
        if stable and unstable:
            break
    return stable, unstable


# --------------------------------------------------------------------------- #
# ハンドビルド SMIL アニメ SVG(matplotlib 非依存, 静止フレーム完成形 + progressive draw)
# --------------------------------------------------------------------------- #
def _poly_points(series, x0, y0, w, h, ymax):
    n = len(series)
    pts = []
    for i, v in enumerate(series):
        x = x0 + w * (i / (n - 1))
        y = y0 + h * (1.0 - min(v, ymax) / ymax)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def build_svg(dist_stable, dist_unstable, rho_stable, rho_unstable):
    W_, H_ = 720, 420
    x0, y0, pw, ph = 70, 60, 580, 280
    ymax = float(max(dist_unstable.max(), dist_stable.max(), EPS_PERT) * 1.15)
    pts_s = _poly_points(dist_stable, x0, y0, pw, ph, ymax)
    pts_u = _poly_points(dist_unstable, x0, y0, pw, ph, ymax)
    # y 軸グリッド
    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = y0 + ph * (1.0 - frac)
        gv = ymax * frac
        grid.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0+pw}" y2="{gy:.1f}" stroke="#2a2f3a" stroke-width="1"/>')
        grid.append(f'<text x="{x0-8}" y="{gy+4:.1f}" fill="#7a8294" font-size="11" text-anchor="end">{gv:.2f}</text>')
    grid = "\n  ".join(grid)
    # progressive draw: 全長を dash で隠し dashoffset を 0 へアニメ(静止時は全描画=フレーム完成形)
    dash = pw * 1.6
    BLUE, ORANGE = "#4ea1ff", "#ff9f43"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_}" font-family="Segoe UI,Helvetica,Arial,sans-serif">
  <rect width="{W_}" height="{H_}" fill="#0f1320"/>
  <text x="{W_/2}" y="32" fill="#e8ecf4" font-size="19" font-weight="700" text-anchor="middle">Verified gate: 摂動を忘れるか (echo-state property)</text>
  <text x="{W_/2}" y="50" fill="#9aa3b2" font-size="12" text-anchor="middle">近接 2 軌道の乖離 ‖s_t − s'_t‖ — tanh 有界ゆえ「ノルム発散」でなく「摂動忘却の失敗」として不安定が現れる</text>
  {grid}
  <line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#3a4150" stroke-width="1.5"/>
  <text x="{x0+pw}" y="{y0+ph+20}" fill="#7a8294" font-size="11" text-anchor="end">time step →</text>
  <text x="{x0-44}" y="{y0+ph/2}" fill="#7a8294" font-size="11" text-anchor="middle" transform="rotate(-90 {x0-44} {y0+ph/2})">‖s_t − s'_t‖</text>
  <polyline points="{pts_u}" fill="none" stroke="{ORANGE}" stroke-width="3" stroke-linejoin="round"
    stroke-dasharray="{dash}" stroke-dashoffset="{dash}">
    <animate attributeName="stroke-dashoffset" from="{dash}" to="0" dur="2.4s" begin="0.3s" fill="freeze"/>
  </polyline>
  <polyline points="{pts_s}" fill="none" stroke="{BLUE}" stroke-width="3" stroke-linejoin="round"
    stroke-dasharray="{dash}" stroke-dashoffset="{dash}">
    <animate attributeName="stroke-dashoffset" from="{dash}" to="0" dur="2.4s" begin="0.3s" fill="freeze"/>
  </polyline>
  <g font-size="13">
    <rect x="{x0+pw-260}" y="{y0+6}" width="250" height="58" rx="6" fill="#171c2b" stroke="#2a2f3a"/>
    <line x1="{x0+pw-248}" y1="{y0+26}" x2="{x0+pw-218}" y2="{y0+26}" stroke="{ORANGE}" stroke-width="3"/>
    <text x="{x0+pw-210}" y="{y0+30}" fill="#e8ecf4">no-gate (cert reject): ρ≈{rho_unstable:.2f} ≥ 1 → 忘れない</text>
    <line x1="{x0+pw-248}" y1="{y0+48}" x2="{x0+pw-218}" y2="{y0+48}" stroke="{BLUE}" stroke-width="3"/>
    <text x="{x0+pw-210}" y="{y0+52}" fill="#e8ecf4">verified gate (admit): ρ≈{rho_stable:.2f} &lt; 1 → 指数的に忘れる</text>
  </g>
  <text x="{W_/2}" y="{H_-18}" fill="#9aa3b2" font-size="12" text-anchor="middle">sound contraction certificate (ρ&lt;1) を満たす構造だけ admit = online 適応しても発散・破滅的忘却しない homeostatic gate</text>
</svg>'''


def main():
    rng = np.random.default_rng(SEED)
    # |x|≤1 の固定 bounded 入力列(cert max_input_abs=1.0 と整合)
    X = np.clip(rng.normal(0, 0.5, (T, N)), -1, 1)

    stable, unstable = find_genes(rng)
    if stable is None or unstable is None:
        print(f"ERROR: gene 探索失敗 stable={stable is not None} unstable={unstable is not None}", flush=True)
        sys.exit(1)
    gs, rho_s = stable
    gu, rho_u = unstable

    dist_s, snorm_s = perturbation_forgetting_trace(gs, X, np.random.default_rng(SEED + 10))
    dist_u, snorm_u = perturbation_forgetting_trace(gu, X, np.random.default_rng(SEED + 10))

    results = {
        "meta": {"n": N, "T": T, "eps_pert": EPS_PERT, "seed": SEED,
                 "substrate": "CoupledNDGene s'=decay⊙s+(1-decay)⊙tanh(Ws+x) (tanh 有界=ノルム発散せず)",
                 "honest_note": "instability manifests as failure-to-forget-perturbations (echo-state failure), "
                                "NOT norm divergence; matches PHASE_2_VERDICT STABLE-gate false-admit insight"},
        "stable_gene": {"cert_inf": bool(C.cert_inf(gs)), "cert_two": bool(C.cert_two(gs)),
                        "empirical_rho": rho_s, "decay": gs.decay.tolist(), "W": gs.W.tolist()},
        "unstable_gene": {"cert_inf": bool(C.cert_inf(gu)), "cert_two": bool(C.cert_two(gu)),
                          "empirical_rho": rho_u, "decay": gu.decay.tolist(), "W": gu.W.tolist()},
        "forget_distance_stable": dist_s.tolist(),
        "forget_distance_unstable": dist_u.tolist(),
        "state_norm_stable": snorm_s.tolist(),
        "state_norm_unstable": snorm_u.tolist(),
        "verdict": (f"verified gate admit (ρ≈{rho_s:.2f}<1): 摂動乖離 {dist_s[0]:.3f}→{dist_s[-1]:.4f} (指数的に忘却) / "
                    f"no-gate reject (ρ≈{rho_u:.2f}≥1): {dist_u[0]:.3f}→{dist_u[-1]:.3f} (忘れない/増幅)。"
                    f"状態ノルムは両者とも tanh 有界(stable max {snorm_s.max():.2f} / unstable max {snorm_u.max():.2f})"
                    "=不安定は echo-state 失敗として現れる(ノルム発散でない)。"),
    }
    out_json = os.path.join(os.path.dirname(__file__), "phase2_demo_perturbation_forgetting_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    svg = build_svg(dist_s, dist_u, rho_s, rho_u)
    out_svg = os.path.join(os.path.dirname(__file__), "phase2_demo_perturbation_forgetting.svg")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)

    print("=== verified gate 摂動忘却 demo ===", flush=True)
    print(f"stable gene  : cert_inf={results['stable_gene']['cert_inf']} ρ≈{rho_s:.3f}  "
          f"摂動乖離 {dist_s[0]:.3f}→{dist_s[-1]:.4f} (忘却)", flush=True)
    print(f"unstable gene: cert_inf={results['unstable_gene']['cert_inf']} ρ≈{rho_u:.3f}  "
          f"摂動乖離 {dist_u[0]:.3f}→{dist_u[-1]:.3f} (持続/増幅)", flush=True)
    print(f"状態ノルム max: stable {snorm_s.max():.2f} / unstable {snorm_u.max():.2f} (両者 tanh 有界)", flush=True)
    print(f"\nJSON: {out_json}\nSVG : {out_svg}", flush=True)


if __name__ == "__main__":
    main()
