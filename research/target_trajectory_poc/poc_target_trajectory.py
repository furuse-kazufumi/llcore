# SPDX-License-Identifier: Apache-2.0
"""PoC — 「理想目標軌道への収束 (tracking)」を contraction ゲートで検査する数値実証.

目的 (design doc target_trajectory_verifier_2026_06_06.md の §4 に対応):
  既存 llcore 検証器は ρ(J)<1 (contraction, *どこかに* 収束) を Z3 / vertex-LMI で
  証明する。本 PoC は、それに加えて「**特定の参照軌道 s_ref[t] に追従する**」ことが
  contraction から *導出* できる (新たな proof obligation はほぼ不要) ことを、小次元
  (n=2,3) の CoupledNDGene で数値確認する。

検証する 3 命題 (すべて falsifiable):
  (P1) 実現可能性 (feasibility, 数値チェック): 参照軌道 s_ref が系
       s' = decay⊙s + (1-decay)⊙tanh(W s + V x_ref) の解である
       (各 t で残差 r[t] = s_ref[t+1] - step(s_ref[t], x_ref[t]) ≈ 0)。
       → これは「理想軌道がこの gene で実現可能」を保証する。証明ではなく数値同定。
  (P2) contraction (Z3 で証明済み): cert_inf/cert_two が True ⇒ sup-box 上 ρ(J)<1。
       これは既存ゲートそのもの。L_inf = infnorm_sup(J) < 1 を tube 定数に使う。
  (P3) tracking tube (理論導出 + 数値確認): 外乱 w (||w||_∞ ≤ w_bar) を入力に注入した
       実軌道 s_act は、参照軌道 s_ref から
            ||s_act[t] - s_ref[t]||_∞  ≤  w_bar * G / (1 - L)        (定常 tube)
       以内に閉じ込められる。ここで L = sup ||J_s||_∞ < 1 (状態方向 contraction),
       G = sup ||J_x||_∞ = sup ||(1-decay)⊙t ⊙ V||_∞ (入力方向ゲイン)。
       → これが「望ましい軌道に収束する」の sound な定式化の数値裏付け。

tube 導出 (sup-norm incremental / 離散 contraction, design doc §3 と一致):
  e[t] = s_act[t] - s_ref[t] と置く。両軌道は同じ gene の写像 F(s,x) に従い、
  s_act は x_ref + 外乱 d[t] (||d||_∞ ≤ w_bar) を入力、s_ref は x_ref を入力とする。
  F は s について L-Lipschitz (||J_s||_∞ ≤ L<1), x について G-Lipschitz (||J_x||_∞ ≤ G):
    ||e[t+1]||_∞ = ||F(s_act[t], x_ref[t]+d[t]) - F(s_ref[t], x_ref[t])||_∞
                 ≤ L||e[t]||_∞ + G||d[t]||_∞ ≤ L||e[t]||_∞ + G w_bar.
  幾何級数を解くと  limsup ||e[t]||_∞ ≤ G w_bar / (1 - L)  (定常 tube 半径)。
  L<1 が contraction ゲートの出力なので、tube 半径は「証明済み L + 数値 G + 設計 w_bar」
  から閉形式で出る。これが本 PoC の主張。

honest 留保:
  - L, G の Lipschitz 上界は achievable-t box [t_min,1]^n 上の sup を端点列挙 (sound)。
    Z3 / vertex 列挙と同じ box を使うので contraction ゲートと整合。
  - 参照軌道 feasibility (P1) は数値同定であって *証明ではない* (design doc の核心的限界)。
  - 「その参照軌道が *良い* 軌道か」は本枠組では検査不能 (タスク側の責任)。
"""
from __future__ import annotations

import os
import sys

import numpy as np

# --- 既存 n-dim verified substrate を再利用 (soundness-critical) --- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_SDP_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _SDP_GATE not in sys.path:
    sys.path.insert(0, _SDP_GATE)

from coupled_nd import (  # noqa: E402
    CoupledNDGene,
    cert_inf,
    cert_two,
    infnorm_sup,
    jacobian,
    step,
    t_min_per_coord,
)


def state_lipschitz_inf(g: CoupledNDGene, max_input_abs: float = 1.0) -> float:
    """L = sup_{achievable t} ||J_s||_∞ < 1 (状態方向 Lipschitz, 既存 infnorm_sup を流用).

    これは contraction ゲートが ρ(J)<1 の証拠に使う sup-box inf-norm そのもの
    (cert_inf の中身)。tube 定数 L として再利用する。"""
    return infnorm_sup(g, t_min_per_coord(g, max_input_abs))


def input_gain_inf(g: CoupledNDGene, max_input_abs: float = 1.0) -> float:
    """G = sup_{achievable t} ||J_x||_∞ = sup ||diag((1-decay)⊙t) V||_∞.

    入力方向ヤコビ ∂s'/∂x = diag((1-decay)⊙t) @ V。各 t_i は (0,1] (sup は t=1)。
    行 i の abs-sum = (1-decay_i)*1*Σ_j|V_ij| が t=1 で最大 (各成分 t に単調増加)。"""
    gc = g.clipped()
    n = gc.n
    rows = (1.0 - gc.decay) * np.abs(gc.V).sum(axis=1)  # t_i=1 で sup
    return float(np.max(rows))


def synthesize_reference(g: CoupledNDGene, x_ref: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """参照入力列 x_ref から「この gene で実現可能な」参照軌道 s_ref を順方向で合成.

    s_ref[0]=s0, s_ref[t+1]=step(g, s_ref[t], x_ref[t]) と置けば、構成的に
    残差 0 の feasible 軌道になる (= 理想軌道が *この系の解* であるケースを作る)。
    実運用では逆問題 (望ましい s_ref が与えられ x_ref を同定) になるが、ここでは
    feasibility と tube の数値関係を見るため前者で十分。"""
    T = x_ref.shape[0]
    n = g.n
    out = np.empty((T + 1, n), dtype=np.float64)
    out[0] = s0
    for t in range(T):
        out[t + 1] = step(g, out[t], x_ref[t])
    return out


def feasibility_residual(g: CoupledNDGene, s_ref: np.ndarray, x_ref: np.ndarray) -> float:
    """P1: 参照軌道が系の解か (max_t ||s_ref[t+1]-step(s_ref[t],x_ref[t])||_∞)."""
    T = x_ref.shape[0]
    mx = 0.0
    for t in range(T):
        r = s_ref[t + 1] - step(g, s_ref[t], x_ref[t])
        mx = max(mx, float(np.max(np.abs(r))))
    return mx


def rollout_with_disturbance(
    g: CoupledNDGene, x_ref: np.ndarray, s0: np.ndarray, w_bar: float, seed: int
) -> np.ndarray:
    """P3: 外乱 d[t] (||d||_∞ ≤ w_bar) を入力に注入した実軌道 s_act を回す.

    s_act[0]=s0 (= s_ref[0]; 初期一致), s_act[t+1]=step(g, s_act[t], x_ref[t]+d[t])。
    d は一様 [-w_bar, w_bar]^n (sup-norm worst をなぞる)。"""
    rng = np.random.default_rng(seed)
    T = x_ref.shape[0]
    n = g.n
    out = np.empty((T + 1, n), dtype=np.float64)
    out[0] = s0
    for t in range(T):
        d = rng.uniform(-w_bar, w_bar, size=n)
        out[t + 1] = step(g, out[t], x_ref[t] + d)
    return out


def run_case(name: str, g: CoupledNDGene, *, T: int = 200, w_bar: float = 0.05,
             n_dist_seeds: int = 64, max_input_abs: float = 1.0) -> dict:
    """1 gene について feasibility / contraction / tube を検査して結果 dict を返す."""
    g = g.clipped()
    n = g.n
    rng = np.random.default_rng(12345)

    # 参照入力列 (有界 |x_ref|_∞ <= max_input_abs。tanh-embedding の input bound と同じ流儀)。
    x_ref = rng.uniform(-max_input_abs, max_input_abs, size=(T, n))
    s0 = np.zeros(n)
    s_ref = synthesize_reference(g, x_ref, s0)

    # P1 feasibility
    resid = feasibility_residual(g, s_ref, x_ref)

    # P2 contraction (既存 Z3 / vertex ゲート)
    ok_inf = bool(cert_inf(g, max_input_abs))
    ok_two = bool(cert_two(g, max_input_abs))
    L = state_lipschitz_inf(g, max_input_abs)       # = sup ||J_s||_inf
    G = input_gain_inf(g, max_input_abs)            # = sup ||J_x||_inf

    # 理論 tube 半径 (L<1 のときのみ有限)
    tube = float(G * w_bar / (1.0 - L)) if L < 1.0 else float("inf")

    # P3 数値: 外乱注入で実軌道の追従誤差 (定常区間 = 後半) の sup を多シードで測る
    max_err_steady = 0.0
    max_err_all = 0.0
    settle = T // 2  # 過渡を捨てて定常 tube を見る
    for sd in range(n_dist_seeds):
        s_act = rollout_with_disturbance(g, x_ref, s0, w_bar, seed=1000 + sd)
        err = np.max(np.abs(s_act - s_ref), axis=1)  # 各 t の sup-norm 誤差
        max_err_all = max(max_err_all, float(np.max(err)))
        max_err_steady = max(max_err_steady, float(np.max(err[settle:])))

    inside = bool(max_err_steady <= tube + 1e-9)

    # 過渡 vs 定常の比 (contraction なら誤差は減衰; 非 contraction なら過渡誤差が定常へ
    # 残る/増幅する)。負例 D で「tube が無い = 追従誤差が tube に閉じ込められない」を
    # 構造的に示すための補助指標。L>=1 では理論 tube=inf なので、誤差が小さくても
    # *保証は無い* ことが要点 (tanh で状態自体は有界に留まるが、追従は保証されない)。
    ratio_steady_to_input = max_err_steady / max(w_bar, 1e-12)
    return {
        "name": name, "n": n, "T": T, "w_bar": w_bar,
        "feasibility_residual": resid,
        "cert_inf": ok_inf, "cert_two": ok_two,
        "L_state_lipschitz": L, "G_input_gain": G,
        "theoretical_tube_radius": tube,
        "empirical_max_err_steady": max_err_steady,
        "empirical_max_err_all": max_err_all,
        "err_amplification_vs_input": ratio_steady_to_input,
        "tube_holds": inside,
        "slack": (tube - max_err_steady) if np.isfinite(tube) else float("inf"),
    }


def main() -> int:
    cases: list[tuple[str, CoupledNDGene]] = []

    # Case A (n=2): 強 contraction (大 decay, 小 W) — inf ゲート PASS, tube タイト。
    cases.append(("A_strong_contraction_n2",
                  CoupledNDGene.make(decay=[0.9, 0.85], W=[[0.1, 0.05], [0.0, 0.12]])))

    # Case B (n=3): 中 contraction + 結合あり — inf PASS 想定、G が大きく tube 緩い。
    cases.append(("B_coupled_n3",
                  CoupledNDGene.make(decay=[0.7, 0.75, 0.6],
                                     W=[[0.2, 0.1, 0.0], [0.05, 0.15, 0.1], [0.0, 0.1, 0.2]])))

    # Case C (n=2): 境界付近 (弱 contraction) — L が 1 に近く tube 大、それでも閉じる。
    cases.append(("C_near_boundary_n2",
                  CoupledNDGene.make(decay=[0.4, 0.4], W=[[0.3, 0.2], [0.2, 0.3]])))

    # Case D (n=2): NON-contracting control (大 W, 小 decay) — ゲート REJECT, tube 定義不能。
    #   → 追従保証が出ないことを示す negative control (L>=1 で tube=inf)。
    cases.append(("D_noncontract_control_n2",
                  CoupledNDGene.make(decay=[0.1, 0.1], W=[[1.8, 1.5], [1.5, 1.8]])))

    results = []
    print(f"{'case':28s} {'cert_inf':8s} {'L':>7s} {'G':>7s} {'tube':>9s} "
          f"{'emp_err':>9s} {'holds':>6s} {'feas_res':>9s}")
    print("-" * 96)
    for name, g in cases:
        r = run_case(name, g)
        results.append(r)
        tube_s = f"{r['theoretical_tube_radius']:.4f}" if np.isfinite(r['theoretical_tube_radius']) else "inf"
        print(f"{r['name']:28s} {str(r['cert_inf']):8s} {r['L_state_lipschitz']:7.4f} "
              f"{r['G_input_gain']:7.4f} {tube_s:>9s} {r['empirical_max_err_steady']:9.5f} "
              f"{str(r['tube_holds']):>6s} {r['feasibility_residual']:9.2e}")

    print("\n--- interpretation ---")
    for r in results:
        if r["cert_inf"] and np.isfinite(r["theoretical_tube_radius"]):
            verdict = ("TUBE HOLDS: contraction-certified gene + feasible reference "
                       "=> tracking error inside theoretical tube"
                       if r["tube_holds"] else
                       "WARN: empirical error EXCEEDS tube (bug or unsound bound!)")
        else:
            verdict = ("NEGATIVE CONTROL: gate REJECTS (no contraction) "
                       "=> no tracking tube guaranteed (as expected)")
        print(f"  {r['name']:28s}: {verdict}")

    # save
    import json
    out_path = os.path.join(_HERE, "poc_target_trajectory_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {out_path}")

    # 健全性 assertion: contraction-certified な A/B/C で tube が必ず成立すること。
    certified = [r for r in cases_results(results) if r["cert_inf"]]
    all_hold = all(r["tube_holds"] for r in certified)
    print(f"\nASSERTION (all certified cases tube_holds): {all_hold}")
    return 0 if all_hold else 1


def cases_results(results):
    return results


if __name__ == "__main__":
    raise SystemExit(main())
