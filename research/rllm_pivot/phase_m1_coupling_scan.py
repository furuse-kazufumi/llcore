# SPDX-License-Identifier: Apache-2.0
"""Phase -1 (part 2): block 間 coupling soundness scan (F6 第二存立条件, $0/CPU)。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) の North Star #2 / red-team F6 を実データで決着する。

問い:
  計画 §5.2 の中核設計「block を小 n に切り、各 block 独立 cert_inf を AND」は、block 間 coupling
  (residual/attention 経由) を未 certify のまま残す。**per-block AND が admit した構成で、block 間
  coupling を含む合成系の真の spectral radius ρ が 1 を越える (=実際は発散する) ことがどれだけ起きるか。**

設計:
  2 block の coupled adapter:
    s1' = decay1⊙s1 + (1-decay1)⊙tanh(W1 s1 + γ·C12 s2 + V1 x)
    s2' = decay2⊙s2 + (1-decay2)⊙tanh(W2 s2 + γ·C21 s1 + V2 x)
  合成 Jacobian (2n×2n):
    [[diag(decay1)+diag((1-decay1)t1)W1,  diag((1-decay1)t1)·γC12],
     [diag((1-decay2)t2)·γC21,            diag(decay2)+diag((1-decay2)t2)W2]]
  per-block cert_inf は対角 block (W1,W2) のみ検査し off-diagonal coupling (γC) を無視する。

測るもの (γ=coupling 強度を sweep):
  - per_block_admits: cert_inf(block1)<1 ∧ cert_inf(block2)<1 (現計画の gate)
  - full_cert_inf: 合成 2n 系の cert_inf (coupling 込みの sound 上界)
  - full_true_rho: 合成 Jacobian の真の ρ を achievable-t box 上 sample 最大 (経験的 truth)
  - **coupling_blind_spot: per_block_admits ∧ full_true_rho≥1** (危険: per-block 安全と言うが実際は発散)

honest:
  full_true_rho は box 上 sample 最大で sup の下界 (真の sup は ≥ これ) ゆえ blind-spot は過小評価寄り
  = 報告される blind-spot 率は下限。per-block AND の危険性は「これ以上」。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llcore.verifier.backends import _infnorm_sup, _t_min  # noqa: E402


def cert_inf_sup(decay, W, V, max_input_abs=1.0):
    return _infnorm_sup(decay, W, _t_min(decay, W, V, max_input_abs))


def sample_admitted(rng, n, max_input_abs=1.0, max_tries=60):
    """W を admit するまで縮小して cert_inf PASS な (decay, W) を返す (任意 n で確実)。"""
    V = np.eye(n)
    for _ in range(max_tries):
        decay = rng.uniform(0.0, 1.0, size=n)
        W = (rng.normal(0.0, 1.0, size=(n, n)) / np.sqrt(n)) * float(rng.uniform(0.2, 0.9))
        for _ in range(50):
            if cert_inf_sup(decay, W, V, max_input_abs) < 1.0:
                return decay, W
            W = W * 0.85
    return None


def full_jac_at(decay_f, Wf, t):
    """合成系 Jacobian J(t) = diag(decay) + diag((1-decay)*t) @ Wf。"""
    return np.diag(decay_f) + np.diag((1.0 - decay_f) * t) @ Wf


def true_rho_over_box(decay_f, Wf, V_f, rng, max_input_abs=1.0, n_samp=200):
    """合成 Jacobian の真の ρ を achievable-t box 上 sample 最大 (sup の下界)。"""
    t_lo = _t_min(decay_f, Wf, V_f, max_input_abs)
    N = decay_f.shape[0]
    best = 0.0
    # 端点 + ランダム内点の混合 sample
    for _ in range(n_samp):
        t = t_lo + (1.0 - t_lo) * rng.random(N)
        rho = float(np.max(np.abs(np.linalg.eigvals(full_jac_at(decay_f, Wf, t)))))
        best = max(best, rho)
    # ti=1 全部 (支配的) も明示評価
    rho1 = float(np.max(np.abs(np.linalg.eigvals(full_jac_at(decay_f, Wf, np.ones(N))))))
    return max(best, rho1)


def build_full(decay1, W1, decay2, W2, C12, C21, gamma):
    n = decay1.shape[0]
    decay_f = np.concatenate([decay1, decay2])
    Wf = np.zeros((2 * n, 2 * n))
    Wf[:n, :n] = W1
    Wf[n:, n:] = W2
    Wf[:n, n:] = gamma * C12
    Wf[n:, :n] = gamma * C21
    V_f = np.eye(2 * n)
    return decay_f, Wf, V_f


def main():
    rng = np.random.default_rng(20260609)
    results = {"meta": {"seed": 20260609, "note": "full_true_rho は box sample 最大 = sup の下界 (blind-spot は下限)"},
               "per_n": {}}

    for n in (4, 8):
        n_pairs, n_dirs = 50, 3
        gammas = np.linspace(0.0, 2.0, 21)
        rows = []  # (gamma, per_block_admits, full_cert_admits, full_true_rho)
        got = 0
        while got < n_pairs:
            b1 = sample_admitted(rng, n)
            b2 = sample_admitted(rng, n)
            if b1 is None or b2 is None:
                continue
            decay1, W1 = b1
            decay2, W2 = b2
            pb = (cert_inf_sup(decay1, W1, np.eye(n)) < 1.0) and (cert_inf_sup(decay2, W2, np.eye(n)) < 1.0)
            for _ in range(n_dirs):
                C12 = rng.normal(size=(n, n)) / np.sqrt(n)
                C21 = rng.normal(size=(n, n)) / np.sqrt(n)
                for g in gammas:
                    decay_f, Wf, V_f = build_full(decay1, W1, decay2, W2, C12, C21, g)
                    full_cert = cert_inf_sup(decay_f, Wf, V_f) < 1.0
                    rho = true_rho_over_box(decay_f, Wf, V_f, rng, n_samp=120)
                    rows.append((float(g), bool(pb), bool(full_cert), float(rho)))
            got += 1

        arr_g = np.array([r[0] for r in rows])
        arr_pb = np.array([r[1] for r in rows])
        arr_fc = np.array([r[2] for r in rows])
        arr_rho = np.array([r[3] for r in rows])

        # coupling blind-spot: per-block admit だが合成真 ρ≥1
        blind = arr_pb & (arr_rho >= 1.0)
        # full cert_inf が捕まえる率 (per-block admit のうち full_cert も admit する割合 = soundness 保てる構成)
        per_block_set = arr_pb
        blind_rate_overall = float(blind.sum() / max(1, per_block_set.sum()))

        # γ ごとの blind-spot 率
        by_gamma = []
        for g in gammas:
            m = (arr_g == g) & arr_pb
            if m.sum() == 0:
                continue
            bs = (arr_g == g) & arr_pb & (arr_rho >= 1.0)
            fc = (arr_g == g) & arr_pb & arr_fc
            by_gamma.append({
                "gamma": float(g),
                "blind_spot_rate": float(bs.sum() / m.sum()),
                "full_cert_admit_rate": float(fc.sum() / m.sum()),
                "mean_true_rho": float(arr_rho[m].mean()),
            })

        results["per_n"][str(n)] = {
            "n_rows": len(rows),
            "blind_spot_rate_overall": blind_rate_overall,
            "by_gamma": by_gamma,
        }
        print(f"[n={n}] per-block AND の coupling blind-spot 率 (全γ平均)={blind_rate_overall:.3f}")
        for bg in by_gamma:
            if bg["gamma"] in (0.0, 0.5, 1.0, 1.5, 2.0):
                print(f"    γ={bg['gamma']:.1f}: blind-spot={bg['blind_spot_rate']:.3f}  "
                      f"full_cert救済={bg['full_cert_admit_rate']:.3f}  真ρ平均={bg['mean_true_rho']:.3f}")

    out = os.path.join(os.path.dirname(__file__), "phase_m1_coupling_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}")
    return results


if __name__ == "__main__":
    main()
