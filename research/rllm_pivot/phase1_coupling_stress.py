# SPDX-License-Identifier: Apache-2.0
"""Phase 1.4: block 間 coupling soundness stress (Decision gate 1 (4) / North Star #2)。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) Phase 1 step 4 / H-coupling-soundness:
  2 block を residual で結合した最小系の **実 Jacobian の真 ρ** を独立 eigen で測り、
  per-block cert が admit した構成で合成 ρ≥1 が 1 件でも出れば FAIL。per-block AND を禁止し
  block 間 coupling 込み full-system cert を要求。

Phase −1 (phase_m1_coupling_scan) との差分:
  - Phase −1 は box-sample max を真 ρ の下界 proxy にした blind-spot。
  - 本 Phase 1.4 は **真 ρ = empirical_rho (eigenvalue from-below) で全 2n 系を直接測定**し、
    かつ **full-system cert に cert_sdp を追加** (clarabel 在環境で初測定; full cert_inf の
    過保守を SDP/2-norm がどれだけ救済するか)。Decision gate 1 (4) 判定。

構成:
  full W = [[W_A, γ·C_AB], [γ·C_BA, W_B]],  decay=[decay_A, decay_B],  V=I_{2n_b}
  - per-block cert  = cert_X(block A) AND cert_X(block B)  (計画が禁止する盲点)
  - full cert       = cert_X(合成 2n_b gene)               (sound だが 2^{2n_b})
  - 真 ρ            = empirical_rho(合成 gene)

honest:
  - empirical_rho は from-below。blind-spot (per-block admit ∧ 真ρ≥1) は **下限** (真の盲点は
    これ以上)。Phase −1 と同様、negative (盲点あり) を過小評価する方向ゆえ結論は保守的。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
import phase1_structural_surgery as S  # noqa: E402
import coupled_nd as C  # noqa: E402

SEED = 20260609
RHO_SAMPLES = 2500
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def sample_block(rng, nb, mia=1.0):
    """cert_inf PASS な block gene (decay, W) を返す。"""
    g = S.sample_admitted_base(rng, nb, mia=mia)
    return g  # CoupledNDGene or None


def compose(gA, gB, gamma, rng):
    """2 block を residual 結合した合成 CoupledNDGene (2nb)。coupling は単位スケール×γ。"""
    nb = gA.n
    n = 2 * nb
    decay = np.concatenate([gA.decay, gB.decay])
    W = np.zeros((n, n))
    W[:nb, :nb] = gA.W
    W[nb:, nb:] = gB.W
    CAB = rng.normal(0, 1, (nb, nb)) / np.sqrt(nb)
    CBA = rng.normal(0, 1, (nb, nb)) / np.sqrt(nb)
    W[:nb, nb:] = gamma * CAB
    W[nb:, :nb] = gamma * CBA
    return C.CoupledNDGene.make(decay=decay, W=W)


def run_gamma(rng, nb, gamma, n_pairs, mia=1.0):
    rows = []
    got, attempts = 0, 0
    while got < n_pairs and attempts < n_pairs * 12:
        attempts += 1
        gA = sample_block(rng, nb, mia)
        gB = sample_block(rng, nb, mia)
        if gA is None or gB is None:
            continue
        # per-block AND (各 block を独立に cert) — 計画が禁止する盲点
        pb_inf = bool(C.cert_inf(gA, mia) and C.cert_inf(gB, mia))
        pb_two = bool(C.cert_two(gA, mia) and C.cert_two(gB, mia))
        full = compose(gA, gB, gamma, rng)
        rho = float(C.empirical_rho(full, n_samples=RHO_SAMPLES, seed=int(rng.integers(0, 2**31))))
        rows.append({
            "pb_inf": pb_inf, "pb_two": pb_two,
            "full_inf": bool(C.cert_inf(full, mia)),
            "full_two": bool(C.cert_two(full, mia)),
            "full_sdp": bool(C.cert_sdp(full, mia)),
            "rho": rho,
        })
        got += 1

    rho = np.array([r["rho"] for r in rows])
    pb_inf = np.array([r["pb_inf"] for r in rows])
    full_inf = np.array([r["full_inf"] for r in rows])
    full_two = np.array([r["full_two"] for r in rows])
    full_sdp = np.array([r["full_sdp"] for r in rows])
    diverge = rho >= 1.0
    contracting = rho < 1.0
    n_pb = int(pb_inf.sum())
    # blind-spot: per-block AND admit ∧ 真 ρ≥1 (= per-block AND が不 sound な率)
    blind = int(np.sum(pb_inf & diverge))
    # full cert の soundness: full admit ∧ 真 ρ≥1 (0 が sound)
    fa_inf = int(np.sum(full_inf & diverge))
    fa_two = int(np.sum(full_two & diverge))
    fa_sdp = int(np.sum(full_sdp & diverge))
    # full cert の navigability: 真に収縮 (ρ<1) のうち admit する率
    nc = int(contracting.sum())
    cov_inf = float(full_inf[contracting].mean()) if nc else None
    cov_two = float(full_two[contracting].mean()) if nc else None
    cov_sdp = float(full_sdp[contracting].mean()) if nc else None
    return {
        "n_pairs": got, "rho_mean": float(rho.mean()),
        "pb_inf_admit": n_pb,
        "blind_spot_count": blind,
        "blind_spot_rate_of_pb_admit": (blind / n_pb) if n_pb else 0.0,
        "full_false_admit": {"inf": fa_inf, "two": fa_two, "sdp": fa_sdp},
        "full_coverage_on_contracting": {"inf": cov_inf, "two": cov_two, "sdp": cov_sdp},
    }


def main():
    rng = np.random.default_rng(SEED)
    results = {"meta": {"seed": SEED, "rho_samples": RHO_SAMPLES, "gammas": list(GAMMAS),
                        "note": "真ρ=empirical_rho + full cert_sdp 追加。Decision gate 1 (4)。"},
               "per_nb": {}}
    for nb in (2, 3):       # 合成 = 4, 6 (small-n per-component)
        print(f"=== nb={nb} (full={2*nb}) ===", flush=True)
        per_g = {}
        for g in GAMMAS:
            cell = run_gamma(rng, nb, g, n_pairs=80 if nb == 2 else 50)
            per_g[str(g)] = cell
            cov = cell["full_coverage_on_contracting"]
            print(f"  γ={g:.1f} pairs={cell['n_pairs']} ρ平均={cell['rho_mean']:.3f} | "
                  f"per-block盲点={cell['blind_spot_count']}/{cell['pb_inf_admit']} "
                  f"({cell['blind_spot_rate_of_pb_admit']:.3f}) | "
                  f"full false-admit inf/two/sdp={cell['full_false_admit']['inf']}/"
                  f"{cell['full_false_admit']['two']}/{cell['full_false_admit']['sdp']} | "
                  f"full coverage inf/two/sdp="
                  f"{cov['inf'] if cov['inf'] is None else round(cov['inf'],3)}/"
                  f"{cov['two'] if cov['two'] is None else round(cov['two'],3)}/"
                  f"{cov['sdp'] if cov['sdp'] is None else round(cov['sdp'],3)}", flush=True)
        results["per_nb"][str(nb)] = per_g

    out = os.path.join(os.path.dirname(__file__), "phase1_coupling_stress_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
