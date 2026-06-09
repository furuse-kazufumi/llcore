# SPDX-License-Identifier: Apache-2.0
"""Phase 1.3: per-row 成長 stress + 非自明性 AND (Decision gate 1 (3))。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) Phase 1 step 3 / North Star #1:
  実 width_grow (phase1_structural_surgery) で成長操作を N 回行い、
  (a) **0 false-admit を成長操作下で再確認** — 各 gate が admit した成長 gene が真 ρ<1 を保つ。
  (b) **PASS 条件に「非自明な進化価値を持つ admit ≥1」を AND** — 死んだ unit (Δfunc≈0) の
      自明 PASS を排除し、admit のうち少なくとも 1 つが Δfunc ≥ τ を満たすこと。

Phase −1 (phase_m1_coviability_scan) との差分:
  - Phase −1 は合成 ε-sweep + box-sup change を proxy にした「両立帯率」。
  - 本 Phase 1.3 は **真 ρ = empirical_rho(grown gene)** で soundness を直接検証し、
    Decision gate 1 (3) の「0 false-admit ∧ 非自明 admit ≥1」を gate ごとに判定する。
  - 証明器格子 (per_row/inf=cheap-sound-trivial, two/sdp=navigable-small_n) を成長手術
    レベルで対比 → 計画 §⑩「per-component gate を cert_two/sdp に格上げ・small-n 限定」の実証。

honest:
  - empirical_rho は from-below 推定 (真 box-sup ρ の下界)。false-admit = admit ∧ 推定ρ≥1 は
    soundness 反証の探索 (証明器は数学的に sound; 本 check はその consistency 監査)。
  - net2net は incoming-copy 近似 (exact function-preserving でない; module docstring 留保)。
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
TAU = 0.05            # 非自明な進化価値の閾 (既存出力相対 L2 変化)
RHO_SAMPLES = 1500    # empirical_rho の sample 数 (near-boundary 感度)
EPS_GRID = np.linspace(0.0, 3.0, 16)

GATES = ("per_row", "cert_inf", "cert_two", "cert_sdp")


def gate_admits(name, grown, original_n, mia=1.0):
    if name == "per_row":
        return S.per_row_growth_ok(grown, original_n)
    if name == "cert_inf":
        return S.cert_inf_ok(grown, mia)
    if name == "cert_two":
        return S.cert_two_ok(grown, mia)
    if name == "cert_sdp":
        return S.cert_sdp_ok(grown, mia)
    raise ValueError(name)


def run_cell(rng, n, headroom_steps, n_bases, n_dirs, Xs):
    """1 (n, headroom) セル: 全成長 gene を集め gate ごとに soundness/非自明性を集計。"""
    records = []  # 各成長 gene の (admits dict, rho, dfunc)
    base_rhos = []
    got, attempts = 0, 0
    while got < n_bases and attempts < n_bases * 12:
        attempts += 1
        g = S.sample_admitted_base(rng, n, headroom_steps=headroom_steps)
        if g is None:
            continue
        base_rhos.append(float(C.empirical_rho(g, n_samples=RHO_SAMPLES, seed=int(rng.integers(0, 2**31)))))
        for _ in range(n_dirs):
            d = S.random_directions(rng, g)
            for mode in ("fresh", "net2net"):
                for eps in EPS_GRID:
                    grown = S.width_grow(g, eps=float(eps), in_dir=d.in_dir, out_dir=d.out_dir,
                                         self_w=d.self_w, new_decay=d.new_decay, mode=mode, k=d.k)
                    rho = float(C.empirical_rho(grown, n_samples=RHO_SAMPLES, seed=int(rng.integers(0, 2**31))))
                    dfunc = S.function_change(g, grown, Xs)
                    admits = {gt: bool(gate_admits(gt, grown, n)) for gt in GATES}
                    records.append((admits, rho, dfunc))
        got += 1

    rho = np.array([r[1] for r in records])
    dfunc = np.array([r[2] for r in records])
    n_rec = len(records)
    per_gate = {}
    for gt in GATES:
        adm = np.array([r[0][gt] for r in records])
        n_adm = int(adm.sum())
        false_admit = int(np.sum(adm & (rho >= 1.0)))                  # soundness 反証 (0 が PASS)
        nontrivial = int(np.sum(adm & (dfunc >= TAU)))                 # 非自明な admit 数
        nt_sound = int(np.sum(adm & (dfunc >= TAU) & (rho < 1.0)))     # 非自明かつ sound な admit
        per_gate[gt] = {
            "n_admit": n_adm,
            "admit_rate": n_adm / n_rec if n_rec else 0.0,
            "false_admit": false_admit,
            "n_nontrivial_admit": nontrivial,
            "n_nontrivial_sound_admit": nt_sound,
            "max_dfunc_among_sound_admit": (
                float(dfunc[(adm) & (rho < 1.0)].max()) if np.any((adm) & (rho < 1.0)) else 0.0),
            # Decision gate 1 (3): 0 false-admit ∧ 非自明 sound admit ≥1
            "gate3_pass": bool(false_admit == 0 and nt_sound >= 1),
        }
    return {
        "n_records": n_rec, "n_bases": got,
        "base_rho_median": float(np.median(base_rhos)) if base_rhos else None,
        "rho_max_overall": float(rho.max()) if n_rec else None,
        "per_gate": per_gate,
    }


def main():
    rng = np.random.default_rng(SEED)
    L = 48
    results = {"meta": {"seed": SEED, "tau": TAU, "rho_samples": RHO_SAMPLES,
                        "eps_grid": [float(e) for e in EPS_GRID],
                        "note": "実 width_grow + empirical_rho 真ρ。Decision gate 1 (3) 判定。"},
               "cells": {}}
    for n in (4, 6):
        Xs = [rng.normal(size=(L, n)) for _ in range(3)]
        for headroom_steps in (0, 2):  # edge (最保守) と headroom (成長余地あり)
            key = f"n{n}_hr{headroom_steps}"
            print(f"=== {key} ===", flush=True)
            cell = run_cell(rng, n, headroom_steps, n_bases=24, n_dirs=2, Xs=Xs)
            results["cells"][key] = cell
            print(f"  records={cell['n_records']} base_ρ中央={cell['base_rho_median']:.3f} "
                  f"成長ρ最大={cell['rho_max_overall']:.3f}", flush=True)
            for gt in GATES:
                m = cell["per_gate"][gt]
                print(f"  [{gt:8s}] admit率={m['admit_rate']:.3f} false-admit={m['false_admit']} "
                      f"非自明admit={m['n_nontrivial_admit']} 非自明∧sound={m['n_nontrivial_sound_admit']} "
                      f"maxΔfunc={m['max_dfunc_among_sound_admit']:.4f} → gate3_PASS={m['gate3_pass']}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase1_growth_stress_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
