# SPDX-License-Identifier: Apache-2.0
"""Stage 3b kernel 多様化 — mechanism feasibility smoke (BG1/BG2/BG3/BG5).

本 smoke は DESIGN_kernel_diversification_3b.md の §2-§5 で主張した「4 kernel が Stage1 gate を
通せる (非空 admit) / Z3 verdict と閉形式上界が一致 / over-approx が sound / rwkv 後方互換」を
**小サンプルで実測**し、設計の mechanism feasibility を数値接地する。

honest 留保 (重要):
- これは feasibility smoke であって完全実証ではない。N は小さく (BG1/2/3 で各 kernel 400 gene)、
  Stage 3b の specialist (BG6-8) / 欺瞞地形 (BG9) は **本 smoke の対象外** (別実験)。
- 各 kernel の state-update は教科書 full 実装でなく **対角スカラ mock** (DESIGN §2 のスコープ宣言)。
- src/ は一切 import 改変しない。state_update / verifier から **読むだけ** で再利用する。
- z3 不在環境では state_norm/Lipschitz は skip (used_z3=False) し、その旨を結果に記録する。

実行: py -3.11 research/kernel_diversification/smoke_kernel_gates.py
出力: research/kernel_diversification/smoke_kernel_gates_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# src を import path に (DESIGN の「読むだけ再利用」: 既存 run_sequence / Z3 を呼ぶためのみ)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.state_update import StateUpdateGene, run_sequence  # noqa: E402
from llcore.verifier import is_z3_available  # noqa: E402

try:
    import z3
    _HAS_Z3 = True
except ImportError:  # pragma: no cover
    _HAS_Z3 = False


# ===========================================================================
# 各 kernel の対角スカラ写像 (DESIGN §2) — research 隔離実装。
# 契約: step(s, x, theta) -> s'   (s,x はスカラ float64 array, 各座標独立=対角)
#       jac_freevars(theta) -> (free_lo, free_hi, J_endpoints(theta, corners))
#         free_*  : Lipschitz 用 free 変数の hypercube bounds (over-approx)
#         J 閉形式: |J| sup は free hypercube 頂点で達成 (DESIGN §5)
# ===========================================================================


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z: np.ndarray) -> np.ndarray:
    # 数値安定 softplus
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


# --- rwkv (既存と同式) theta=(decay, mix, gate_str) ---
def rwkv_step(s, x, th):
    decay, mix, gate = th
    return decay * s + (1.0 - decay) * np.tanh(mix * x + gate * s)


def rwkv_L_ub(th):
    decay, mix, gate = th
    # J(t)=decay+(1-decay)*gate*t, t in [0,1]; sup|J| at t in {0,1}
    j0 = decay
    j1 = decay + (1.0 - decay) * gate
    return max(abs(j0), abs(j1))


# --- mamba_selective theta=(alpha, beta, gain) ---
def mamba_step(s, x, th):
    alpha, beta, gain = th
    a = _sigmoid(alpha * x + beta)
    return a * s + (1.0 - a) * (gain * x)


def mamba_L_ub(th):
    # J = a in (0,1), theta 非依存 -> sup|J| = 1 (端点)。real は <1 (開区間) だが
    # over-approx free a in [0,1] では端点 1 を含むので保守的に 1.0。
    return 1.0


# --- hopfield_dense theta=(eta, beta, xi) ---
def hopfield_step(s, x, th):
    eta, beta, xi = th
    z = beta * (xi * np.tanh(s) + x)
    return (1.0 - eta) * s + eta * np.tanh(z)


def hopfield_L_ub(th):
    eta, beta, xi = th
    # J = (1-eta) + eta*u*beta*xi*v, u,v in [0,1] double-overapprox。
    # sup|J| は u,v in {0,1}^2 の 4 隅。u=v=0 -> (1-eta); u=v=1 -> (1-eta)+eta*beta*xi
    cands = []
    for u in (0.0, 1.0):
        for v in (0.0, 1.0):
            cands.append((1.0 - eta) + eta * u * beta * xi * v)
    return max(abs(c) for c in cands)


# --- linear_attn theta=(w, lam, v_gain) ---
def linattn_step(s, x, th):
    w, lam, v_gain = th
    phi = _softplus(w * x)
    return lam * s + phi * (v_gain * x)


def linattn_L_ub(th):
    w, lam, v_gain = th
    # J = lam (phi は x のみ依存), theta 非依存に近い -> sup|J| = lam
    return abs(lam)


KERNELS = {
    "rwkv": dict(
        step=rwkv_step, L_ub=rwkv_L_ub,
        lo=np.array([0.0, -1.0, -2.0]), hi=np.array([1.0, 1.0, 2.0]),
    ),
    "mamba_selective": dict(
        step=mamba_step, L_ub=mamba_L_ub,
        lo=np.array([-2.0, -2.0, -1.0]), hi=np.array([2.0, 2.0, 1.0]),
    ),
    "hopfield_dense": dict(
        step=hopfield_step, L_ub=hopfield_L_ub,
        lo=np.array([0.0, 0.0, -1.0]), hi=np.array([1.0, 3.0, 1.0]),
    ),
    "linear_attn": dict(
        step=linattn_step, L_ub=linattn_L_ub,
        lo=np.array([-2.0, 0.0, -1.0]), hi=np.array([1.0, 1.0, 1.0]),  # lam<1 帯で有界化
    ),
}


# ===========================================================================
# BG1 — state_norm admit 率 (Z3, 対角 over-approx)。
#   実装方針: 各 kernel の更新を s' = c·s + d (c,d は free 変数 bounds で over-approx) に
#   線形化し、|s|<=1,|x|<=1 で |s'|<=1 反例を Z3 探索。kernel ごとに c,d の bounds を変える。
#   honest: これは DESIGN §2 の各 kernel state_norm 論証を Z3 で機械チェックする最小版。
# ===========================================================================

STATE_BOUND = 1.0
MAX_INPUT = 1.0


def state_norm_admit(kernel: str, th: np.ndarray) -> str:
    """単一 gene の state_norm gate verdict ("unsat"=admit / "sat"=reject / "unknown")."""
    if not _HAS_Z3:
        return "unknown"
    solver = z3.Solver()
    solver.set("timeout", 500)
    s = z3.Real("s")
    x = z3.Real("x")
    solver.add(s >= -STATE_BOUND, s <= STATE_BOUND, x >= -MAX_INPUT, x <= MAX_INPUT)

    if kernel == "rwkv":
        decay, mix, gate = (z3.RealVal(float(v)) for v in th)
        pre = mix * x + gate * s
        tanh_v = z3.Real("tv")
        solver.add(tanh_v * tanh_v <= pre * pre, tanh_v >= -1, tanh_v <= 1, tanh_v * pre >= 0)
        s_next = decay * s + (1 - decay) * tanh_v
    elif kernel == "mamba_selective":
        _, _, gain = th
        gain_q = z3.RealVal(float(gain))
        a = z3.Real("a")  # a=sigmoid(.) over-approx free in (0,1) -> [0,1]
        solver.add(a >= 0, a <= 1)
        s_next = a * s + (1 - a) * (gain_q * x)
    elif kernel == "hopfield_dense":
        eta, _, _ = th
        eta_q = z3.RealVal(float(eta))
        upd = z3.Real("upd")  # outer tanh in [-1,1]
        solver.add(upd >= -1, upd <= 1)
        s_next = (1 - eta_q) * s + eta_q * upd
    elif kernel == "linear_attn":
        w, lam, v_gain = th
        lam_q, vg_q = z3.RealVal(float(lam)), z3.RealVal(float(v_gain))
        # phi = softplus(w*x) >=0, 上界 softplus(|w|*max_input) を定数化 (DESIGN §2.4)
        phi_hi = float(_softplus(np.array([abs(float(w)) * MAX_INPUT]))[0])
        phi = z3.Real("phi")
        solver.add(phi >= 0, phi <= phi_hi)
        s_next = lam_q * s + phi * (vg_q * x)
    else:  # pragma: no cover
        raise ValueError(kernel)

    solver.add(z3.Or(s_next > STATE_BOUND, s_next < -STATE_BOUND))
    r = solver.check()
    return {z3.unsat: "unsat", z3.sat: "sat"}.get(r, "unknown")


# ===========================================================================
# BG2 — Lipschitz Z3 verdict と閉形式上界の一致 (state 方向)。
#   free 変数 over-approx で |J|>=1 反例探索。閉形式 L_ub と (unsat <=> L_ub<1) を照合。
# ===========================================================================


def lipschitz_verdict(kernel: str, th: np.ndarray) -> str:
    if not _HAS_Z3:
        return "unknown"
    solver = z3.Solver()
    solver.set("timeout", 1000 if kernel == "hopfield_dense" else 500)

    if kernel == "rwkv":
        decay, _, gate = (z3.RealVal(float(v)) for v in th)
        t = z3.Real("t")
        solver.add(t >= 0, t <= 1)
        j = decay + (1 - decay) * gate * t
    elif kernel == "mamba_selective":
        a = z3.Real("a")
        solver.add(a >= 0, a <= 1)
        j = a  # J = a, theta 非依存
    elif kernel == "hopfield_dense":
        eta, beta, xi = (z3.RealVal(float(v)) for v in th)
        u = z3.Real("u")
        v = z3.Real("v")
        solver.add(u >= 0, u <= 1, v >= 0, v <= 1)
        j = (1 - eta) + eta * u * beta * xi * v  # 双線形
    elif kernel == "linear_attn":
        _, lam, _ = th
        j = z3.RealVal(float(lam))  # J = lam
    else:  # pragma: no cover
        raise ValueError(kernel)

    solver.add(z3.Or(j >= 1, j <= -1))
    r = solver.check()
    return {z3.unsat: "unsat", z3.sat: "sat"}.get(r, "unknown")


def empirical_L(kernel: str, th: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    """中央差分で max|∂s'/∂s| (BG3 用)。"""
    step = KERNELS[kernel]["step"]
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1.0, 1.0, size=n)
    x = rng.uniform(-MAX_INPUT, MAX_INPUT, size=n)
    h = 1e-6
    d = (step(s + h, x, th) - step(s - h, x, th)) / (2 * h)
    return float(np.max(np.abs(d)))


# ===========================================================================
# 実行
# ===========================================================================


def run(n_gene: int = 400, seed: int = 20260601) -> dict:
    rng = np.random.default_rng(seed)
    results: dict = {
        "meta": {
            "z3_available": bool(is_z3_available()),
            "n_gene_per_kernel": n_gene,
            "state_bound": STATE_BOUND,
            "max_input": MAX_INPUT,
            "note": "feasibility smoke; diagonal scalar kernel mocks; NOT full impl",
        },
        "kernels": {},
        "BG5_rwkv_backcompat": {},
    }

    for name, spec in KERNELS.items():
        lo, hi = spec["lo"], spec["hi"]
        thetas = lo + (hi - lo) * rng.random((n_gene, 3))
        sn_unsat = sn_sat = sn_unknown = 0
        lip_match = lip_total = lip_timeout = 0
        bg3_violations = 0
        l_ub_min, l_ub_max = np.inf, -np.inf
        contraction_count = 0

        for th in thetas:
            # BG1
            sv = state_norm_admit(name, th)
            if sv == "unsat":
                sn_unsat += 1
            elif sv == "sat":
                sn_sat += 1
            else:
                sn_unknown += 1

            # BG2 + BG3
            l_ub = float(spec["L_ub"](th))
            l_ub_min, l_ub_max = min(l_ub_min, l_ub), max(l_ub_max, l_ub)
            if l_ub < 1.0:
                contraction_count += 1
            lv = lipschitz_verdict(name, th)
            if lv == "unknown":
                lip_timeout += 1
            else:
                lip_total += 1
                # 一致: (L_ub<1) <=> (unsat)
                agree = (l_ub < 1.0) == (lv == "unsat")
                if agree:
                    lip_match += 1
            emp = empirical_L(name, th, n=500)
            if emp > l_ub + 1e-3:
                bg3_violations += 1

        results["kernels"][name] = {
            "BG1_state_norm": {
                "admit_rate": sn_unsat / n_gene,
                "reject": sn_sat,
                "unknown": sn_unknown,
                "pass": (sn_unsat > 0) if is_z3_available() else None,
            },
            "BG2_lipschitz_match": {
                "match_rate": (lip_match / lip_total) if lip_total else None,
                "timeout": lip_timeout,
                "pass": (lip_match == lip_total and lip_total > 0) if is_z3_available() else None,
            },
            "BG3_overapprox": {
                "violations": bg3_violations,
                "pass": bg3_violations == 0,
            },
            "L_upper_bound_range": [l_ub_min, l_ub_max],
            "contraction_frac_closedform": contraction_count / n_gene,
        }

    # BG5 — rwkv 後方互換 (kernel_id=0 経路 == 既存 run_sequence)。
    # KernelGenome(kernel_id=0, theta=[decay,mix,gate, junk]) を decode して
    # 既存 StateUpdateGene + run_sequence と bit 一致するか。
    L, dim = 64, 8
    n_bc = 50
    all_match = True
    for i in range(n_bc):
        th = np.array([rng.random(), rng.uniform(-1, 1), rng.uniform(-2, 2)])
        junk = rng.random()  # max_dim=4 の余り 1 次元
        gene = StateUpdateGene(decay=float(th[0]), mix=float(th[1]), gate_str=float(th[2]))
        inputs = rng.uniform(-1, 1, size=(L, dim))
        ref = run_sequence(inputs, gene)
        # research decode 経路 (kernel_id=0 は theta[:3] を StateUpdateGene に, junk 無視)
        genome_theta = np.array([th[0], th[1], th[2], junk])
        decoded = StateUpdateGene.from_array(genome_theta[:3])
        got = run_sequence(inputs, decoded)
        if not np.array_equal(ref, got):
            all_match = False
            break
    results["BG5_rwkv_backcompat"] = {
        "n_tested": n_bc,
        "all_bit_match": all_match,
        "pass": all_match,
    }

    return results


if __name__ == "__main__":
    res = run()
    out = Path(__file__).resolve().parent / "smoke_kernel_gates_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"z3_available={res['meta']['z3_available']}")
    for k, v in res["kernels"].items():
        print(
            f"{k:18s} BG1_admit={v['BG1_state_norm']['admit_rate']:.3f} "
            f"BG2_match={v['BG2_lipschitz_match']['match_rate']} "
            f"BG3_viol={v['BG3_overapprox']['violations']} "
            f"L_ub={v['L_upper_bound_range'][0]:.3f}..{v['L_upper_bound_range'][1]:.3f} "
            f"contract_frac={v['contraction_frac_closedform']:.3f}"
        )
    print(f"BG5_backcompat pass={res['BG5_rwkv_backcompat']['pass']}")
    print(f"written: {out}")
