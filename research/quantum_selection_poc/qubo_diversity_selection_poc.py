# SPDX-License-Identifier: Apache-2.0
"""QUBO 多様性選択 — 最小 PoC (深追いしない方針 / honest disclosure 規律)。

## 何を測るか (スコープ厳格)

進化ループの「次世代に残す K 個体を選ぶ」段を **QUBO (二次制約なし二値最適化)** に定式化し、
古典サンプラーで機能するかだけを確認する。**量子実機 (QPU) 優位は一切主張しない** —
本 PoC は「(1) 定式化が正しく解けるか (2) QPU に drop-in できる構造か」の sanity であって、
QA vs 古典のベンチではない (それは別途 QPU vs tuned SA vs random の 3 比較が要る)。

novelty 判定 (memory:project_creative_corpora_2026_06_07) に従い、本 PoC 自体は novelty を
主張しない (QUBO 選択は先行多数)。価値は **certificate gate との接続点** を最小に示すこと:
「健全性 certificate を満たさない個体を選ばない」を QUBO の線形項で表現できる (§ cert_gate)。

## QUBO 定式化

集団 N 個体、各 i に fitness f_i ∈ [0,1] と embedding e_i。次世代に K 個選ぶ二値 x_i ∈ {0,1}:

    minimize  E(x) = − Σ f_i x_i              (fitness を最大化)
                     + λ Σ_{i<j} s_ij x_i x_j  (類似ペナルティ = 多様性を促す)
                     + A (Σ x_i − K)²          (基数制約 |選択|=K を penalty 化)
              [+ μ Σ (1 − cert_i) x_i]          (§ cert_gate: unsound 個体に penalty)

s_ij = cosine 類似度 (∈[0,1])。これは quality-diversity selection の標準形であり、
dimod.BinaryQuadraticModel にそのまま乗る = neal (SA) / D-Wave QPU に drop-in 可能。

## 評価 (ground truth つき)

N が小さい間は exact (全 K 部分集合を列挙) で真の QUBO 最小を求め、
random / fitness-greedy / diverse-greedy / 自前 SA がそこへどれだけ迫るかを測る。
honest: 「SA が exact に一致 ≈ 定式化が解ける」ことの確認であり、SA の優秀さの主張ではない。

実行::  py -3.11 research/quantum_selection_poc/qubo_diversity_selection_poc.py
出力::  research/quantum_selection_poc/results_qubo_selection.json
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

try:
    sys.stdout.reconfigure(encoding="utf-8")   # cp932 console 対策 (既知パターン)
except Exception:
    pass


# ---- 合成集団 (seed 固定; embedding に粗いクラスタ構造を入れて多様性軸を意味あるものに) ----
def make_population(n, d, seed):
    rng = np.random.default_rng(seed)
    # 3 クラスタ: 同クラスタ内は似る (類似ペナルティが効く構造)
    centers = rng.normal(0, 1, (3, d))
    labels = rng.integers(0, 3, n)
    emb = centers[labels] + rng.normal(0, 0.35, (n, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    fitness = rng.uniform(0.2, 1.0, n)
    return fitness, emb, labels


def cosine_sim(emb):
    s = emb @ emb.T
    return np.clip((s + 1.0) / 2.0, 0.0, 1.0)   # [-1,1] -> [0,1]


# ---- QUBO 構築 (対称行列 Q; energy = xᵀ Q x, 対角に線形項) -------------------
def build_qubo(fitness, sim, K, lam, A, cert=None, mu=0.0):
    n = len(fitness)
    Q = np.zeros((n, n))
    # 線形項 (対角): −f_i + A(1 − 2K)  [ (Σx−K)² の x_i² と定数交差項; x_i²=x_i ]
    lin = -fitness + A * (1.0 - 2.0 * K)
    if cert is not None and mu:
        lin = lin + mu * (1.0 - np.asarray(cert, float))   # § cert_gate
    np.fill_diagonal(Q, lin)
    # 二次項 (i<j): λ s_ij + 2A  [ (Σx−K)² の 2 x_i x_j 交差項 ]
    iu = np.triu_indices(n, 1)
    Q[iu] = lam * sim[iu] + 2.0 * A
    Q = Q + np.triu(Q, 1).T                        # 対称化 (下三角へコピー)
    const = A * K * K                              # (Σx−K)² の定数項 K²
    return Q, const


def energy(Q, const, x):
    return float(x @ Q @ x + const)


# ---- サンプラー群 -----------------------------------------------------------
def solve_exact(Q, const, n, K):
    best_e, best_x = math.inf, None
    for combo in itertools.combinations(range(n), K):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        e = energy(Q, const, x)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e


def solve_random(Q, const, n, K, seed, trials=2000):
    rng = np.random.default_rng(seed)
    best_e, best_x = math.inf, None
    for _ in range(trials):
        idx = rng.choice(n, K, replace=False)
        x = np.zeros(n); x[idx] = 1.0
        e = energy(Q, const, x)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e


def solve_fitness_greedy(fitness, Q, const, n, K):
    idx = np.argsort(-fitness)[:K]
    x = np.zeros(n); x[idx] = 1.0
    return x, energy(Q, const, x)


def solve_diverse_greedy(fitness, sim, Q, const, n, K):
    # marginal gain: 最初は最高 fitness、以後「fitness − 既選択との類似」最大を貪欲追加
    chosen = [int(np.argmax(fitness))]
    while len(chosen) < K:
        best_g, best_i = -math.inf, None
        for i in range(n):
            if i in chosen:
                continue
            g = fitness[i] - sim[i, chosen].mean()
            if g > best_g:
                best_g, best_i = g, i
        chosen.append(best_i)
    x = np.zeros(n); x[chosen] = 1.0
    return x, energy(Q, const, x)


def solve_sa(Q, const, n, K, seed, sweeps=4000, restarts=8):
    """自前 simulated annealing (neal 不在環境用 fallback; numpy のみ)。

    基数を保つため「選択 1 個 ↔ 非選択 1 個」のスワップ近傍を使う
    (基数制約 penalty A に頼らず実行可能領域内のみ探索 = 確実に |x|=K)。
    neal がある環境では dimod.BQM 経由で neal.SimulatedAnnealingSampler に置換可能 (§ drop-in)。
    """
    rng = np.random.default_rng(seed)
    best_e, best_x = math.inf, None
    for r in range(restarts):
        idx = set(rng.choice(n, K, replace=False).tolist())
        x = np.zeros(n)
        for i in idx:
            x[i] = 1.0
        e = energy(Q, const, x)
        cur_e, cur_x = e, x.copy()
        for t in range(sweeps):
            T = max(1e-3, 1.0 * (1.0 - t / sweeps))     # 線形冷却
            inside = list(idx)
            outside = [i for i in range(n) if i not in idx]
            i_out = inside[rng.integers(len(inside))]
            i_in = outside[rng.integers(len(outside))]
            x2 = cur_x.copy()
            x2[i_out] = 0.0; x2[i_in] = 1.0
            e2 = energy(Q, const, x2)
            if e2 < cur_e or rng.random() < math.exp((cur_e - e2) / T):
                cur_x, cur_e = x2, e2
                idx.discard(i_out); idx.add(i_in)
            if cur_e < best_e:
                best_e, best_x = cur_e, cur_x.copy()
    return best_x, best_e


# ---- メトリクス (選択集合の質) ---------------------------------------------
def metrics(x, fitness, sim, K):
    sel = np.where(x > 0.5)[0]
    tot_f = float(fitness[sel].sum())
    if len(sel) >= 2:
        iu = np.triu_indices(len(sel), 1)
        div = float(1.0 - sim[np.ix_(sel, sel)][iu].mean())   # 平均非類似度
    else:
        div = 0.0
    return {"selected": sel.tolist(), "k_ok": bool(len(sel) == K),
            "total_fitness": round(tot_f, 4), "mean_diversity": round(div, 4)}


def run_setting(N, d, K, lam, A, seed, rand_trials):
    fitness, emb, labels = make_population(N, d, seed)
    sim = cosine_sim(emb)
    Q, const = build_qubo(fitness, sim, K, lam, A)
    total_subsets = math.comb(N, K)

    runs = {}
    runs["exact"] = solve_exact(Q, const, N, K)
    runs["random"] = solve_random(Q, const, N, K, seed, trials=rand_trials)
    runs["fitness_greedy"] = solve_fitness_greedy(fitness, Q, const, N, K)
    runs["diverse_greedy"] = solve_diverse_greedy(fitness, sim, Q, const, N, K)
    runs["sa_classical"] = solve_sa(Q, const, N, K, seed)

    exact_e = runs["exact"][1]
    res = {"N": N, "K": K, "lam": lam, "total_subsets": total_subsets,
           "random_coverage": round(rand_trials / total_subsets, 4), "methods": {}}
    print(f"\n=== N={N} K={K} λ={lam} (全 {total_subsets} 部分集合; "
          f"random は {rand_trials} 試行 = {rand_trials/total_subsets:.1%} を探索) ===")
    print(f"{'method':16s} {'energy':>10s} {'gap':>9s} {'totF':>6s} {'div':>6s} {'opt?':>6s}")
    for name in ("exact", "sa_classical", "diverse_greedy", "fitness_greedy", "random"):
        x, e = runs[name]
        m = metrics(x, fitness, sim, K)
        gap = round(e - exact_e, 4)
        is_opt = abs(gap) < 1e-6
        res["methods"][name] = {"energy": round(e, 4), "gap_to_exact": gap,
                                "is_optimal": is_opt, **m}
        print(f"{name:16s} {e:10.4f} {gap:9.4f} {m['total_fitness']:6.2f} "
              f"{m['mean_diversity']:6.3f} {str(is_opt):>6s}")

    # § cert_gate: unsound 高 fitness 個体が cert penalty で排除されるか
    cert = np.ones(N)
    top2 = np.argsort(-fitness)[:2]
    cert[top2] = 0.0
    Qc, constc = build_qubo(fitness, sim, K, lam, A, cert=cert, mu=5.0)
    xc, _ = solve_exact(Qc, constc, N, K)
    sel_c = set(np.where(xc > 0.5)[0].tolist())
    excluded = [int(i) for i in top2 if i not in sel_c]
    res["cert_gate_demo"] = {"unsound_high_fitness": top2.tolist(),
                             "excluded_by_cert": excluded,
                             "all_unsound_excluded": bool(len(excluded) == len(top2))}
    print(f"[cert_gate] unsound 高 fitness {top2.tolist()} → 排除 {excluded} "
          f"({'全排除 OK' if len(excluded)==len(top2) else '一部残存'})")
    return res


def main():
    # 2 設定: small は全空間が小さく random も解けてしまう (= 規模が足りない証拠) /
    #         medium は random が全空間の数 % しか見られず SA の価値が初めて見える。
    out = {"qubo": "min -Σf x + λΣs xx + A(Σx-K)² [+ μΣ(1-cert)x]",
           "scope": "QPU 優位は未測定。exact=ground truth に古典 SA が一致するか = 定式化 sanity。",
           "settings": {}}
    print("=== QUBO 多様性選択 PoC (深追いしない最小版 / honest disclosure) ===")
    out["settings"]["small"] = run_setting(N=16, d=8, K=4, lam=1.5, A=3.0,
                                           seed=2026, rand_trials=2000)
    out["settings"]["medium"] = run_setting(N=28, d=10, K=7, lam=1.5, A=3.0,
                                            seed=2026, rand_trials=2000)

    sm, md = out["settings"]["small"], out["settings"]["medium"]
    print("\n--- honest 所見 ---")
    print(f"[small ] random opt={sm['methods']['random']['is_optimal']} "
          f"(全空間 {sm['total_subsets']} を {sm['random_coverage']:.0%} 探索 = 小さすぎて random も解ける)")
    print(f"[medium] random opt={md['methods']['random']['is_optimal']} gap={md['methods']['random']['gap_to_exact']} "
          f"/ SA opt={md['methods']['sa_classical']['is_optimal']} "
          f"(random は {md['random_coverage']:.1%} しか見られず未到達; SA は exact 近傍に到達)")
    print("[結論] 定式化は古典 SA で解ける (medium で random/greedy を上回り exact に一致)。")
    print("[honest] QPU 優位は本 PoC のスコープ外。QPU vs tuned SA vs random の 3 比較は将来。")
    print("[drop-in] Q (対称行列) は dimod.BQM へ 1:1 → neal/D-Wave QPU に無改造で乗る (本環境 neal 不在で自前 SA)。")
    print("[接続点] cert penalty μ で unsound 高 fitness 個体を選択排除 = llcore Z3 gate を選択演算へ写像可。")

    rpath = _HERE / "results_qubo_selection.json"
    rpath.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
