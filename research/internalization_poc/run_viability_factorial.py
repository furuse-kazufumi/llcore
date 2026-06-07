# SPDX-License-Identifier: Apache-2.0
"""R-endo viability factorial 2³ — 記憶形成 3 機構の組み合わせ + 交互作用 (相乗効果)。

5-arm A/B (run_viability_ab) は各機構を単独で比較した。本 runner は **E (ENDO 自己予見) /
R (REVIVE 復活修復) / O (OBSERVE 社会観察) の全 2³=8 組み合わせ**を同一 seed で回し、
factorial 効果分解 (main effects + 2-way/3-way interactions) で相乗・冗長を測る。

## 合成の意味論 (5-arm 実装の自然な拡張; 助走版 phase 構成を踏襲)

- **E**: 評価前 pre-screen — `_admits(g, κ_current, V)` (sound, 環境結合) を満たさない child を resample。
- **O**: 評価前 pre-screen — 観察された死 (death_memory) 近傍の child を resample。E と同時 on なら
  両条件の AND (sound gate ∧ empirical 回避)。
- **R**: pre-screen しない。死を経験したら蘇生 (現 κ で `_repair`, 記憶 mix 保持) して再評価・集団へ。
- 全機構 off = NONE 相当。EXO_fixed は factorial 対象外 (5-arm でカバー済)。

## 事前登録 (PRE-REGISTRATION — 結果取得前に commit)

指標 = measure 期 deaths (死回避) と pop_mean_fitness (記憶保存)。n=20 paired seeds、
factorial contrast (係数 ±1, 正規化 1/4 = main effect が mean_on − mean_off に一致) を per-seed で
計算し sign-flip permutation (10^5, seed=13) で検定。

- **H_ER_best_of_both (本丸)**: E+R は「予見が死を消し、漏れを蘇生が保つ」= 両軸の良いとこ取り。
  判定: deaths_ER ≤ deaths_E + 1.0 かつ pop_mean_ER ≥ max(pop_mean_E, pop_mean_R) − 0.02
  (per substrate, mean 比較; 対応する contrast p も報告)。
- **H_EO_redundant (sound が empirical を冗長化)**: O の死削減の限界効果は E on でほぼ消える。
  判定: |deaths_EO − deaths_E| < |deaths_O − deaths_NONE| (E が O を包含)。
  interaction_EO (deaths) > 0 方向 (O の負の効果が E で打ち消される)。
- **交互作用の探索的報告**: 3-way (ERO) と R 系交互作用は方向を事前登録しない (探索的と明記)。
- soundness: E/R は 5-arm と同一 `_admits`/`_repair` を流用 (viol=0 は run_viability_ab で確認済)。
- substrate = linear + highgain (死境界 active な 2 基質; softsat は境界弱で除外)。

実行::  py -3.11 research/internalization_poc/run_viability_factorial.py
出力::  research/internalization_poc/results_viability_factorial.json
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[1] / "src"), str(_HERE.parents[0] / "verified_memory_poc"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_viability_ab as M  # noqa: E402
from llcore.evolution.minimal_ga import (  # noqa: E402
    Individual, Population, crossover_uniform, initialize_random_population,
    tournament_select, uniform_mutate,
)
from viability_substrates import HighGainSubstrate, LinearSubstrate  # noqa: E402

SUBSTRATES = [LinearSubstrate(), HighGainSubstrate()]
MECHS = ("E", "R", "O")
COMBOS = [frozenset(c) for k in range(4) for c in combinations(MECHS, k)]   # 8 combos, NONE 含む
ER_TOL_DEATHS = 1.0     # H_ER_best_of_both: deaths_ER ≤ deaths_E + これ
ER_TOL_POPMEAN = 0.02   # H_ER_best_of_both: pop_mean_ER ≥ max(E,R) − これ


def _combo_name(mech: frozenset) -> str:
    return "".join(m for m in MECHS if m in mech) or "NONE"


def _ga_phase_combo(sub, mech, kappa, V, init_genes, rng, n_gen, death_memory):
    """M._ga_phase の combo 版 — 機構集合 mech ⊆ {E,R,O} を合成適用。"""
    deaths = [0]
    dm = death_memory

    def _eval_count(g):
        fit, surv = M._eval(sub, g, kappa, V, rng)
        if surv < 1.0:
            deaths[0] += 1
            dm.append(M._normgene(g.clipped()))
            if "R" in mech:
                g2 = M._repair(sub, g, kappa, V)        # 蘇生: 現 κ で修復 (環境結合, 記憶 mix 保持)
                fit2, _ = M._eval(sub, g2, kappa, V, rng)
                return g2, fit2
        return g, fit

    def _ok(child):
        if "E" in mech and not M._admits(sub, child, kappa, V):
            return False
        if "O" in mech and M._near_observed_death(child, dm):
            return False
        return True

    def _pre_screen(child):
        if not ({"E", "O"} & mech):
            return child
        tries = 0
        while tries < M.RESAMPLE_CAP:
            if _ok(child):
                return child
            src = tournament_select(pop, M.TOURN_K, rng).gene
            child = uniform_mutate(src, M.SIGMA, rng)
            tries += 1
        return child

    genes = initialize_random_population(M.POP, rng) if init_genes is None else list(init_genes)
    pop = Population(tuple(Individual(*_eval_count(g)) for g in genes))
    for _ in range(n_gen):
        ranked = sorted(pop.individuals, key=lambda i: i.fitness, reverse=True)
        children = [e.gene for e in ranked[:M.ELITISM]]
        while len(children) < M.POP:
            pa = tournament_select(pop, M.TOURN_K, rng).gene
            if rng.random() < M.CX_RATE:
                pb = tournament_select(pop, M.TOURN_K, rng).gene
                child = crossover_uniform(pa, pb, rng)
            else:
                child = uniform_mutate(pa, M.SIGMA, rng)
            children.append(_pre_screen(child))
        pop = Population(tuple(Individual(*_eval_count(g)) for g in children))
    return [i.gene for i in pop.individuals], pop, deaths[0]


def _run_combo(sub, mech, seed, V):
    """M._run_arm と同じ 3 phase 構成 (G1 κ_low → 助走 κ_high → 測定 κ_high)。"""
    rng = np.random.default_rng(seed)
    dm = []
    genes1, _, d1 = _ga_phase_combo(sub, mech, M.KAPPA_LOW, V, None, rng, M.G1, dm)
    genes_wu, _, _ = _ga_phase_combo(sub, mech, M.KAPPA_HIGH, V, genes1, rng, M.G2_WARMUP, dm)
    genes2, pop2, d2 = _ga_phase_combo(sub, mech, M.KAPPA_HIGH, V, genes_wu, rng, M.G2_MEASURE, dm)
    return {
        "combo": _combo_name(mech), "seed": seed, "phase2_deaths": int(d2),
        "phase2_pop_mean_fitness": float(pop2.fitness_array.mean()),
        "phase2_final_best_fitness": float(pop2.best.fitness),
    }


def _contrast(values_by_combo, effect: frozenset):
    """factorial contrast (per-seed): (1/4)·Σ_c y(c)·Π_{m∈effect} x_m(c), x_m=+1(on)/−1(off)。
    正規化 1/4 で main effect = mean_on − mean_off に一致。"""
    acc = 0.0
    for mech in COMBOS:
        coef = 1.0
        for m in effect:
            coef *= 1.0 if m in mech else -1.0
        acc += coef * values_by_combo[_combo_name(mech)]
    return acc / 4.0


def _stat(deltas: np.ndarray):
    from run_c_decision import signflip_pvalue
    return {
        "mean": float(deltas.mean()), "median": float(np.median(deltas)),
        "p_signflip_two_sided": signflip_pvalue(
            deltas, n_resamples=M.PERM_N_RESAMPLES, seed=M.PERM_RNG_SEED),
    }


def run_substrate(sub):
    V = M.V_BY_SUBSTRATE[sub.name]
    recs = [_run_combo(sub, mech, seed, V) for mech in COMBOS for seed in M.D_SEEDS]
    means = {_combo_name(mech): {
        "deaths": float(np.mean([r["phase2_deaths"] for r in recs
                                 if r["combo"] == _combo_name(mech)])),
        "pop_mean": float(np.mean([r["phase2_pop_mean_fitness"] for r in recs
                                   if r["combo"] == _combo_name(mech)])),
        "best": float(np.mean([r["phase2_final_best_fitness"] for r in recs
                               if r["combo"] == _combo_name(mech)])),
    } for mech in COMBOS}

    # per-seed factorial contrasts (deaths / pop_mean)
    effects = {}
    eff_sets = ([frozenset({m}) for m in MECHS]
                + [frozenset(c) for c in combinations(MECHS, 2)] + [frozenset(MECHS)])
    for metric, key in (("deaths", "phase2_deaths"), ("pop_mean", "phase2_pop_mean_fitness")):
        by_seed = {}
        for r in recs:
            by_seed.setdefault(r["seed"], {})[r["combo"]] = float(r[key])
        for eff in eff_sets:
            deltas = np.array([_contrast(by_seed[s], eff) for s in M.D_SEEDS])
            effects[f"{metric}__{_combo_name(eff)}"] = _stat(deltas)

    # 事前登録判定
    d, pm = ({c: means[c]["deaths"] for c in means}, {c: means[c]["pop_mean"] for c in means})
    h_er = bool(d["ER"] <= d["E"] + ER_TOL_DEATHS
                and pm["ER"] >= max(pm["E"], pm["R"]) - ER_TOL_POPMEAN)
    h_eo = bool(abs(d["EO"] - d["E"]) < abs(d["O"] - d["NONE"]))
    boundary_active = bool(d["NONE"] > 1.0)
    return {
        "substrate": sub.name, "V": V, "combo_means": means, "effects": effects,
        "boundary_active": boundary_active,
        "H_ER_best_of_both": h_er, "H_EO_redundant": h_eo,
        "records": recs,
    }


def run_all():
    M._ensure_utf8_stdout()
    t0 = time.time()
    out = {"preregistration": {
        "combos": [_combo_name(c) for c in COMBOS], "seeds": M.D_SEEDS,
        "substrates": [s.name for s in SUBSTRATES],
        "G1": M.G1, "G2_WARMUP": M.G2_WARMUP, "G2_MEASURE": M.G2_MEASURE,
        "kappa_low": M.KAPPA_LOW, "kappa_high": M.KAPPA_HIGH,
        "er_tol_deaths": ER_TOL_DEATHS, "er_tol_popmean": ER_TOL_POPMEAN,
        "alpha": M.ALPHA, "perm_seed": M.PERM_RNG_SEED,
        "hypotheses": ["H_ER_best_of_both", "H_EO_redundant"],
        "exploratory": ["R 系交互作用", "3-way ERO"],
    }, "substrates": {}}
    for sub in SUBSTRATES:
        res = run_substrate(sub)
        out["substrates"][sub.name] = res
        m = res["combo_means"]
        print(f"  [{sub.name}] deaths " + " ".join(
            f"{c}={m[c]['deaths']:.0f}" for c in ("NONE", "E", "R", "O", "ER", "EO", "RO", "ERO"))
            + " | pop_mean " + " ".join(
            f"{c}={m[c]['pop_mean']:.2f}" for c in ("NONE", "E", "R", "ER", "ERO"))
            + f" | H_ER={res['H_ER_best_of_both']} H_EO={res['H_EO_redundant']} "
            f"({time.time()-t0:.0f}s)", flush=True)
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def main():
    out = run_all()
    (_HERE / "results_viability_factorial.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {_HERE / 'results_viability_factorial.json'}")
    print("\n=== factorial 2^3 — 記憶形成 3 機構の組み合わせ (n=20, 助走版) ===")
    for name, res in out["substrates"].items():
        print(f"\n[{name}] boundary_active={res['boundary_active']} "
              f"H_ER_best_of_both={res['H_ER_best_of_both']} H_EO_redundant={res['H_EO_redundant']}")
        for c in ("NONE", "E", "R", "O", "ER", "EO", "RO", "ERO"):
            m = res["combo_means"][c]
            print(f"   {c:5s} deaths={m['deaths']:6.1f} pop_mean={m['pop_mean']:.4f} best={m['best']:.4f}")
        for k in ("deaths__ER", "deaths__EO", "pop_mean__ER"):
            e = res["effects"][k]
            print(f"   int {k}: mean={e['mean']:+.3f} (p={e['p_signflip_two_sided']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
