# SPDX-License-Identifier: Apache-2.0
"""R-endo viability A/B — 発散しうる基質 3 種で内的 gate に「自己保存の仕事」を与える決着 run.

## 背景 (なぜなぜ分析の帰結)

R-endo (run_d_internal_ab) は有界 CopyTask で autonomy null だった。根本原因 =「無条件有界な基質では
証明対象 (収縮) と生存がデカップル」。本 runner は生存を収縮に再結合した 3 基質 (viability_substrates) で
内的 gate に自己保存の仕事を与え、ENDO (環境結合 self-gate) が EXO_fixed (固定 gate) より生存・適応するかを測る。

## 設計 (環境依存の死 = #4)

DisturbedCopy 的 memory タスク (delay=8) を各発散基質で評価。環境 = 外乱 w_env を mid-run でステップ:
phase1 (G1 世代) w_env=W_LOW → phase2 (G2 世代) w_env=W_HIGH。集団継続 + 同一 rng。

**死** = 実測誤差包絡 (obs_gain 込み, max_t |s_disturbed−s_ref|∞) > 生存閾 V、または非有限/|state|>OVERFLOW。
死んだ gene の fitness = 0 (memory score を得られない)。生存閾 V は環境非依存だが、誤差包絡が w_env に比例
するため、**w_env↑ で死の境界が動く** = marginal-収縮 gene が W_HIGH で死ぬ。

**gate** (admit 条件) = 収縮 (L<1) ∧ 認証 tube `r=G·w̄/(1−L) ≤ V`。**soundness の帰結**: admit された gene の
実測包絡 ≤ 認証 tube ≤ V → **死なない** (gerrymander でなく定理)。
- **NONE**: gate なし。 **EXO_fixed**: gate w̄=W_LOW 固定 (設計時環境)。 **ENDO**: gate w̄=現 w_env (環境結合)。

phase1 は ENDO=EXO_fixed (同 w̄=W_LOW)。分岐は phase2 (W_HIGH) のみ = 効果を環境変化に isolate。

## honest 前提

内的化 ≡ 環境結合適応 gating (ENDO ≡ 外部 adaptive gate; location shift 自体は無価値)。本 runner が問うのは
「**適応的自己保存** (内的化が可能にする) が、固定 gate が死ぬ環境で生存・適応を生むか」。
V は基質ごとに固定の pre-registered 定数 (死の境界が active = NONE が W_HIGH で死ぬ regime を smoke で確認済)。
結果の頑健性は survival rate (V の正確値に依らず解釈可能) で担保。

## 事前登録 (PRE-REGISTRATION — 結果取得前に commit)

各基質 (linear/softsat/highgain) 独立に、phase2 (W_HIGH) で ENDO vs EXO_fixed を paired 比較 (seeds 3000-3019, n=20):
- **H_survival (本丸, confirmatory)**: ENDO の最終集団 survival rate > EXO_fixed。sign-flip permutation
  (両側, 10^5 resamples, seed=13), α=0.05。「内的 gate が自己保存の仕事を持つ」の中核。
- **H1 (fitness, confirmatory)**: ENDO の phase2 final best fitness ≥ EXO_fixed (死=0 を含む net fitness)。
- **H2 (re-adaptation AUC, exploratory)**: ENDO phase2 AUC vs EXO_fixed。
- **H3 (soundness, 必須)**: ENDO の最終集団 survival rate == 1.0 (admit gene は死なない = soundness 実証)。
  < 1.0 が出れば認証/実装の破綻 (致命)。
- 反証: (F1) H_survival で ENDO ≤ EXO → 自己保存に優位なし。 (F2) NONE が W_HIGH で死なない (死境界 inactive)
  → 本基質は viability 非脅威で実験無効 (V 要再設計)。 (F3) ENDO survival < 1.0 → soundness 破綻。
- 判定: H_survival p<0.05 ∧ ENDO>EXO ∧ H3 (ENDO surv=1.0) → **内的 gate の自己保存に固有価値あり** (基質別)。

実行::  py -3.11 research/internalization_poc/run_viability_ab.py
出力::  research/internalization_poc/results_viability_ab.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_VMP = _HERE.parents[0] / "verified_memory_poc"
for _p in (str(_HERE.parents[1] / "src"), str(_VMP), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llcore.evolution.minimal_ga import (  # noqa: E402  (production GA operator 再利用)
    Individual, Population, crossover_uniform, initialize_random_population,
    tournament_select, uniform_mutate,
)
from llcore.fitness import make_fixed_readout  # noqa: E402
from llcore.state_update import StateUpdateGene  # noqa: E402

from run_3arm_ab import _ensure_utf8_stdout  # noqa: E402
from run_c_decision import signflip_pvalue  # noqa: E402
from viability_substrates import ALL_SUBSTRATES  # noqa: E402

# ---- 事前登録パラメータ ------------------------------------------------------
STATE_DIM = 8
SEQ_LEN = 32
DELAY = 8
W_LOW = 0.05
W_HIGH = 0.20
G1 = 10
G2 = 10
D_SEEDS = list(range(3000, 3020))
ARMS = ["NONE", "EXO_fixed", "ENDO"]
N_TRIALS = 3            # fitness eval の trial 数 (disturbance 平均)
N_SURV_TRIALS = 10      # survival rate 計測の trial 数
OVERFLOW = 1e6
SETTLE_FRAC = 0.5
PERM_N_RESAMPLES = 100_000
PERM_RNG_SEED = 13
ALPHA = 0.05
# 生存閾 V (基質別 fixed; 死境界が active になる pre-registered 値)
V_BY_SUBSTRATE = {"linear": 0.3, "softsat": 0.3, "highgain": 5.0}
# GA (run_3arm_ab と同等)
POP, TOURN_K, SIGMA, CX_RATE, ELITISM, RESAMPLE_CAP = 20, 3, 0.15, 0.5, 1, 50

_READOUT = make_fixed_readout(STATE_DIM, STATE_DIM, seed=1001)


def _rollout(sub, inputs: np.ndarray, gene: StateUpdateGene) -> np.ndarray:
    """state 軌跡 (L+1, dim)。発散は大きな有限値として出る (32 step で inf には至らない)。"""
    g = gene.clipped()
    L, dim = inputs.shape
    s = np.zeros(dim, dtype=np.float64)
    out = np.empty((L + 1, dim), dtype=np.float64)
    out[0] = s
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(L):
            s = sub.step(s, inputs[t], g)
            out[t + 1] = s
    return out


def _eval(sub, gene: StateUpdateGene, w_env: float, V: float, rng: np.random.Generator):
    """(mean_fitness, survival_prob) を返す。死 (包絡>V or 非有限) なら trial fitness=0。"""
    settle = int((SEQ_LEN + 1) * SETTLE_FRAC)
    fits, alives = [], []
    for _ in range(N_TRIALS):
        clean = rng.uniform(-1.0, 1.0, size=(SEQ_LEN, STATE_DIM))
        target = clean[SEQ_LEN - 1 - DELAY].copy()
        d = rng.uniform(-w_env, w_env, size=clean.shape)
        s_ref = _rollout(sub, clean, gene)
        s_act = _rollout(sub, clean + d, gene)
        finite = np.isfinite(s_act).all() and np.isfinite(s_ref).all()
        if finite:
            env = sub.obs_gain * float(np.max(np.abs(s_act[settle:] - s_ref[settle:])))
            overflow = float(np.max(np.abs(s_act))) > OVERFLOW
        else:
            env, overflow = np.inf, True
        alive = bool(finite and not overflow and env <= V)
        if alive:
            pred = _READOUT(s_act[-1])
            mse = float(np.mean((pred - target) ** 2))
            mem = 1.0 / (1.0 + mse)
        else:
            mem = 0.0
        fits.append(mem if alive else 0.0)
        alives.append(1.0 if alive else 0.0)
    return float(np.mean(fits)), float(np.mean(alives))


def _admits(sub, gene: StateUpdateGene, w_bar: float, V: float) -> bool:
    g = gene.clipped()
    L = sub.L(g)
    if not (L < 1.0):
        return False
    tube = sub.G(g) * w_bar / (1.0 - L)
    return bool(tube <= V)


def _gate_fn(sub, arm: str, w_env: float, V: float):
    if arm == "NONE":
        return lambda g: True
    gate_w = w_env if arm == "ENDO" else W_LOW   # EXO_fixed は設計時 W_LOW 固定
    return lambda g: _admits(sub, g, gate_w, V)


def _ga_phase(sub, w_env: float, gate_fn, V: float, init_genes, rng, n_gen):
    """minimal GA を 1 phase 走らせる (production operator 再利用, 任意 substrate/gate)。"""
    def ff(g):
        return _eval(sub, g, w_env, V, rng)[0]

    genes = initialize_random_population(POP, rng) if init_genes is None else list(init_genes)
    pop = Population(tuple(Individual(gene=g, fitness=ff(g)) for g in genes))
    best_curve = [pop.best.fitness]
    for _ in range(n_gen):
        ranked = sorted(pop.individuals, key=lambda i: i.fitness, reverse=True)
        children = [e.gene for e in ranked[:ELITISM]]
        while len(children) < POP:
            pa = tournament_select(pop, TOURN_K, rng).gene
            if rng.random() < CX_RATE:
                pb = tournament_select(pop, TOURN_K, rng).gene
                child = crossover_uniform(pa, pb, rng)
            else:
                child = uniform_mutate(pa, SIGMA, rng)
            tries = 0
            while not gate_fn(child) and tries < RESAMPLE_CAP:
                src = tournament_select(pop, TOURN_K, rng).gene
                child = uniform_mutate(src, SIGMA, rng)
                tries += 1
            children.append(child)
        pop = Population(tuple(Individual(gene=g, fitness=ff(g)) for g in children))
        best_curve.append(pop.best.fitness)
    return [i.gene for i in pop.individuals], list(best_curve), pop


def _pop_survival_rate(sub, genes, w_env: float, V: float, seed: int) -> float:
    rng = np.random.default_rng(800000 + seed)
    return float(np.mean([_eval(sub, g, w_env, V, rng)[1] for g in genes]))


def _run_arm(sub, arm: str, seed: int, V: float) -> dict:
    rng = np.random.default_rng(seed)
    genes1, curve1, _ = _ga_phase(sub, W_LOW, _gate_fn(sub, arm, W_LOW, V), V, None, rng, G1)
    genes2, curve2, pop2 = _ga_phase(sub, W_HIGH, _gate_fn(sub, arm, W_HIGH, V), V, genes1, rng, G2)
    surv = _pop_survival_rate(sub, genes2, W_HIGH, V, seed)
    return {
        "arm": arm, "seed": seed,
        "phase2_final_best_fitness": float(pop2.best.fitness),
        "phase2_auc": float(np.sum(curve2)),
        "phase2_survival_rate": surv,
    }


def _paired(recs, a, b, key):
    va = np.array([r[key] for r in recs if r["arm"] == a])
    vb = np.array([r[key] for r in recs if r["arm"] == b])
    return va - vb


def _stat(deltas):
    return {
        "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
        "n_positive": int((deltas > 0).sum()), "n_negative": int((deltas < 0).sum()),
        "p_signflip_two_sided": signflip_pvalue(deltas, n_resamples=PERM_N_RESAMPLES, seed=PERM_RNG_SEED),
    }


def run_substrate(sub) -> dict:
    V = V_BY_SUBSTRATE[sub.name]
    recs = []
    for arm in ARMS:
        for seed in D_SEEDS:
            recs.append(_run_arm(sub, arm, seed, V))
    means = {arm: {
        "survival": float(np.mean([r["phase2_survival_rate"] for r in recs if r["arm"] == arm])),
        "fitness": float(np.mean([r["phase2_final_best_fitness"] for r in recs if r["arm"] == arm])),
        "auc": float(np.mean([r["phase2_auc"] for r in recs if r["arm"] == arm])),
    } for arm in ARMS}

    h_surv = _stat(_paired(recs, "ENDO", "EXO_fixed", "phase2_survival_rate"))
    h1 = _stat(_paired(recs, "ENDO", "EXO_fixed", "phase2_final_best_fitness"))
    h2 = _stat(_paired(recs, "ENDO", "EXO_fixed", "phase2_auc"))
    endo_surv = means["ENDO"]["survival"]
    none_surv = means["NONE"]["survival"]
    h3_sound = bool(endo_surv >= 0.999)
    boundary_active = bool(none_surv < 0.95)   # NONE が W_HIGH で死ぬ regime か (F2)

    if not boundary_active:
        verdict = (f"INVALID (死境界 inactive, F2): NONE survival={none_surv:.3f} ≥0.95 = 本基質は "
                   f"W_HIGH でも viability 非脅威。V 再設計が必要。")
    elif not h3_sound:
        verdict = (f"SOUNDNESS 破綻 (F3, 致命): ENDO survival={endo_surv:.3f} <1.0 = admit gene が死んだ。"
                   f"認証 tube ≤ V が実測包絡を bound できていない。")
    elif h_surv["p_signflip_two_sided"] < ALPHA and h_surv["mean_delta"] > 0:
        verdict = (f"内的 gate の自己保存に固有価値あり [{sub.name}]: ENDO survival > EXO_fixed "
                   f"(Δ={h_surv['mean_delta']:+.3f}, p={h_surv['p_signflip_two_sided']:.4f}); "
                   f"ENDO 健全に生存 (surv={endo_surv:.3f}) ∧ EXO は固定 gate で W_HIGH で死ぬ "
                   f"(surv={means['EXO_fixed']['survival']:.3f})。fitness Δ={h1['mean_delta']:+.3f}。")
    else:
        verdict = (f"自己保存優位なし [{sub.name}] (F1): ENDO survival ≈ EXO_fixed "
                   f"(Δ={h_surv['mean_delta']:+.3f}, p={h_surv['p_signflip_two_sided']:.4f})。")

    return {
        "substrate": sub.name, "V": V, "obs_gain": sub.obs_gain,
        "arm_means_phase2": means,
        "H_survival_endo_vs_exo": h_surv,
        "H1_fitness_endo_vs_exo": h1,
        "H2_auc_endo_vs_exo": h2,
        "H3_endo_survival_rate": endo_surv, "h3_soundness_ok": h3_sound,
        "boundary_active": boundary_active, "none_survival": none_surv,
        "verdict": verdict, "records": recs,
    }


def run_all() -> dict:
    _ensure_utf8_stdout()
    t0 = time.time()
    out = {"preregistration": {
        "W_LOW": W_LOW, "W_HIGH": W_HIGH, "G1": G1, "G2": G2, "seeds": D_SEEDS,
        "V_by_substrate": V_BY_SUBSTRATE, "delay": DELAY, "seq_len": SEQ_LEN,
        "alpha": ALPHA, "perm_seed": PERM_RNG_SEED, "arms": ARMS,
        "pop": POP, "tourn_k": TOURN_K, "sigma": SIGMA,
    }, "substrates": {}}
    for sub in ALL_SUBSTRATES:
        res = run_substrate(sub)
        out["substrates"][sub.name] = res
        m = res["arm_means_phase2"]
        print(f"  [{sub.name}] surv NONE={m['NONE']['survival']:.2f} EXO={m['EXO_fixed']['survival']:.2f} "
              f"ENDO={m['ENDO']['survival']:.2f} | H_surv Δ={res['H_survival_endo_vs_exo']['mean_delta']:+.3f} "
              f"p={res['H_survival_endo_vs_exo']['p_signflip_two_sided']:.4f} | "
              f"H3 endo_surv={res['H3_endo_survival_rate']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def main() -> int:
    out = run_all()
    (_HERE / "results_viability_ab.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {_HERE / 'results_viability_ab.json'}")
    print("\n=== R-endo viability A/B (ENDO − EXO_fixed, phase2 W_HIGH, paired n=20) ===")
    for name, res in out["substrates"].items():
        print(f"\n[{name}] V={res['V']} obs_gain={res['obs_gain']}")
        m = res["arm_means_phase2"]
        for arm in ARMS:
            print(f"   {arm:10s} survival={m[arm]['survival']:.3f} fitness={m[arm]['fitness']:.4f} auc={m[arm]['auc']:.2f}")
        hs = res["H_survival_endo_vs_exo"]
        print(f"   H_survival Δ={hs['mean_delta']:+.3f} p={hs['p_signflip_two_sided']:.4f} +{hs['n_positive']}/-{hs['n_negative']}")
        print(f"   H3 soundness: ENDO surv={res['H3_endo_survival_rate']:.3f} ok={res['h3_soundness_ok']} | boundary_active={res['boundary_active']}")
        print(f"   verdict: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
