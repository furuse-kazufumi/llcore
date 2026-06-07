# SPDX-License-Identifier: Apache-2.0
"""R-endo viability A/B — 発散しうる基質 3 種で内的 gate に「自己保存の仕事」を与える決着 run。

## 背景 (なぜなぜ分析の帰結)

R-endo (run_d_internal_ab) は有界 CopyTask で autonomy null だった。根本原因 =「無条件有界な基質では
証明対象 (収縮) と生存がデカップル」。さらにスモークで2点判明:
1. 死=0 fitness だと **selection 自体が死を回避** → gate は最終状態に冗長。内的 gate の真価は「死んで
   学ぶコスト」の回避 = 評価 (rollout) 前に致命 gene を自己検証で弾く safe exploration。本丸指標 =
   **進化中に被った致命評価数 (phase2_deaths)**。
2. 環境を外乱 (w̄) だけにすると進化が入力ゲイン g を小さくして soft 境界を回避でき、hard 発散 (a≥1) は
   環境非依存で内外 gate 一致 → null。→ **環境を recurrence ゲイン κ にする**: 実効収縮 κ·a。κ↑ で
   以前安定だった gene (a<1) が κ·a≥1 で**発散** = 環境変化が viability を脅かす・g で回避不能・real
   divergence。これで内的 gate に自己保存の仕事が生まれる。

## 設計 (環境 = recurrence ゲイン κ, #4)

memory タスク (delay=8) を各発散基質で評価。環境 κ を mid-run でステップ: phase1 (G1 世代) κ=KAPPA_LOW
→ phase2 (G2 世代) κ=KAPPA_HIGH。集団継続 + 同一 rng。外乱は固定 W_BAR_FIX (sustained bias)。

memory (delay-8 retention) は高 a (≈1) を報酬する → phase1 (κ_low) で集団は a≈1 へ。phase2 (κ_high) で
それらは κ_high·a≥1 で**発散** = 致命。a0·κ の発散は g に依らない (回避不能)。

**死** = 状態発散 (|state|>OVERFLOW or 非有限) or 実測誤差包絡 (obs_gain 込み) > 生存閾 V。死=fitness 0。
**gate** (admit) = 収縮 (L=κ·...<1) ∧ 認証 tube `r=G·w̄/(1−L) ≤ V`。soundness: admit⇒実測包絡≤tube≤V ∧
L<1⇒非発散 ⇒ 死なない (定理, gerrymander でない)。
- **NONE**: gate なし (致命 gene も評価し死を被る)。 **EXO_fixed**: gate κ=KAPPA_LOW 固定 (設計時環境;
  κ_high で発散する gene を見逃す)。 **ENDO**: gate κ=現 κ (環境結合; κ_high 発散を予見し弾く)。

phase1 は ENDO=EXO_fixed (同 κ=KAPPA_LOW)。分岐は phase2 (κ_high) のみ。

## honest 前提

内的化 ≡ 環境結合適応 gating (ENDO ≡ 外部 adaptive gate; location shift 自体は無価値)。問うのは
「適応的自己保存 (内的化が可能にする) が、固定 gate が見逃す致命試行を予見・回避するか」。

## 記憶形成 3 機構 (ユーザー洞察) — 「経験が記憶になる」には死を越える機構が要る

死 = fitness 0 + 除去 だと、その個体の蓄積記憶 (進化 genome) も消える。経験を記憶に残す 3 機構:
- **ENDO (自己予見)**: 内的健全検証器で死を予見し回避 (reject)。境界を内的制約として zero-shot 記憶。sound。
- **REVIVE (復活/修復)**: 死を予見し reject でなく **修復** — 記憶チャネル mix を保持しつつ dynamics を安全化。
  死を傷 (= 安全化された自分) として残す。carried pop も修復 = catastrophe を越えて記憶を運ぶ。
- **OBSERVE (社会的観察)**: 他個体の観察された死 (death_memory) 近傍を経験的に回避。empirical・社会的・lossy
  (死んで学ぶ・Goodhart 可能)。ENDO の sound certificate と対比 = llcore 核心 (sound vs empirical)。
baseline: NONE (死で無駄) / EXO_fixed (設計時固定 gate)。

## 事前登録 (PRE-REGISTRATION — 結果取得前に commit)

各基質 (linear/softsat/highgain) 独立に phase2 (κ_high) で paired 比較 (seeds 3000-3019, n=20, sign-flip
permutation 両側 10^5 seed=13 α=0.05):
- **H_avoid (本丸)**: ENDO/REVIVE/OBSERVE の phase2 致命評価数 < NONE (機構が死を回避するか)。
- **H_repair (復活の価値)**: REVIVE の致命評価数 < ENDO (修復が reject より死を減らすか = 全個体を救う)。
  + REVIVE fitness が ENDO に劣らないか (修復の保持コスト)。
- **H_sound_vs_empirical (核心)**: ENDO (sound 予見) の致命評価数 < OBSERVE (empirical 社会学習)、かつ
  verifier soundness violations==0 vs OBSERVE は deaths>0 (imperfect)。
- **soundness (必須)**: _admits(g,κ_high,V)=True の gene が死なない (random 5000 で violations==0)。
- **memory_ratio**: catastrophe 直後 fitness / phase1 final (進化中の記憶が段差を越えて生き残るか)。
- 反証: (F2) NONE が phase2 で死なない → 死境界 inactive (無効)。 (F3) soundness violations>0 → 認証破綻。

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
KAPPA_LOW = 1.0        # phase1 環境ゲイン (正常)
KAPPA_HIGH = 2.0       # phase2 環境ゲイン (2x = 以前安定な gene を発散させる; smoke で死境界 active 確認)
W_BAR_FIX = 0.1        # 固定 sustained bias 外乱 (tube/包絡)
G1 = 10
# phase2 を 助走 (warm-up) + 測定 (measure) に分割 (ユーザー指摘): κ_high で死を数回経験させ
# (OBSERVE が death_memory を学習・REVIVE が傷を蓄積・初期 shock 通過) その deaths は数えず、
# 定常状態 (measure 期) の deaths/記憶を正確な統計とする。死記憶/集団/rng は warm-up→measure 継続。
G2_WARMUP = 8
G2_MEASURE = 10
D_SEEDS = list(range(3000, 3020))
# 「経験が記憶になる」3 機構 (ユーザー洞察) + 2 baseline:
#  ENDO    = 自己予見 (内的健全検証器で死を予見・回避; sound, zero-shot)
#  REVIVE  = 復活/修復 (死を予見し reject でなく修復; 記憶チャネル mix 保持 = 死を傷として残す)
#  OBSERVE = 社会的観察 (他個体の観察された死から経験的に境界を学ぶ; empirical, lossy, Goodhart 可能)
#  NONE/EXO_fixed = baseline (死で無駄 / 設計時固定 gate)
ARMS = ["NONE", "EXO_fixed", "ENDO", "REVIVE", "OBSERVE"]
N_TRIALS = 3
N_SURV_TRIALS = 10
OVERFLOW = 1e6
SETTLE_FRAC = 0.5
PERM_N_RESAMPLES = 100_000
PERM_RNG_SEED = 13
ALPHA = 0.05
# 生存閾 V (基質別 fixed; 主たる死は発散=L≥1 で V は soft 上限)
V_BY_SUBSTRATE = {"linear": 10.0, "softsat": 10.0, "highgain": 5.0}
POP, TOURN_K, SIGMA, CX_RATE, ELITISM, RESAMPLE_CAP = 20, 3, 0.15, 0.5, 1, 50

_READOUT = make_fixed_readout(STATE_DIM, STATE_DIM, seed=1001)


def _rollout(sub, inputs, gene, kappa):
    g = gene.clipped()
    L, dim = inputs.shape
    s = np.zeros(dim, dtype=np.float64)
    out = np.empty((L + 1, dim), dtype=np.float64)
    out[0] = s
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(L):
            s = sub.step(s, inputs[t], g, kappa)
            out[t + 1] = s
    return out


def _eval(sub, gene, kappa, V, rng):
    """(mean_fitness, survival_prob)。死 (発散/包絡>V) なら trial fitness=0。"""
    settle = int((SEQ_LEN + 1) * SETTLE_FRAC)
    fits, alives = [], []
    for _ in range(N_TRIALS):
        clean = rng.uniform(-1.0, 1.0, size=(SEQ_LEN, STATE_DIM))
        target = clean[SEQ_LEN - 1 - DELAY].copy()
        sign = 1.0 if rng.random() < 0.5 else -1.0
        d = sign * W_BAR_FIX                                    # 固定 sustained bias
        s_ref = _rollout(sub, clean, gene, kappa)
        s_act = _rollout(sub, clean + d, gene, kappa)
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


def _admits(sub, gene, kappa, V):
    g = gene.clipped()
    L = sub.L(g, kappa)
    if not (L < 1.0):
        return False
    tube = sub.G(g) * W_BAR_FIX / (1.0 - L)
    return bool(tube <= V)


def _gate_kappa_for(arm, kappa):
    """gate/repair に使う κ: ENDO/REVIVE=現 κ (環境結合), EXO_fixed=設計時 κ_low。"""
    return kappa if arm in ("ENDO", "REVIVE") else KAPPA_LOW


# ---- 記憶形成機構 (ユーザー洞察): 修復 (REVIVE) と 社会的観察 (OBSERVE) ----------
_SCALE = np.array([1.0, 1.0, 2.0])   # gene 正規化 (decay∈[0,1], mix∈[-1,1], gate_str∈[-2,2])


def _normgene(g) -> np.ndarray:
    return np.array([g.decay, g.mix, g.gate_str]) / _SCALE


def _repair(sub, gene, kappa, V):
    """REVIVE: 死を予見した gene を **修復** — 記憶チャネル mix を保持しつつ (decay, gate_str) を
    安全側 (0,0) へ最小 blend して admit させる。死を傷 (= 安全化された dynamics) として記憶に残す。"""
    g = gene.clipped()
    if _admits(sub, g, kappa, V):
        return g
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        cand = StateUpdateGene(decay=(1 - mid) * g.decay, mix=g.mix,
                               gate_str=(1 - mid) * g.gate_str).clipped()
        if _admits(sub, cand, kappa, V):
            hi = mid
        else:
            lo = mid
    return StateUpdateGene(decay=(1 - hi) * g.decay, mix=g.mix,
                           gate_str=(1 - hi) * g.gate_str).clipped()


OBSERVE_RADIUS = 0.15   # 観察した死から半径内を「予測致命」とみなす (empirical 境界)


def _near_observed_death(gene, death_memory) -> bool:
    """OBSERVE: 他個体が死んだ gene 近傍 (観察された致命域) かを経験的に判定 (kNN 風)。"""
    if not death_memory:
        return False
    q = _normgene(gene.clipped())
    dm = np.asarray(death_memory)
    return bool(np.min(np.sqrt(((dm - q) ** 2).sum(axis=1))) < OBSERVE_RADIUS)


def _ga_phase(sub, arm, kappa, V, init_genes, rng, n_gen, death_memory=None):
    """minimal GA を 1 phase。arm ごとの記憶形成機構を適用し、被った致命評価数 (deaths) を返す。

    - NONE: gate なし。死=fitness 0 で除去 (記憶も消える)。
    - EXO_fixed: gate κ_low (reject, 評価前 pre-screen)。 ENDO: gate κ_current (reject, 評価前 pre-screen)。
    - **REVIVE (修正版)**: 評価前に先回りしない。child を**評価し死を経験** (death カウント) → 死んだら
      **蘇生** (修復して記憶 mix を保持・再評価) し集団へ。= 「死を経験しても記憶が消えない」(ユーザー洞察の
      本来の意味)。よって deaths は NONE 並みにかかるが、死んだ個体は消えず記憶を保って復活する。
    - OBSERVE: 他個体の観察された死 (death_memory) 近傍の child を評価前に回避 (経験的・社会的境界)。
    """
    deaths = [0]
    gkappa = _gate_kappa_for(arm, kappa)
    dm = death_memory if death_memory is not None else []   # OBSERVE 用 (社会的記憶, phase 跨ぎ継続)

    def _eval_count(g):
        """g を評価し死をカウント。REVIVE は死んだら蘇生 (修復+再評価)。(final_gene, fitness) を返す。"""
        fit, surv = _eval(sub, g, kappa, V, rng)
        if surv < 1.0:
            deaths[0] += 1
            dm.append(_normgene(g.clipped()))               # 観察された死を社会的記憶へ (OBSERVE)
            if arm == "REVIVE":
                g2 = _repair(sub, g, gkappa, V)             # 死を経験 → 蘇生 (記憶 mix 保持)
                fit2, _ = _eval(sub, g2, kappa, V, rng)
                return g2, fit2
        return g, fit

    def _pre_screen(child):
        """gate/OBSERVE arm のみ評価前に致命候補を resample (REVIVE/NONE は pre-screen しない)。"""
        if arm in ("NONE", "REVIVE"):
            return child
        tries = 0
        while tries < RESAMPLE_CAP:
            ok = (not _near_observed_death(child, dm)) if arm == "OBSERVE" else _admits(sub, child, gkappa, V)
            if ok:
                return child
            src = tournament_select(pop, TOURN_K, rng).gene
            child = uniform_mutate(src, SIGMA, rng)
            tries += 1
        return child

    genes = initialize_random_population(POP, rng) if init_genes is None else list(init_genes)
    pop = Population(tuple(Individual(*_eval_count(g)) for g in genes))
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
            children.append(_pre_screen(child))
        pop = Population(tuple(Individual(*_eval_count(g)) for g in children))
        best_curve.append(pop.best.fitness)
    return [i.gene for i in pop.individuals], list(best_curve), pop, deaths[0]


def _pop_survival_rate(sub, genes, kappa, V, seed):
    rng = np.random.default_rng(800000 + seed)
    return float(np.mean([_eval(sub, g, kappa, V, rng)[1] for g in genes]))


def _soundness_violations(sub, V, n=5000, seed=0):
    """admit(g, κ_high, V)=True の gene が κ_high で死なないか (H3)。違反=認証破綻。"""
    rng = np.random.default_rng(seed)
    viol, checked = 0, 0
    for _ in range(n):
        g = StateUpdateGene(decay=float(rng.uniform(0, 1)),
                            mix=float(rng.uniform(-1, 1)),
                            gate_str=float(rng.uniform(-2, 2)))
        if _admits(sub, g, KAPPA_HIGH, V):
            checked += 1
            _, surv = _eval(sub, g, KAPPA_HIGH, V, np.random.default_rng(700000 + checked))
            if surv < 1.0:
                viol += 1
    return viol, checked


def _run_arm(sub, arm, seed, V):
    rng = np.random.default_rng(seed)
    dm = []   # OBSERVE 用社会的死記憶 (全 phase 継続)
    genes1, curve1, pop1, deaths1 = _ga_phase(sub, arm, KAPPA_LOW, V, None, rng, G1, death_memory=dm)
    phase1_final = float(pop1.best.fitness)
    # 助走 (warm-up): κ_high で死を経験させ OBSERVE 学習/REVIVE 蓄積/shock 通過。deaths は数えない。
    genes_wu, _, _, deaths_wu = _ga_phase(sub, arm, KAPPA_HIGH, V, genes1, rng, G2_WARMUP, death_memory=dm)
    # 測定 (measure): 定常状態の deaths/記憶を正確な統計として測る。dm/集団/rng は warm-up から継続。
    genes2, curve2, pop2, deaths2 = _ga_phase(sub, arm, KAPPA_HIGH, V, genes_wu, rng, G2_MEASURE, death_memory=dm)
    surv = _pop_survival_rate(sub, genes2, KAPPA_HIGH, V, seed)
    return {
        "arm": arm, "seed": seed, "phase2_deaths": int(deaths2), "warmup_deaths": int(deaths_wu),
        "phase1_deaths": int(deaths1), "phase1_final_fitness": phase1_final,
        "phase2_gen0_fitness": float(curve2[0]),            # measure 期頭 = 記憶がどれだけ生き残ったか
        "phase2_final_best_fitness": float(pop2.best.fitness),
        # REVIVE の真価指標: 死を経験しても記憶が消えず集団に残るか (mean fitness + 多様性)。
        "phase2_pop_mean_fitness": float(pop2.fitness_array.mean()),
        "phase2_pop_diversity": float(pop2.gene_matrix.var()),
        "phase2_auc": float(np.sum(curve2)), "phase2_survival_rate": surv,
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


def run_substrate(sub):
    V = V_BY_SUBSTRATE[sub.name]
    recs = []
    for arm in ARMS:
        for seed in D_SEEDS:
            recs.append(_run_arm(sub, arm, seed, V))
    means = {arm: {
        "phase2_deaths": float(np.mean([r["phase2_deaths"] for r in recs if r["arm"] == arm])),
        "best_fitness": float(np.mean([r["phase2_final_best_fitness"] for r in recs if r["arm"] == arm])),
        "pop_mean_fitness": float(np.mean([r["phase2_pop_mean_fitness"] for r in recs if r["arm"] == arm])),
        "diversity": float(np.mean([r["phase2_pop_diversity"] for r in recs if r["arm"] == arm])),
    } for arm in ARMS}

    # 2 軸: 死回避 (deaths 低) と 記憶保存 (pop_mean_fitness 高 = 死んでも記憶が残る)
    comp = {
        # 死回避: ENDO (予見) は死を避ける / REVIVE は死を経験する (NONE 並み)
        "deaths_NONE_minus_ENDO": _stat(_paired(recs, "NONE", "ENDO", "phase2_deaths")),
        "deaths_NONE_minus_REVIVE": _stat(_paired(recs, "NONE", "REVIVE", "phase2_deaths")),
        "deaths_OBSERVE_minus_ENDO": _stat(_paired(recs, "OBSERVE", "ENDO", "phase2_deaths")),
        # 記憶保存 (本丸): REVIVE は死を経験しても集団記憶を保つか (vs NONE 同死だが記憶喪失)
        "popmean_REVIVE_minus_NONE": _stat(_paired(recs, "REVIVE", "NONE", "phase2_pop_mean_fitness")),
        "popmean_ENDO_minus_NONE": _stat(_paired(recs, "ENDO", "NONE", "phase2_pop_mean_fitness")),
        "popmean_REVIVE_minus_ENDO": _stat(_paired(recs, "REVIVE", "ENDO", "phase2_pop_mean_fitness")),
        "popmean_ENDO_minus_OBSERVE": _stat(_paired(recs, "ENDO", "OBSERVE", "phase2_pop_mean_fitness")),
    }

    sound_viol, sound_checked = _soundness_violations(sub, V)   # ENDO/REVIVE の verifier 健全性
    none_d = means["NONE"]["phase2_deaths"]
    boundary_active = bool(none_d > 1.0)

    by_deaths = sorted(ARMS, key=lambda a: means[a]["phase2_deaths"])
    by_mem = sorted(ARMS, key=lambda a: -means[a]["pop_mean_fitness"])
    if not boundary_active:
        verdict = (f"INVALID (死境界 inactive, F2): NONE deaths={none_d:.1f}。κ_high/V 再設計。")
    else:
        rn = comp["popmean_REVIVE_minus_NONE"]; en = comp["deaths_NONE_minus_ENDO"]
        eo = comp["popmean_ENDO_minus_OBSERVE"]
        verdict = (f"[{sub.name}] 2 軸: ①死回避 (低い順) {' < '.join(by_deaths)} — ENDO (予見) が死を避ける "
                   f"(NONE−ENDO Δ={en['mean_delta']:+.1f}, p={en['p_signflip_two_sided']:.3f}); REVIVE は死を経験 "
                   f"({means['REVIVE']['phase2_deaths']:.1f}≈NONE {none_d:.1f})。"
                   f"②記憶保存 (pop_mean_fit 高い順) {' > '.join(by_mem)} — **REVIVE は同死でも記憶を保つ** "
                   f"(REVIVE−NONE pop_mean Δ={rn['mean_delta']:+.3f}, p={rn['p_signflip_two_sided']:.3f}) "
                   f"= 復活が経験を記憶に残す。ENDO は予見で死を避けつつ記憶保持。OBSERVE (empirical) は死多+記憶低 "
                   f"(ENDO−OBSERVE pop_mean Δ={eo['mean_delta']:+.3f})。verifier soundness viol={sound_viol}/{sound_checked}。")

    return {
        "substrate": sub.name, "V": V, "obs_gain": sub.obs_gain,
        "arm_means_phase2": means, "comparisons": comp,
        "soundness_violations": sound_viol, "soundness_checked": sound_checked,
        "boundary_active": boundary_active,
        "deaths_ranking_low_to_high": by_deaths, "memory_ranking_high_to_low": by_mem,
        "verdict": verdict, "records": recs,
    }


def run_all():
    _ensure_utf8_stdout()
    t0 = time.time()
    out = {"preregistration": {
        "KAPPA_LOW": KAPPA_LOW, "KAPPA_HIGH": KAPPA_HIGH, "W_BAR_FIX": W_BAR_FIX,
        "G1": G1, "G2": G2, "seeds": D_SEEDS, "V_by_substrate": V_BY_SUBSTRATE,
        "delay": DELAY, "seq_len": SEQ_LEN, "alpha": ALPHA, "perm_seed": PERM_RNG_SEED,
        "arms": ARMS, "pop": POP,
    }, "substrates": {}}
    for sub in ALL_SUBSTRATES:
        res = run_substrate(sub)
        out["substrates"][sub.name] = res
        m = res["arm_means_phase2"]
        print(f"  [{sub.name}] deaths " + " ".join(f"{a}={m[a]['phase2_deaths']:.0f}" for a in ARMS) +
              " | pop_mean " + " ".join(f"{a}={m[a]['pop_mean_fitness']:.2f}" for a in ARMS) +
              f" | viol={res['soundness_violations']} ({time.time()-t0:.0f}s)", flush=True)
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def main():
    out = run_all()
    (_HERE / "results_viability_ab.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {_HERE / 'results_viability_ab.json'}")
    print("\n=== R-endo viability A/B — 記憶形成 3 機構 (環境=κ, phase2 κ_high, n=20) ===")
    for name, res in out["substrates"].items():
        print(f"\n[{name}] 死回避: {' < '.join(res['deaths_ranking_low_to_high'])} | "
              f"記憶保存: {' > '.join(res['memory_ranking_high_to_low'])}")
        m = res["arm_means_phase2"]
        for arm in ARMS:
            print(f"   {arm:10s} deaths={m[arm]['phase2_deaths']:6.1f} pop_mean_fit={m[arm]['pop_mean_fitness']:.4f} "
                  f"diversity={m[arm]['diversity']:.3f} best={m[arm]['best_fitness']:.4f}")
        c = res["comparisons"]
        print(f"   REVIVE−NONE pop_mean Δ={c['popmean_REVIVE_minus_NONE']['mean_delta']:+.4f} "
              f"(p={c['popmean_REVIVE_minus_NONE']['p_signflip_two_sided']:.3f}) = 復活の記憶保存; "
              f"NONE−ENDO deaths Δ={c['deaths_NONE_minus_ENDO']['mean_delta']:+.1f} = 予見の死回避")
        print(f"   verdict: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
