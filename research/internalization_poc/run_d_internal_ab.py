# SPDX-License-Identifier: Apache-2.0
"""R-endo PoC — 内的検証器 (entity が自分の sound verifier を持つ) の A/B 決着 run.

## 背景: ユーザー提案「検証器を llcore 自身が持ってはどうか」

現状の trajectory_tube gate は外部 GA が admit/reject に適用する **exogenous selection**。
ユーザー提案 = entity 自身が検証器を持ち「自分が現在の生存環境で contraction-certified か」を
自己判断すれば、より短いサイクルで自律適応でき、より多くの外的要因で多様な進化の土壌ができるのでは。

## honest な設計前提 (内的化評価 Workflow の結論)

- **概念は Gödel Machine (Schmidhuber 2003) 既出** (proof-gated self-modification)。新規性は薄い。
- **判定主体を GA→gene に移す location shift 自体は安全性も挙動も上げない**。安全性は「何を証明
  するか (dynamics)」由来。意味ある差は entity が **環境 (現在の外乱 w̄) に結合して自己検証**し、
  環境変化に即応する場合にのみ生まれる = 内的化 ≡ 環境結合適応 gating。
- entity の self-verdict (`StateUpdateGene.is_verified_trajectory_tube`) は外部 gate
  (`minimal_ga._gate_admits(..., "trajectory_tube")`) と **構成的に同一** (H3: random 5000 gene で
  0 disagreement 確認済)。よって本 runner は ENDO arm を `evolve(gate_mode="trajectory_tube",
  w_bar=現在の w_env)` で実装し、「これは entity 自己検証と挙動同一 (H3 等価)」と honest に framing する。

## 実験: 環境 (task 外乱 w_env) を mid-run でステップ変化させ、再適応を測る

DisturbedCopyTask (delay=8, seq_len=32) を fitness 基質に使い、w_env を 2 段で変える:
- phase 1 (G1 世代): w_env = W_LOW。
- phase 2 (G2 世代): w_env = W_HIGH (環境悪化ステップ)。`evolve` の initial_pop で集団継続 + 同一 rng。

arms (evolve 中の gate 挙動):
- **NONE**:      gate_mode="none" 両 phase (gate なし、pure selection の anchor)。
- **EXO_fixed**: gate w̄ = W_LOW 固定 両 phase (外部 gate は設計時 w̄ に固定、環境変化に盲目)。
- **ENDO**:      gate w̄ = 現在の w_env (entity が現環境に結合して自己再 gate)。phase2 で W_HIGH に再 gate。

phase 1 では ENDO と EXO_fixed は同一 w̄ (=W_LOW) なので **挙動一致** (同一 rng で byte-identical)。
両 arm は phase 2 (環境ステップ後) でのみ分岐する = 効果を環境変化に isolate する設計。

## honest 予測 (高 null 事前確率)

w̄ dose-response 実験 (VERDICT §11) は「外乱負荷 ↑ で tube gate の fitness 価値は増えない (NEGATIVE)」
を示した: memory タスクでは外乱除去 (tight contraction) と保持 (retention) がトレードオフするため。
よって ENDO (環境結合で W_HIGH に tight 再 gate) は保持を犠牲にしうる → 再適応を助けない / むしろ害す
事前確率が高い。さらに本基質 (CopyTask) は convex-combination で有界 = 環境悪化が viability を脅かさない
(発散しない) ため、内的 gate が防ぐべき破綻が無い。**この PoC は「内的化がこの基質で autonomy 上の
優位を生むか」を honest に測る** — null でも「内的化が効くのは環境変化が viability を脅かす (発散しうる)
基質に限る」という方向を指す有効な決着になる (feedback_benchmark_honest_disclosure)。

## 事前登録 (PRE-REGISTRATION — 結果取得前に本ファイルを commit すること)

- **H1 (confirmatory, non-inferiority)**: phase 2 終了時の held-out test fitness (W_HIGH task) で
  ENDO が EXO_fixed に対し非劣 — paired delta (ENDO − EXO_fixed) の下側を sign-flip permutation で
  評価し、mean delta が non-inferiority margin −δ を上回る。δ = Phase 2a |mean Δ| (≈0.0134) の半分
  = 0.0067。両側 sign-flip permutation (n_resamples=100,000, rng seed=11), α=0.05。
- **H2 (confirmatory, autonomy 本丸)**: phase 2 の train best_fitness_curve の AUC (= 再適応の速さ・
  面積) で ENDO > EXO_fixed (paired, 同検定)。これが「環境結合自己検証が短サイクル適応を生む」の唯一の
  confirmatory autonomy 指標。
- **H3 (safety, 別途確認済)**: entity self-verdict == 外部 gate verdict。random 5000 gene で 0
  disagreement を確認済 (本 runner 冒頭でも再 assert)。1 件でも出れば内的判定 unsound = 致命。
- **exploratory (報告のみ)**: NONE arm の同指標 / diversity (final pop gene-matrix var) ENDO vs EXO /
  phase 2 の recovery 世代数。
- **判定**:
  - H1 非劣 ∧ H2 p<0.05 で ENDO>EXO ∧ H3=0 → **HIGH_VALUE 昇格** (内的化に autonomy 優位)。
  - H1 非劣 ∧ H2 p≥0.05 ∧ H3=0 → **CONDITIONAL 据置** (feasible だが本基質で autonomy benefit なし
    = location shift にすぎず)。advisory-only で park、外部 gate canonical 維持。
  - H1 劣位 or H3≥1 → **LOW_VALUE park** (fitness tax / self-trust unsound)。external-only に revert。
- **反証条件 (結果前に固定)**: (F1) H2 null (再適応 ENDO≈EXO) → 短サイクル autonomy は幻想。
  (F2) H1 劣位 → inline filter が evolvability を害す。 (F3) diversity ENDO<EXO×0.9 → 探索を狭める。
  (F4) NONE が両 gated arm と同等以上 → gate 自体が本基質で無効。
- **seeds**: 3000..3019 (n=20、pilot/run_c/run_wbar と独立に再利用)。
- **GA/gate パラメータ**: run_3arm_ab.py から GA_KW / R_MAX / STATE_DIM / W_BAR を import (構成同一)。

実行::

    py -3.11 research/internalization_poc/run_d_internal_ab.py

出力::

    research/internalization_poc/results_d_internal_ab.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_VMP = _HERE.parents[0] / "verified_memory_poc"   # run_3arm_ab / run_wbar / run_c がある場所
for _p in (str(_HERE.parents[1] / "src"), str(_VMP), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llcore.evolution import evolve  # noqa: E402
from llcore.fitness import calibrate_baseline, evaluate_gene  # noqa: E402
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.tracking_tube import tracking_tube  # noqa: E402

from run_3arm_ab import (  # noqa: E402  (Phase 2a と同一構成を保証)
    GA_KW,
    R_MAX,
    STATE_DIM,
    TEST_N_TRIALS,
    TRAIN_N_TRIALS,
    W_BAR,
    _READOUT,
    _ensure_utf8_stdout,
)
from run_wbar_decision import DisturbedCopyTask  # noqa: E402
from run_c_decision import signflip_pvalue  # noqa: E402

# ---- 事前登録パラメータ ------------------------------------------------------
W_LOW = 0.05                  # phase 1 環境外乱
W_HIGH = 0.20                 # phase 2 環境外乱 (悪化ステップ)
G1 = 10                       # phase 1 世代数
G2 = 10                       # phase 2 世代数 (total = Phase 2a の 20)
FIXED_DELAY = 8
FIXED_SEQ_LEN = 32
D_SEEDS = list(range(3000, 3020))   # n=20
ARMS = ["NONE", "EXO_fixed", "ENDO"]
NONINF_DELTA = 0.0067         # H1 non-inferiority margin (= Phase 2a |Δ|/2)
PERM_N_RESAMPLES = 100_000
PERM_RNG_SEED = 11
ALPHA = 0.05


def _build_task(w_env: float) -> DisturbedCopyTask:
    t = DisturbedCopyTask(w_task=w_env, seq_len=FIXED_SEQ_LEN, delay=FIXED_DELAY,
                          state_dim=STATE_DIM, out_dim=STATE_DIM)
    b = calibrate_baseline(t, _READOUT)
    return replace(t, baseline_mse=float(b))


def _ff_for(task):
    def _ff(gene: StateUpdateGene, rng: np.random.Generator) -> float:
        return evaluate_gene(gene, task, _READOUT, rng, n_trials=TRAIN_N_TRIALS)
    return _ff


def _test_fitness(gene: StateUpdateGene, task, seed: int) -> float:
    rng = np.random.default_rng(900000 + seed)
    return evaluate_gene(gene, task, _READOUT, rng, n_trials=TEST_N_TRIALS)


def _gate_kw(arm: str, w_env: float) -> dict:
    """arm + 現在の環境 w_env から evolve の gate 引数を作る."""
    if arm == "NONE":
        return dict(gate_mode="none")
    if arm == "EXO_fixed":
        return dict(gate_mode="trajectory_tube", w_bar=W_LOW, r_max=R_MAX)   # 設計時固定
    if arm == "ENDO":
        return dict(gate_mode="trajectory_tube", w_bar=w_env, r_max=R_MAX)   # 環境結合
    raise ValueError(arm)


def _evolve_two_phase(arm: str, seed: int):
    """phase1 (W_LOW) → phase2 (W_HIGH) を集団継続 + 同一 rng で走らせる."""
    rng = np.random.default_rng(seed)
    task1 = _build_task(W_LOW)
    task2 = _build_task(W_HIGH)

    base_kw = {k: v for k, v in GA_KW.items() if k != "n_generations"}  # n_gen は phase で上書き
    # phase 1
    res1 = evolve(_ff_for(task1), rng=rng, n_generations=G1,
                  **base_kw, **_gate_kw(arm, W_LOW))
    pop1 = [ind.gene for ind in res1.generations[-1].individuals]
    # phase 2 (環境ステップ; initial_pop で継続, rng は phase1 から継続)
    res2 = evolve(_ff_for(task2), rng=rng, n_generations=G2, initial_pop=pop1,
                  **base_kw, **_gate_kw(arm, W_HIGH))

    best2 = res2.final_best.gene
    return {
        "arm": arm, "seed": seed,
        "phase1_best_curve": list(res1.best_fitness_curve),
        "phase2_best_curve": list(res2.best_fitness_curve),
        "phase2_auc": float(np.sum(res2.best_fitness_curve)),       # H2: 再適応の面積 (世代和)
        "phase2_final_train": float(res2.final_best.fitness),
        "phase2_final_test_whigh": _test_fitness(best2, task2, seed),  # H1: 最終適応品質
        "phase2_final_diversity": float(res2.generations[-1].gene_matrix.var()),
        "best2_gene": [best2.decay, best2.mix, best2.gate_str],
        "gate_rej_p2": (res2.gate_stats.n_rejections if res2.gate_stats else 0),
        "gate_fallback_p2": (res2.gate_stats.fallback_count if res2.gate_stats else 0),
        "gate_children_p2": (res2.gate_stats.n_children_generated if res2.gate_stats else 0),
    }


def _h3_correctness_assert() -> int:
    """H3: entity self-verdict == 外部 gate verdict (random 5000 gene)。0 disagreement 必須."""
    rng = np.random.default_rng(0)
    dis = 0
    for _ in range(5000):
        g = StateUpdateGene(decay=float(rng.uniform(0, 1)),
                            mix=float(rng.uniform(-1, 1)),
                            gate_str=float(rng.uniform(-2, 2)))
        if g.is_verified_trajectory_tube(W_BAR, R_MAX) != bool(
                tracking_tube(g, w_bar=W_BAR, r_max=R_MAX).admits):
            dis += 1
    return dis


def _paired(stats_recs, arm_a, arm_b, key):
    a = np.array([r[key] for r in stats_recs if r["arm"] == arm_a])
    b = np.array([r[key] for r in stats_recs if r["arm"] == arm_b])
    # seed 順は D_SEEDS 固定なので index 対応で paired
    return a - b


def run_all() -> dict:
    _ensure_utf8_stdout()
    t0 = time.time()

    h3_dis = _h3_correctness_assert()
    print(f"  [H3 correctness] entity self-verdict vs external gate disagreements = {h3_dis}/5000",
          flush=True)

    recs = []
    for arm in ARMS:
        for seed in D_SEEDS:
            recs.append(_evolve_two_phase(arm, seed))
        sub = [r for r in recs if r["arm"] == arm]
        print(f"  [{arm}] mean phase2 final test(W_HIGH)="
              f"{np.mean([r['phase2_final_test_whigh'] for r in sub]):.4f}  "
              f"mean AUC={np.mean([r['phase2_auc'] for r in sub]):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- H1 / H2 判定 (ENDO − EXO_fixed, paired) -----------------------------
    def _stat(deltas):
        p = signflip_pvalue(deltas, n_resamples=PERM_N_RESAMPLES, seed=PERM_RNG_SEED)
        return {
            "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
            "n_positive": int((deltas > 0).sum()), "n_negative": int((deltas < 0).sum()),
            "p_signflip_two_sided": p,
        }

    h1_delta = _paired(recs, "ENDO", "EXO_fixed", "phase2_final_test_whigh")
    h2_delta = _paired(recs, "ENDO", "EXO_fixed", "phase2_auc")
    div_delta = _paired(recs, "ENDO", "EXO_fixed", "phase2_final_diversity")

    h1 = _stat(h1_delta)
    h2 = _stat(h2_delta)
    h1_noninferior = bool(h1["mean_delta"] > -NONINF_DELTA)
    h2_endo_faster = bool(h2["p_signflip_two_sided"] < ALPHA and h2["mean_delta"] > 0)

    # NONE control (gate 自体の有効性 F4)
    none_vs_exo = _paired(recs, "NONE", "EXO_fixed", "phase2_final_test_whigh")

    if h3_dis > 0:
        verdict = (f"LOW_VALUE park (致命): H3 self-trust unsound — entity self-verdict が外部 gate と "
                   f"{h3_dis} 件乖離。内的判定を信頼できない。external-only に revert。")
    elif not h1_noninferior:
        verdict = (f"LOW_VALUE park: H1 劣位 — ENDO が EXO_fixed に fitness tax "
                   f"(mean Δ={h1['mean_delta']:+.4f} < −δ={-NONINF_DELTA})。環境結合 tight 再 gate が "
                   f"保持を害す (w̄ NEGATIVE と整合)。external gate canonical 維持。")
    elif h2_endo_faster:
        verdict = (f"HIGH_VALUE 昇格: 内的化に autonomy 優位 — ENDO の phase2 再適応 AUC が EXO_fixed 超 "
                   f"(mean Δ={h2['mean_delta']:+.4f}, p={h2['p_signflip_two_sided']:.4f}) ∧ H1 非劣 ∧ H3=0。"
                   f"Phase 4 系 sub-phase へ格上げ可。")
    else:
        verdict = (f"CONDITIONAL 据置 (advisory-only park): 内的化は feasible・sound (H3=0)・非劣 (H1) "
                   f"だが本基質で autonomy benefit なし — phase2 再適応 AUC は ENDO≈EXO_fixed "
                   f"(Δ={h2['mean_delta']:+.4f}, p={h2['p_signflip_two_sided']:.4f})。location shift に"
                   f"すぎず。外部 gate を canonical に維持。内的化が効くのは環境変化が viability を脅かす"
                   f"(発散しうる) 基質に限る、という方向を指す negative。")

    return {
        "preregistration": {
            "W_LOW": W_LOW, "W_HIGH": W_HIGH, "G1": G1, "G2": G2,
            "delay": FIXED_DELAY, "seq_len": FIXED_SEQ_LEN, "r_max": R_MAX,
            "noninf_delta": NONINF_DELTA, "alpha": ALPHA,
            "test": "sign-flip permutation, two-sided", "n_resamples": PERM_N_RESAMPLES,
            "perm_rng_seed": PERM_RNG_SEED, "seeds": D_SEEDS, "arms": ARMS,
            "inherits_config_from": "run_3arm_ab.py (GA_KW/R_MAX/STATE_DIM/W_BAR/readout)",
            "endo_equiv_note": "ENDO arm = evolve(gate w_bar=現w_env)。entity.is_verified_trajectory_tube と "
                               "verdict 構成的一致 (H3)。内的化 ≡ 環境結合適応 gating。",
        },
        "h3_correctness_disagreements": h3_dis,
        "records": recs,
        "hypotheses": {
            "H1_noninferiority_endo_vs_exo": {**h1, "noninf_margin": NONINF_DELTA,
                                              "non_inferior": h1_noninferior},
            "H2_readaptation_auc_endo_vs_exo": {**h2, "endo_faster": h2_endo_faster},
            "diversity_endo_vs_exo": _stat(div_delta),
            "none_vs_exo_final_test": _stat(none_vs_exo),
        },
        "verdict": verdict,
        "wall_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    out = run_all()
    out_path = _HERE / "results_d_internal_ab.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    h1 = out["hypotheses"]["H1_noninferiority_endo_vs_exo"]
    h2 = out["hypotheses"]["H2_readaptation_auc_endo_vs_exo"]
    print(f"\n=== R-endo A/B (ENDO − EXO_fixed, paired n=20) ===")
    print(f"H1 final test(W_HIGH) Δ={h1['mean_delta']:+.4f} p={h1['p_signflip_two_sided']:.4f} "
          f"+{h1['n_positive']}/-{h1['n_negative']}  non_inferior={h1['non_inferior']}")
    print(f"H2 phase2 AUC       Δ={h2['mean_delta']:+.4f} p={h2['p_signflip_two_sided']:.4f} "
          f"+{h2['n_positive']}/-{h2['n_negative']}  endo_faster={h2['endo_faster']}")
    print(f"H3 disagreements={out['h3_correctness_disagreements']}/5000")
    print(f"\nverdict: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
