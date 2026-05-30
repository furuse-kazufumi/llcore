# SPDX-License-Identifier: Apache-2.0
"""distractor_recall_diag — distractor_recall 候補の③検定土俵診断.

E-A の C-gen1/C-gen2 + niche_signal を測る (measure-first / honest 規律)。

(1) 各 regime (distractor 振幅 a) の random-search held-out R² ceiling
    (N_RANDOM=120 本の max、n_seeds=5、LeakyDelayLineReservoir(n_taps=8) + make_eval_once)。
    selection-on-noise 水増し排除のため、各 seed の best random gene を honest_reevaluate で
    fresh-seed 再評価した値も併記する。difficulty_class を付ける。
(2) train/test split (extrapolation hold-out: 未学習 a) で MAP-E
    (map_elites_full, n_evals=200, 5 seeds) を train で進化 → honest_reevaluate で
    train/test 汎化と汎化ギャップ (train - test) を測る。
(3) niche_signal: 各 regime で「最良 random gene」の behavior descriptor (eff_mem_norm,
    leak_std) を集計し、regime 間で好まれる leak/eff_mem の散らばり (=niche 構造) を測る。
    散らばりが大きいほど regime ごとに異なる記憶戦略が最適 = behavior descriptor と整合し
    ③ (選択圧/分離) が効きうる。

結果を distractor_recall_diag_results.json に保存。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

_HERE = Path(__file__).resolve().parent          # research/ea_multitask/candidates
_EA = _HERE.parent                                # research/ea_multitask
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_EA))                       # ea_lab, task_mixture
sys.path.insert(0, str(_EA.parent / "step_c_memory_tasks"))  # reservoir
sys.path.insert(0, str(_EA.parents[1] / "src"))    # llcore/src (honest_eval)

import numpy as np  # noqa: E402

from distractor_recall import DistractorRecallTask  # noqa: E402
from ea_lab import map_elites_full  # noqa: E402
from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    gene_bounds,
    make_behavior,
    make_eval_once,
)
from task_mixture import TaskMixture, split_regimes  # noqa: E402

# --- config (軽量 budget) ---
N_TAPS = 8
IN_DIM = 1
SEQ_LEN = 30
AMPS = (0.2, 0.5, 0.8, 1.1)   # regime 軸 = distractor 振幅 a
N_RANDOM = 120
N_SEEDS = 5
N_EVALS = 200
HONEST_N = 12
SIGMA = 0.12
GRID = (6, 6)
# extrapolation hold-out: train={a=0.2,0.5}, test={a=0.8,1.1} (未学習の大振幅)
TEST_AMPS = (0.8, 1.1)


def _difficulty_class(mean_ceiling: float) -> str:
    """random-search 天井 (honest 再評価平均) から難易度クラスを判定."""
    if mean_ceiling < 0.5:
        return "floor"
    if mean_ceiling > 0.9:
        return "saturated"
    return "medium"


def _random_search_best(res, task, n_random: int, seed: int):
    """random search で max held-out R² を取り、その best gene を返す.

    Returns
    -------
    (best_r2_noisy, best_gene) : tuple[float, np.ndarray]
        best_r2_noisy は selection-on-noise を含む生の max。best_gene は後で honest 再評価する。
    """
    ev = make_eval_once(res, task, n_train=48, n_eval=48)
    rng = np.random.default_rng(seed)
    best_r2 = -np.inf
    best_gene = None
    for _ in range(n_random):
        g = res.random_gene(rng)
        r2 = ev(g, rng)
        if r2 > best_r2:
            best_r2 = r2
            best_gene = g
    return float(best_r2), best_gene


def main() -> None:
    print("=== distractor_recall_diag: ③検定土俵診断 ===")
    print(f"seq_len={SEQ_LEN} amps={AMPS} test_amps={TEST_AMPS} "
          f"N_RANDOM={N_RANDOM} n_seeds={N_SEEDS} n_evals={N_EVALS}\n", flush=True)

    res = LeakyDelayLineReservoir(n_taps=N_TAPS, in_dim=IN_DIM)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim

    # ---------- (1) regime 別 random-search 天井 + difficulty ----------
    print("--- (1) regime 別 random-search held-out R² ceiling ---", flush=True)
    per_regime = []
    # niche_signal 用: 各 regime の best random gene の behavior descriptor を貯める。
    regime_best_behaviors: dict[float, list[np.ndarray]] = {a: [] for a in AMPS}
    for a in AMPS:
        task = DistractorRecallTask(seq_len=SEQ_LEN, distractor_amp=a)
        noisy_maxes = []
        honest_bests = []
        for s in range(N_SEEDS):
            best_r2_noisy, best_gene = _random_search_best(res, task, N_RANDOM, 7000 + s)
            noisy_maxes.append(best_r2_noisy)
            # selection-on-noise 排除: best を fresh-seed で honest 再評価。
            ev = make_eval_once(res, task, n_train=48, n_eval=48)
            honest_bests.append(
                honest_reevaluate(ev, best_gene, n_trials=HONEST_N,
                                  rng=np.random.default_rng(8000 + s))
            )
            regime_best_behaviors[a].append(behavior(best_gene))
        noisy_arr = np.array(noisy_maxes)
        honest_arr = np.array(honest_bests)
        # difficulty は selection-on-noise を排した honest 平均で判定 (honest 規律)。
        mean_honest = float(honest_arr.mean())
        max_honest = float(honest_arr.max())
        solvable = bool(max_honest > 0.5)
        per_regime.append({
            "regime": f"a={a}",
            "amp": a,
            "noisy_max_mean": float(noisy_arr.mean()),
            "noisy_max_max": float(noisy_arr.max()),
            "honest_mean": mean_honest,
            "honest_max": max_honest,
            "solvable": solvable,
            "difficulty_class": _difficulty_class(mean_honest),
        })
        print(f"  a={a:<4}: noisy_max(mean)={noisy_arr.mean():.3f} "
              f"honest(mean)={mean_honest:.3f} honest(max)={max_honest:.3f} "
              f"solvable={solvable} [{_difficulty_class(mean_honest)}]", flush=True)

    # ---------- (3) niche_signal: regime 間の leak/eff_mem 選好の散らばり ----------
    # 各 regime で best random gene の behavior descriptor を平均し、regime 間の標準偏差を取る。
    # descriptor = (eff_mem_norm, leak_std)。eff_mem_norm は leak の小ささ (長期保持) を表す。
    print("\n--- (3) niche_signal: regime 間 best-gene 選好の散らばり ---", flush=True)
    per_regime_mean_behavior = {}
    eff_mem_means = []
    leak_std_means = []
    for a in AMPS:
        bh = np.array(regime_best_behaviors[a])  # (n_seeds, 2)
        m = bh.mean(axis=0)
        per_regime_mean_behavior[f"a={a}"] = {
            "eff_mem_norm": float(m[0]), "leak_std": float(m[1])}
        eff_mem_means.append(float(m[0]))
        leak_std_means.append(float(m[1]))
        print(f"  a={a:<4}: best-gene eff_mem_norm={m[0]:.3f} leak_std={m[1]:.3f}",
              flush=True)
    # regime 間の散らばり: 2 descriptor 軸それぞれの across-regime 標準偏差。
    eff_mem_spread = float(np.std(eff_mem_means))
    leak_std_spread = float(np.std(leak_std_means))
    # leak 選好の散らばり (descriptor 全体のユークリッド広がり) を 1 数値に集約。
    leak_spread_across_regimes = float(
        np.sqrt(eff_mem_spread ** 2 + leak_std_spread ** 2))
    print(f"  -> eff_mem_spread={eff_mem_spread:.4f} leak_std_spread={leak_std_spread:.4f} "
          f"leak_spread_across_regimes={leak_spread_across_regimes:.4f}", flush=True)

    # ---------- (2) train/test split で MAP-E 進化 → 汎化ギャップ ----------
    print("\n--- (2) MAP-E train 進化 → hold-out test 汎化ギャップ ---", flush=True)
    regimes = [DistractorRecallTask(seq_len=SEQ_LEN, distractor_amp=a) for a in AMPS]
    test_idx = [i for i, a in enumerate(AMPS) if a in TEST_AMPS]
    train_regimes, test_regimes = split_regimes(regimes, test_idx=test_idx)
    ev_tr = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    ev_te = make_eval_once(res, TaskMixture(test_regimes), n_train=48, n_eval=48)

    train_scores, test_scores = [], []
    for s in range(N_SEEDS):
        r = map_elites_full(
            ev_tr, behavior, dim=dim, bounds=bounds,
            behavior_bounds=(np.zeros(2), np.ones(2)), grid_shape=GRID,
            n_evals=N_EVALS, init_batch=max(20, N_EVALS // 10),
            sigma=SIGMA, rng=np.random.default_rng(1000 + s))
        tr = honest_reevaluate(ev_tr, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(2000 + s))
        te = honest_reevaluate(ev_te, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(3000 + s))
        train_scores.append(tr); test_scores.append(te)
        print(f"  seed {s}: train R²={tr:.3f}  test(hold-out) R²={te:.3f}  "
              f"gap={tr - te:+.3f}", flush=True)

    train_arr, test_arr = np.array(train_scores), np.array(test_scores)
    train_mean = float(train_arr.mean())
    test_mean = float(test_arr.mean())
    gap = train_mean - test_mean
    overall_difficulty = _difficulty_class(test_mean)
    train_amps = tuple(a for a in AMPS if a not in TEST_AMPS)

    out = {
        "experiment": "distractor_recall diagnostic (C-gen1 ceiling + C-gen2 gap + niche)",
        "config": {"n_taps": N_TAPS, "in_dim": IN_DIM, "seq_len": SEQ_LEN,
                   "amps": list(AMPS), "test_amps": list(TEST_AMPS),
                   "n_random": N_RANDOM, "n_seeds": N_SEEDS, "n_evals": N_EVALS,
                   "honest_n": HONEST_N, "grid": list(GRID)},
        "per_regime_ceiling": per_regime,
        "generalization": {
            "split": f"train a{{{TRAIN_a}}} / test a{{{TEST_AMPS}}}, seq_len={SEQ_LEN}".replace(
                "TRAIN_a", str(tuple(a for a in AMPS if a not in TEST_AMPS))),
            "train_mean": train_mean, "test_mean": test_mean,
            "gap": gap, "difficulty_class": overall_difficulty,
            "train_per_seed": train_arr.tolist(), "test_per_seed": test_arr.tolist(),
        },
        "niche_signal": {
            "per_regime_mean_behavior": per_regime_mean_behavior,
            "eff_mem_spread": eff_mem_spread,
            "leak_std_spread": leak_std_spread,
            "leak_spread_across_regimes": leak_spread_across_regimes,
        },
    }
    (_HERE / "distractor_recall_diag_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 診断まとめ ===")
    n_solv = sum(1 for v in per_regime if v["solvable"])
    print(f"  (1) C-gen1: {n_solv}/{len(per_regime)} regime が solvable (honest max R²>0.5)")
    print(f"  (2) C-gen2: train_mean={train_mean:.3f} test_mean={test_mean:.3f} "
          f"gap={gap:+.3f}  difficulty={overall_difficulty}")
    print(f"  (3) niche_signal: leak_spread_across_regimes={leak_spread_across_regimes:.4f}")
    viable = bool(overall_difficulty == "medium" and gap > 0
                  and leak_spread_across_regimes > 0.02)
    if viable:
        print("  → medium ∧ 正ギャップ ∧ niche 構造あり = ③検定に viable。")
    elif test_mean > 0.9:
        print("  → test 飽和 = too-easy (③検定 不適、honest negative)。")
    elif test_mean < 0.5:
        print("  → test 床 = too-hard/基質床 (③以前のボトルネック)。")
    else:
        print("  → 部分的 (medium だがギャップ/niche のいずれか不足)。要検討。")


if __name__ == "__main__":
    main()
