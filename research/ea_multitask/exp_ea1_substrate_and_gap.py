# SPDX-License-Identifier: Apache-2.0
"""exp_ea1 — C-gen1 (基質が解ける土俵か) + C-gen2 (汎化ギャップ/難易度) の診断.

E_A_DESIGN のゲート C-gen1/C-gen2 を測る。目的は「③ を検定するに値する土俵か」を
**measure-first** で確かめること (baseline-first / honest 規律)。

統合 smoke で FlipFlop(pulse_prob mix, seq_len=30) はランダム gene でも R²>0.5 と判明 =
易しすぎると③が効かない (Step C flip_flop too-easy 罠と同型)。そこで FlipFlop の難易度を
**regime 軸 (pulse_prob × seq_len)** で走査し、

- **C-gen1**: 各 regime を基質 (単層 leaky reservoir + 線形 ridge) が random search で解けるか
  (held-out max R² > 0.5 = 床でない)。
- **C-gen2**: train regimes で進化 (短予算 MAP-E) → test(hold-out) regimes で honest 再評価し、
  **汎化ギャップ (train R² − test R²)** と難易度 (絶対水準) を測る。

ギャップ大 or 難易度中庸 (0.5–0.95) なら③検定の土俵になりうる。
飽和 (≈1.0) or 床 (<0.5) なら「③不要 (honest negative)」を示唆。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))

import numpy as np  # noqa: E402

from ea_lab import map_elites_full  # noqa: E402
from memory_tasks import FlipFlopTask  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    gene_bounds,
    make_behavior,
    make_eval_once,
)
from task_mixture import TaskMixture, split_regimes

sys.path.insert(0, str(_HERE.parents[1] / "src"))
from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402

N_TAPS = 8
IN_DIM = 2  # FlipFlop
N_RANDOM = 200
N_SEEDS = 6           # 診断なので軽量
N_EVALS = 200
HONEST_N = 12
SIGMA = 0.12
GRID = (6, 6)


def _random_ceiling(res, task, n_random: int, seed: int) -> float:
    """random search で引いた max held-out R² (基質の到達天井, C-gen1)."""
    ev = make_eval_once(res, task, n_train=48, n_eval=48)
    rng = np.random.default_rng(seed)
    return max(ev(res.random_gene(rng), rng) for _ in range(n_random))


def main() -> None:
    print("=== exp_ea1: C-gen1 (基質床) + C-gen2 (汎化ギャップ) 診断 ===")
    res = LeakyDelayLineReservoir(n_taps=N_TAPS, in_dim=IN_DIM)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim

    # --- C-gen1: regime 走査 (pulse_prob × seq_len) の random-search 天井 ---
    print("\n--- C-gen1: regime 別 random-search 天井 (held-out max R²) ---", flush=True)
    pulse_probs = [0.1, 0.2, 0.3, 0.4]
    seq_lens = [30, 60]
    c_gen1: dict[str, dict] = {}
    for sl in seq_lens:
        for pp in pulse_probs:
            task = FlipFlopTask(seq_len=sl, pulse_prob=pp)
            vals = np.array([_random_ceiling(res, task, N_RANDOM, s) for s in range(N_SEEDS)])
            key = f"seq{sl}_pp{pp}"
            solvable = bool(vals.max() > 0.5)
            c_gen1[key] = {"mean": float(vals.mean()), "max": float(vals.max()),
                           "std": float(vals.std()), "solvable": solvable}
            print(f"  {key:14s}: max={vals.max():.3f} mean={vals.mean():.3f} "
                  f"solvable={solvable}", flush=True)

    # --- C-gen2: 汎化ギャップ (train で進化 → test で honest) ---
    # 難易度を上げるため長め seq + sparse pulse 寄りの 4 regime を分布化。
    print("\n--- C-gen2: 汎化ギャップ (MAP-E train 進化 → hold-out test) ---", flush=True)
    regimes = [FlipFlopTask(seq_len=60, pulse_prob=pp) for pp in (0.1, 0.2, 0.3, 0.4)]
    # extrapolation hold-out: train={pp 0.1,0.2}, test={pp 0.3,0.4}
    train_regimes, test_regimes = split_regimes(regimes, test_idx=[2, 3])
    mix_tr, mix_te = TaskMixture(train_regimes), TaskMixture(test_regimes)
    ev_tr = make_eval_once(res, mix_tr, n_train=48, n_eval=48)
    ev_te = make_eval_once(res, mix_te, n_train=48, n_eval=48)

    train_scores, test_scores = [], []
    for s in range(N_SEEDS):
        r = map_elites_full(
            ev_tr, behavior, dim=dim, bounds=bounds, behavior_bounds=(np.zeros(2), np.ones(2)),
            grid_shape=GRID, n_evals=N_EVALS, init_batch=max(20, N_EVALS // 10),
            sigma=SIGMA, rng=np.random.default_rng(1000 + s))
        tr = honest_reevaluate(ev_tr, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(2000 + s))
        te = honest_reevaluate(ev_te, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(3000 + s))
        train_scores.append(tr); test_scores.append(te)
        print(f"  seed {s}: train R²={tr:.3f}  test(hold-out) R²={te:.3f}  gap={tr - te:+.3f}",
              flush=True)

    train_arr, test_arr = np.array(train_scores), np.array(test_scores)
    gap = float(train_arr.mean() - test_arr.mean())

    out = {
        "experiment": "exp_ea1 substrate floor (C-gen1) + generalization gap (C-gen2)",
        "config": {"n_taps": N_TAPS, "in_dim": IN_DIM, "n_random": N_RANDOM,
                   "n_seeds": N_SEEDS, "n_evals": N_EVALS, "honest_n": HONEST_N},
        "c_gen1_random_ceiling": c_gen1,
        "c_gen2_generalization": {
            "split": "train pp{0.1,0.2} / test pp{0.3,0.4}, seq_len=60",
            "train_mean": float(train_arr.mean()), "test_mean": float(test_arr.mean()),
            "generalization_gap": gap,
            "train_per_seed": train_arr.tolist(), "test_per_seed": test_arr.tolist(),
        },
    }
    (_HERE / "exp_ea1_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 診断まとめ ===")
    n_solv = sum(1 for v in c_gen1.values() if v["solvable"])
    print(f"  C-gen1: {n_solv}/{len(c_gen1)} regime が solvable (max R²>0.5)")
    print(f"  C-gen2: train_mean={train_arr.mean():.3f} test_mean={test_arr.mean():.3f} "
          f"gap={gap:+.3f}")
    if test_arr.mean() > 0.95:
        print("  → test が飽和 (>0.95) = 易しすぎ。③検定には too-easy 懸念 (要 regime 強化)。")
    elif test_arr.mean() < 0.5:
        print("  → test が床 (<0.5) = 難しすぎ/基質床。③以前のボトルネック。")
    else:
        print("  → test が中庸 (0.5–0.95) = ③検定の土俵になりうる。exp_ea3 ablation へ。")


if __name__ == "__main__":
    main()
