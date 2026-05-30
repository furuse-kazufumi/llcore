# SPDX-License-Identifier: Apache-2.0
"""exp_ea3 — ③ablation 本実験 (C-gen3 / C-gen4).

E_A_DESIGN の核心。多タスク分布 (FlipFlop regime) で 4 method を equal budget で進化し、
**hold-out (未学習) regime への汎化 R²** を主指標に strict_compare で③(選択圧/分離)の
load-bearing を判定する。

- C-gen3: MAP-E (①②③) > MAP-E_randselect (②③殺し①のみ) — 最もクリーンな selection 対照。
- C-gen4: MAP-E > panmictic-GA かつ > random。
③ load-bearing = C-gen3 ∧ C-gen4 を strict gate (n_seeds≥15, 片側 Wilcoxon p<0.05,
|paired_sign_delta|≥0.147) で通過。負ければ「滑らか/③不要」を honest negative とする。

regime config は exp_ea1 診断の結果で調整 (test が中庸 0.5–0.95 の regime/split を採用)。
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "candidates"))                     # research/ea_multitask/candidates
sys.path.insert(0, str(_HERE.parents[0] / "step_c_memory_tasks"))  # research/step_c_memory_tasks
sys.path.insert(0, str(_HERE.parents[0] / "step4_selection"))      # research/step4_selection
sys.path.insert(0, str(_HERE.parents[1] / "src"))                  # llcore/src

import numpy as np  # noqa: E402

from ea_lab import run_ea_methods_over_seeds  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    gene_bounds,
    make_behavior,
    make_eval_once,
)
from strict_compare import strict_compare  # noqa: E402
from task_mixture import TaskMixture, split_regimes  # noqa: E402
from variable_delay_recall import VariableDelayRecallTask  # noqa: E402

# --- config (exp_ea1 診断 + ③検定土俵探索 workflow で確定) ---
# 勝者分布 = variable_delay_recall (medium 難易度 + 正の汎化ギャップ + niche 構造、敵対検証 trustworthy)。
# FlipFlop は too-easy (全 regime ≈0.95 飽和, 汎化ギャップ負) のため不採用。
N_TAPS = 8
IN_DIM = 2
DISTRACTOR_AMP = 0.2      # 遅延区間ノイズ (診断で medium 難易度に着地した値)
TRAIN_D = (15, 30)        # 学習 regime (遅延長 D = seq_len)
TEST_D = (45, 60)         # hold-out regime (より長い遅延への extrapolation = 時定数外挿)
N_SEEDS = 15              # strict gate 要件
N_EVALS = 400
HONEST_N = 16
SIGMA = 0.12
GRID = (6, 6)


def main() -> None:
    print("=== exp_ea3: ③ablation 本実験 (C-gen3/C-gen4) ===")
    print(f"variable_delay_recall amp={DISTRACTOR_AMP}  train_D={TRAIN_D} test_D={TEST_D}  "
          f"n_seeds={N_SEEDS} n_evals={N_EVALS}\n", flush=True)

    res = LeakyDelayLineReservoir(n_taps=N_TAPS, in_dim=IN_DIM)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim

    all_D = tuple(sorted(set(TRAIN_D) | set(TEST_D)))
    regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=DISTRACTOR_AMP, in_dim=IN_DIM)
               for D in all_D]
    test_idx = [i for i, D in enumerate(all_D) if D in TEST_D]
    train_regimes, test_regimes = split_regimes(regimes, test_idx=test_idx)
    ev_tr = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    ev_te = make_eval_once(res, TaskMixture(test_regimes), n_train=48, n_eval=48)

    scores = run_ea_methods_over_seeds(
        ev_tr, ev_te, behavior, dim=dim, bounds=bounds,
        behavior_bounds=(np.zeros(2), np.ones(2)), grid_shape=GRID,
        n_evals=N_EVALS, n_seeds=N_SEEDS, honest_n_trials=HONEST_N, sigma=SIGMA)

    print("=== method 別 hold-out test 汎化 R² (mean ± std) + train ===")
    for m, sc in scores.items():
        print(f"  {m:24s}: test={sc.test.mean():.4f}±{sc.test.std():.3f}  "
              f"train={sc.train.mean():.4f}±{sc.train.std():.3f}  "
              f"gap={sc.train.mean() - sc.test.mean():+.4f}", flush=True)

    me = scores["map_elites"].test
    cmps = {
        "C-gen3_MAPE_vs_randselect": strict_compare(
            me, scores["map_elites_randselect"].test, "MAP-E", "randselect"),
        "C-gen4a_MAPE_vs_panmictic": strict_compare(
            me, scores["panmictic_ga"].test, "MAP-E", "panmictic"),
        "C-gen4b_MAPE_vs_random": strict_compare(
            me, scores["random"].test, "MAP-E", "random"),
    }

    print("\n=== strict_compare (hold-out test、片側 p<0.05 ∧ |δ|≥0.147 ∧ n≥15) ===")
    for key, c in cmps.items():
        print(f"  {key:28s}: diff={c.diff:+.4f} p={c.wilcoxon_p:.4g} "
              f"δ={c.paired_sign_delta:+.2f} passes={c.passes}", flush=True)

    c3 = cmps["C-gen3_MAPE_vs_randselect"].passes
    c4 = (cmps["C-gen4a_MAPE_vs_panmictic"].passes
          and cmps["C-gen4b_MAPE_vs_random"].passes)
    third_load_bearing = bool(c3 and c4)

    out = {
        "experiment": "exp_ea3 third-factor ablation (multitask hold-out generalization)",
        "config": {"n_taps": N_TAPS, "seq_len": SEQ_LEN, "train_pp": list(TRAIN_PP),
                   "test_pp": list(TEST_PP), "n_seeds": N_SEEDS, "n_evals": N_EVALS,
                   "honest_n": HONEST_N, "grid": list(GRID)},
        "scores": {m: {"test_mean": float(sc.test.mean()), "test_std": float(sc.test.std()),
                       "train_mean": float(sc.train.mean()),
                       "test_per_seed": sc.test.tolist(), "train_per_seed": sc.train.tolist()}
                   for m, sc in scores.items()},
        "comparisons": {k: {"diff": c.diff, "p": c.wilcoxon_p,
                            "paired_sign_delta": c.paired_sign_delta, "passes": c.passes}
                        for k, c in cmps.items()},
        "C_gen3_pass": c3, "C_gen4_pass": c4,
        "third_factor_load_bearing": third_load_bearing,
    }
    (_HERE / "exp_ea3_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 判定 ===")
    print(f"  C-gen3 (MAP-E > randselect)          = {c3}")
    print(f"  C-gen4 (MAP-E > panmictic ∧ > random) = {c4}")
    if third_load_bearing:
        print("  → ③ (選択圧/分離) は多タスク汎化で load-bearing。")
    else:
        print("  → ③ は本分布で load-bearing でない (honest negative: 滑らか/③不要)。")


if __name__ == "__main__":
    main()
