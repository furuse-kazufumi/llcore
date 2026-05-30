# SPDX-License-Identifier: Apache-2.0
"""variable_delay_recall の③検定土俵診断 (per-regime 天井 + 汎化ギャップ + niche signal).

E_A_DESIGN の C-gen1/C-gen2 ゲートを variable_delay_recall (regime 軸 = 遅延長 D) に対して測る。

測定:
1. **per-regime random-search 天井** (C-gen1): 各 D で N_RANDOM 本の random gene を
   引き held-out R² の max/mean を取る。max>0.5 で solvable (床でない)。
2. **train/test extrapolation 汎化** (C-gen2): D を train/test に互いに素分割
   (内側 D を train、両端 D を test = 時定数の外挿) し、train mixture で MAP-E 進化 →
   train/test mixture で honest_reevaluate。汎化ギャップ = train_mean − test_mean。
3. **niche signal**: 各 regime で最良 random gene が好む leak (中央値) と eff_mem を集計し、
   regime 間の散らばり (std) を測る。散らばりが大きい = D ごとに異なる時定数を要求 =
   behavior descriptor (eff_mem/leak) と整合した niche 構造の証拠。

honest 規律: held-out 厳守 (make_eval_once が train/eval 別 draw)。random ceiling は
max over N_RANDOM (selection-on-noise) → best gene を honest_reevaluate で fresh-seed 再評価
した値も併記。raw 数値を JSON 保存。py 実行は 1 プロセスに集約。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Windows cp932 console 対策: stdout/stderr を UTF-8 に reconfigure。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

_HERE = Path(__file__).resolve().parent          # research/ea_multitask/candidates
_EA = _HERE.parent                                # research/ea_multitask
_LLCORE = _EA.parents[1]                          # D:/projects/llcore
sys.path.insert(0, str(_HERE))                    # variable_delay_recall
sys.path.insert(0, str(_EA))                      # ea_lab, task_mixture
sys.path.insert(0, str(_LLCORE / "research" / "step_c_memory_tasks"))  # reservoir
sys.path.insert(0, str(_LLCORE / "src"))          # llcore.evolution.honest_eval

from ea_lab import map_elites_full  # noqa: E402
from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    _sigmoid,
    gene_bounds,
    make_behavior,
    make_eval_once,
)
from task_mixture import TaskMixture, split_regimes  # noqa: E402
from variable_delay_recall import make_regimes  # noqa: E402

# --- budget (軽量, 仕様通り) ---
N_TAPS = 8
IN_DIM = 2  # variable_delay_recall (cue ch + distractor ch)
N_RANDOM = 120
N_SEEDS = 5
N_EVALS = 200
HONEST_N = 12
SIGMA = 0.12
GRID = (6, 6)

DELAYS = (15, 30, 45, 60)
DISTRACTOR_AMP = 0.2
# extrapolation hold-out: train = 短 D {15,30}, test = 長 D {45,60}
# (時定数の外挿 — train で見ない「より長い遅延」へ外挿できるか。長遅延ほど蓄積ノイズが
#  増え本来的に難しいので、汎化ギャップ (train>test) が出やすい honest な hold-out。)
TEST_IDX = [2, 3]


def _random_search_best(res, task, n_random: int, seed: int):
    """random search で held-out R² が最大の gene と、その R² (天井) を返す.

    Returns
    -------
    (best_gene, best_r2_selected)
        best_r2_selected は selection-on-noise を含む生の max。
    """
    ev = make_eval_once(res, task, n_train=48, n_eval=48)
    rng = np.random.default_rng(seed)
    best_gene = None
    best_r2 = -np.inf
    for _ in range(n_random):
        g = res.random_gene(rng)
        r2 = ev(g, rng)
        if r2 > best_r2:
            best_r2 = r2
            best_gene = g
    return best_gene, float(best_r2)


def _difficulty_class(test_mean: float) -> str:
    if test_mean < 0.5:
        return "floor"
    if test_mean > 0.9:
        return "saturated"
    return "medium"


def main() -> None:
    print("=== variable_delay_recall 診断: per-regime 天井 + 汎化ギャップ + niche signal ===",
          flush=True)
    res = LeakyDelayLineReservoir(n_taps=N_TAPS, in_dim=IN_DIM)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim
    regimes = make_regimes(DELAYS, distractor_amp=DISTRACTOR_AMP, in_dim=IN_DIM)

    # --- 1. per-regime random-search 天井 (C-gen1) + niche signal 用の best gene 収集 ---
    print(f"\n--- per-regime random-search 天井 (N_RANDOM={N_RANDOM}, n_seeds={N_SEEDS}) ---",
          flush=True)
    per_regime = []
    # regime ごとに「最良 random gene が好む leak 中央値 / eff_mem」を集計 (niche signal)。
    regime_leak_median: list[float] = []
    regime_eff_mem: list[float] = []
    for d, task in zip(DELAYS, regimes):
        selected_r2: list[float] = []
        honest_r2: list[float] = []
        leak_medians: list[float] = []
        eff_mems: list[float] = []
        for s in range(N_SEEDS):
            best_gene, best_r2 = _random_search_best(res, task, N_RANDOM, seed=1000 * d + s)
            selected_r2.append(best_r2)
            # honest: best gene を進化と独立な fresh seed で再評価 (selection-on-noise 排除)
            ev = make_eval_once(res, task, n_train=48, n_eval=48)
            hr = honest_reevaluate(ev, best_gene, n_trials=HONEST_N,
                                   rng=np.random.default_rng(7_000_000 + 1000 * d + s))
            honest_r2.append(hr)
            # best gene が好む時定数: leak の中央値 (小=長期記憶) と behavior eff_mem
            leak = _sigmoid(best_gene[:N_TAPS])
            leak_medians.append(float(np.median(leak)))
            eff_mems.append(float(behavior(best_gene)[0]))

        sel = np.array(selected_r2)
        hon = np.array(honest_r2)
        max_r2 = float(sel.max())
        mean_r2 = float(hon.mean())  # honest mean (fresh-seed 再評価, 水増し排除)
        solvable = bool(max_r2 > 0.5)
        leak_med = float(np.median(leak_medians))
        eff_med = float(np.median(eff_mems))
        regime_leak_median.append(leak_med)
        regime_eff_mem.append(eff_med)
        per_regime.append({
            "regime": f"D{d}",
            "delay": d,
            "max_r2": max_r2,                       # raw max (selection-on-noise 含む)
            "mean_r2": mean_r2,                     # honest fresh-seed 再評価 mean
            "selected_mean_r2": float(sel.mean()),  # 参考: 選抜時 R² の seed 平均
            "solvable": solvable,
            "best_leak_median": leak_med,
            "best_eff_mem": eff_med,
        })
        print(f"  D{d:<3d}: max_r2(sel)={max_r2:.3f} honest_mean={mean_r2:.3f} "
              f"solvable={solvable} leak_med={leak_med:.3f} eff_mem={eff_med:.3f}",
              flush=True)

    # niche signal: regime 間で最良 gene が好む leak の散らばり (std)。
    leak_spread = float(np.std(regime_leak_median))
    eff_mem_spread = float(np.std(regime_eff_mem))
    print(f"\n  niche signal: leak_median across regimes = {regime_leak_median} "
          f"(std={leak_spread:.4f})", flush=True)
    print(f"                eff_mem across regimes      = {regime_eff_mem} "
          f"(std={eff_mem_spread:.4f})", flush=True)

    # --- 2. train/test extrapolation 汎化 (C-gen2) ---
    print(f"\n--- 汎化ギャップ (MAP-E train 進化 → hold-out test, n_seeds={N_SEEDS}) ---",
          flush=True)
    train_regimes, test_regimes = split_regimes(regimes, test_idx=TEST_IDX)
    train_delays = [r.seq_len for r in train_regimes]
    test_delays = [r.seq_len for r in test_regimes]
    mix_tr, mix_te = TaskMixture(train_regimes), TaskMixture(test_regimes)
    ev_tr = make_eval_once(res, mix_tr, n_train=48, n_eval=48)
    ev_te = make_eval_once(res, mix_te, n_train=48, n_eval=48)

    train_scores: list[float] = []
    test_scores: list[float] = []
    for s in range(N_SEEDS):
        r = map_elites_full(
            ev_tr, behavior, dim=dim, bounds=bounds,
            behavior_bounds=(np.zeros(2), np.ones(2)),
            grid_shape=GRID, n_evals=N_EVALS, init_batch=max(20, N_EVALS // 10),
            sigma=SIGMA, rng=np.random.default_rng(1000 + s))
        tr = honest_reevaluate(ev_tr, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(2000 + s))
        te = honest_reevaluate(ev_te, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(3000 + s))
        train_scores.append(tr)
        test_scores.append(te)
        print(f"  seed {s}: train R²={tr:.3f}  test(hold-out) R²={te:.3f}  gap={tr - te:+.3f}",
              flush=True)

    train_arr = np.array(train_scores)
    test_arr = np.array(test_scores)
    gap = float(train_arr.mean() - test_arr.mean())
    difficulty = _difficulty_class(float(test_arr.mean()))
    viable = bool(
        difficulty == "medium" and gap > 0.0 and leak_spread > 0.02
    )

    out = {
        "experiment": "variable_delay_recall: per-regime ceiling + extrapolation gap + niche",
        "config": {
            "n_taps": N_TAPS, "in_dim": IN_DIM, "n_random": N_RANDOM,
            "n_seeds": N_SEEDS, "n_evals": N_EVALS, "honest_n": HONEST_N,
            "sigma": SIGMA, "grid": list(GRID),
            "delays": list(DELAYS), "distractor_amp": DISTRACTOR_AMP,
            "test_idx": TEST_IDX,
        },
        "per_regime_ceiling": per_regime,
        "niche_signal": {
            "regime_leak_median": regime_leak_median,
            "regime_eff_mem": regime_eff_mem,
            "leak_spread_across_regimes": leak_spread,
            "eff_mem_spread_across_regimes": eff_mem_spread,
        },
        "generalization": {
            "split": f"train D{train_delays} / test(hold-out) D{test_delays}",
            "train_mean": float(train_arr.mean()),
            "test_mean": float(test_arr.mean()),
            "gap": gap,
            "train_per_seed": train_arr.tolist(),
            "test_per_seed": test_arr.tolist(),
        },
        "difficulty_class": difficulty,
        "viable_for_third_factor": viable,
    }
    out_path = _HERE / "variable_delay_recall_diag_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 診断まとめ ===", flush=True)
    n_solv = sum(1 for v in per_regime if v["solvable"])
    print(f"  per-regime: {n_solv}/{len(per_regime)} regime が solvable (max R²>0.5)")
    print(f"  generalization: train_mean={train_arr.mean():.3f} test_mean={test_arr.mean():.3f} "
          f"gap={gap:+.3f}")
    print(f"  niche: leak_spread={leak_spread:.4f}")
    print(f"  difficulty_class={difficulty}  viable_for_third_factor={viable}")
    print(f"  JSON -> {out_path}")


if __name__ == "__main__":
    main()
