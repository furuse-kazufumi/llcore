# SPDX-License-Identifier: Apache-2.0
"""temporal_xor2_delay の③検定土俵診断 (C-gen1 床 + C-gen2 汎化ギャップ + niche_signal).

E_A_DESIGN のゲートを measure-first で測る (baseline-first / honest 規律):

1. **per-regime ceiling (C-gen1)**: 各 delay D ∈ {10,20,30,40} を基質
   (LeakyDelayLineReservoir(n_taps=8, in_dim=1) + 線形 ridge readout) が random search で
   解けるか。held-out max R² > 0.5 を solvable とする (床でない)。
   selection-on-noise 水増し回避のため、各 seed の best random gene を ``honest_reevaluate``
   で fresh-seed 再評価した値も併記する。

2. **generalization (C-gen2)**: train regimes (D={10,20}) で MAP-E 進化 → test (hold-out)
   regimes (D={30,40}) で honest 再評価。train/test 汎化と gap (train_mean - test_mean) を測る。
   外挿 hold-out (test は train より長い delay) なので「未学習 regime への汎化」を検定する。

3. **niche_signal**: 各 regime で最良 random gene が好む leak / eff_mem を behavior descriptor
   で測り、regime 間の散らばりを定量化。散らばり大 = 異なる D が異なる時定数を要求する niche
   構造あり = ③(選択圧/分離) が load-bearing になりうる土俵。

honest: held-out 厳守 (make_eval_once が train/eval 別 draw)。raw 数値を報告。
budget 軽量 (N_RANDOM=120, n_seeds=5, MAP-E n_evals=200)。py 実行は集約 (Windows flash 削減)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Windows cp932 console での em-dash/日本語出力対策に stdout/stderr を UTF-8 reconfigure。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

_HERE = Path(__file__).resolve().parent
# import path: candidates → ea_multitask → research → llcore。
# ea_multitask (ea_lab, task_mixture) / step_c_memory_tasks (reservoir) / src (honest_eval)。
sys.path.insert(0, str(_HERE.parent))  # research/ea_multitask
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # research/step_c_memory_tasks
sys.path.insert(0, str(_HERE.parents[2] / "src"))  # llcore/src

from ea_lab import map_elites_full  # noqa: E402
from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    gene_bounds,
    make_behavior,
    make_eval_once,
)
from task_mixture import TaskMixture, split_regimes  # noqa: E402

from temporal_xor2_delay import TemporalXor2DelayTask  # noqa: E402

N_TAPS = 8
IN_DIM = 1  # XOR ビットを 1 チャネル直列投入 (全 regime 同一)
N_RANDOM = 120
N_SEEDS = 5
N_EVALS = 200
HONEST_N = 12
SIGMA = 0.12
GRID = (6, 6)
DELAYS = [10, 20, 30, 40]
TEST_IDX = [2, 3]  # hold-out (外挿): train D={10,20} / test D={30,40}


def _random_search_best(res, task, n_random: int, seed: int):
    """random search で best held-out R² の gene を返す (gene, raw_best_r2).

    selection-on-noise の水増しがありうるため raw best は別途 honest 再評価する。
    """
    ev = make_eval_once(res, task, n_train=64, n_eval=64)
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


def main() -> None:
    print("=== temporal_xor2_delay 診断: C-gen1 床 + C-gen2 汎化 + niche_signal ===",
          flush=True)
    res = LeakyDelayLineReservoir(n_taps=N_TAPS, in_dim=IN_DIM)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim

    # --- C-gen1: regime 別 random-search 天井 + niche_signal 用 best-gene behavior ---
    print("\n--- C-gen1: regime (delay D) 別 random-search 天井 (held-out R²) ---",
          flush=True)
    per_regime = []
    # niche_signal: 各 regime で最良 (honest 再評価後) random gene が好む leak/eff_mem を収集。
    regime_best_behavior = {}  # D -> (eff_mem_norm, leak_std)
    for D in DELAYS:
        task = TemporalXor2DelayTask(delay=D)
        ev = make_eval_once(res, task, n_train=64, n_eval=64)
        raw_bests = []
        honest_bests = []
        best_overall_gene = None
        best_overall_honest = -np.inf
        for s in range(N_SEEDS):
            g, raw = _random_search_best(res, task, N_RANDOM, seed=10_000 + 137 * D + s)
            raw_bests.append(raw)
            # fresh-seed 再評価 (selection-on-noise 水増し排除)。
            h = honest_reevaluate(ev, g, n_trials=HONEST_N,
                                  rng=np.random.default_rng(50_000 + 137 * D + s))
            honest_bests.append(h)
            if h > best_overall_honest:
                best_overall_honest = h
                best_overall_gene = g
        raw_arr = np.array(raw_bests)
        honest_arr = np.array(honest_bests)
        # mean_r2 = honest 再評価の平均 (water-down 排除後の真の到達水準), max = honest の max。
        mean_r2 = float(honest_arr.mean())
        max_r2 = float(honest_arr.max())
        solvable = bool(max_r2 > 0.5)
        bd = behavior(best_overall_gene)  # (eff_mem_norm, leak_std)
        regime_best_behavior[D] = (float(bd[0]), float(bd[1]))
        leak = 1.0 / (1.0 + np.exp(-np.clip(best_overall_gene[:N_TAPS], -500, 500)))
        per_regime.append({
            "regime": f"D{D}",
            "delay": D,
            "mean_r2": mean_r2,
            "max_r2": max_r2,
            "raw_max_r2": float(raw_arr.max()),
            "raw_mean_r2": float(raw_arr.mean()),
            "solvable": solvable,
            "best_gene_eff_mem_norm": float(bd[0]),
            "best_gene_leak_std": float(bd[1]),
            "best_gene_leak_mean": float(leak.mean()),
        })
        print(f"  D{D:<3d}: honest max={max_r2:.3f} mean={mean_r2:.3f} "
              f"(raw max={raw_arr.max():.3f}) solvable={solvable} "
              f"| best gene leak_mean={leak.mean():.3f} eff_mem={bd[0]:.3f}",
              flush=True)

    # niche_signal: regime 間で最良 gene が好む eff_mem / leak_mean の散らばり。
    eff_mems = np.array([per_regime[i]["best_gene_eff_mem_norm"] for i in range(len(DELAYS))])
    leak_means = np.array([per_regime[i]["best_gene_leak_mean"] for i in range(len(DELAYS))])
    leak_stds = np.array([per_regime[i]["best_gene_leak_std"] for i in range(len(DELAYS))])
    # 散らばり = regime 間の std。eff_mem (記憶長指向) の散らばりを主指標、leak_mean を補助。
    niche_eff_mem_spread = float(np.std(eff_mems))
    niche_leak_mean_spread = float(np.std(leak_means))
    # behavior descriptor (eff_mem_norm, leak_std) と整合する散らばり = 2D descriptor 上の
    # regime 重心距離の代理。eff_mem spread + leak_std spread をまとめる。
    leak_spread_across_regimes = float(np.std(eff_mems) + np.std(leak_stds))

    print(f"\n--- niche_signal: regime 間で最良 gene が好む時定数の散らばり ---", flush=True)
    print(f"  eff_mem_norm per regime = {[round(x, 3) for x in eff_mems.tolist()]}", flush=True)
    print(f"  leak_mean    per regime = {[round(x, 3) for x in leak_means.tolist()]}", flush=True)
    print(f"  niche eff_mem spread (std) = {niche_eff_mem_spread:.4f}", flush=True)
    print(f"  leak_spread_across_regimes = {leak_spread_across_regimes:.4f}", flush=True)

    # --- C-gen2: 外挿 hold-out 汎化 (train D={10,20} 進化 → test D={30,40} honest) ---
    print("\n--- C-gen2: 外挿汎化 (MAP-E train 進化 → hold-out test) ---", flush=True)
    regimes = [TemporalXor2DelayTask(delay=D) for D in DELAYS]
    train_regimes, test_regimes = split_regimes(regimes, test_idx=TEST_IDX)
    mix_tr, mix_te = TaskMixture(train_regimes), TaskMixture(test_regimes)
    ev_tr = make_eval_once(res, mix_tr, n_train=64, n_eval=64)
    ev_te = make_eval_once(res, mix_te, n_train=64, n_eval=64)

    train_scores, test_scores = [], []
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

    train_arr, test_arr = np.array(train_scores), np.array(test_scores)
    train_mean = float(train_arr.mean())
    test_mean = float(test_arr.mean())
    gap = train_mean - test_mean

    # difficulty_class 判定 (per-regime ceiling の honest mean の平均で土俵全体を見る)。
    all_ceiling_mean = float(np.mean([pr["mean_r2"] for pr in per_regime]))
    if test_mean > 0.9 or all_ceiling_mean > 0.95:
        difficulty_class = "saturated"
    elif test_mean < 0.5:
        difficulty_class = "floor"
    else:
        difficulty_class = "medium"

    niche_present = bool(leak_spread_across_regimes > 0.05)
    viable = bool(difficulty_class == "medium" and gap > 0.0 and niche_present)

    out = {
        "experiment": "temporal_xor2_delay diagnostic (C-gen1 ceiling + C-gen2 gap + niche)",
        "config": {
            "n_taps": N_TAPS, "in_dim": IN_DIM, "n_random": N_RANDOM,
            "n_seeds": N_SEEDS, "n_evals": N_EVALS, "honest_n": HONEST_N,
            "delays": DELAYS, "test_idx": TEST_IDX, "sigma": SIGMA, "grid": list(GRID),
        },
        "per_regime_ceiling": per_regime,
        "niche_signal": {
            "eff_mem_norm_per_regime": eff_mems.tolist(),
            "leak_mean_per_regime": leak_means.tolist(),
            "leak_std_per_regime": leak_stds.tolist(),
            "niche_eff_mem_spread": niche_eff_mem_spread,
            "niche_leak_mean_spread": niche_leak_mean_spread,
            "leak_spread_across_regimes": leak_spread_across_regimes,
            "niche_present": niche_present,
        },
        "generalization": {
            "split": f"train D={{{DELAYS[0]},{DELAYS[1]}}} / test(hold-out) D={{{DELAYS[2]},{DELAYS[3]}}}",
            "train_mean": train_mean,
            "test_mean": test_mean,
            "gap": gap,
            "train_per_seed": train_arr.tolist(),
            "test_per_seed": test_arr.tolist(),
        },
        "all_ceiling_mean": all_ceiling_mean,
        "difficulty_class": difficulty_class,
        "viable_for_third_factor": viable,
    }
    out_path = _HERE / "temporal_xor2_delay_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 診断まとめ ===", flush=True)
    n_solv = sum(1 for pr in per_regime if pr["solvable"])
    print(f"  C-gen1: {n_solv}/{len(per_regime)} regime が solvable (honest max R²>0.5)",
          flush=True)
    print(f"  ceiling honest mean (土俵全体) = {all_ceiling_mean:.3f}", flush=True)
    print(f"  C-gen2: train_mean={train_mean:.3f} test_mean={test_mean:.3f} gap={gap:+.3f}",
          flush=True)
    print(f"  niche: leak_spread_across_regimes={leak_spread_across_regimes:.4f} "
          f"(present={niche_present})", flush=True)
    print(f"  difficulty_class = {difficulty_class}", flush=True)
    print(f"  viable_for_third_factor = {viable}", flush=True)
    print(f"  → JSON: {out_path}", flush=True)


if __name__ == "__main__":
    main()
