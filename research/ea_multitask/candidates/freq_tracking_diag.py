# SPDX-License-Identifier: Apache-2.0
"""freq_tracking_diag — freq_tracking 候補の③検定土俵としての適格性診断.

測定項目 (honest 規律: measure-first, held-out 厳守, raw 数値報告):

1. **難易度キャリブレーション** — (delay, noise_std) を数点走査し、各 regime の
   random-search held-out R² 天井を見て medium 帯 (0.5–0.9) に落ちる config を選ぶ。
2. **per-regime ceiling** — 選んだ config で各 regime の random-search held-out R²
   天井 (N_RANDOM 本の max を n_seeds 本、honest_reevaluate で best を fresh-seed 再評価)。
3. **generalization** — train/test を未学習 f で分離 (extrapolation hold-out)、
   MAP-E (map_elites_full, n_evals=200) で train 進化 → honest_reevaluate で train/test
   汎化と gap。
4. **niche_signal** — 各 regime で最良 random gene が好む leak (実効時定数) の散らばり
   = behavior descriptor (eff_mem_norm, leak_std) との整合度。regime ごとに最適 leak が
   違えば niche 構造あり。

budget は軽量 (N_RANDOM=120, n_seeds=5, MAP-E n_evals=200)。py 実行は集約。
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
# import path: candidates/ (this) + research/ea_multitask + research/step_c_memory_tasks + src
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # research/ea_multitask (ea_lab, task_mixture)
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # reservoir
sys.path.insert(0, str(_HERE.parents[2] / "src"))  # llcore/src

from ea_lab import map_elites_full  # noqa: E402
from freq_tracking import FreqTrackingTask, make_regimes  # noqa: E402
from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    gene_bounds,
    make_behavior,
    make_eval_once,
    _sigmoid,
)
from task_mixture import TaskMixture, split_regimes  # noqa: E402

# --- budget (軽量) ---
N_TAPS = 8
IN_DIM = 1  # 単一正弦チャネル
N_RANDOM = 120
N_SEEDS = 5
N_EVALS = 200
HONEST_N = 12
SIGMA = 0.12
GRID = (6, 6)
BEHAVIOR_BOUNDS = (np.zeros(2), np.ones(2))  # (eff_mem_norm, leak_std) ∈ [0,1]^2

FREQS = (0.05, 0.1, 0.2, 0.4)  # 周波数 regime 軸


def _best_random_gene(res, task, n_random: int, seed: int):
    """random search で held-out R² を最大化する gene と その R² を返す (selection-on-noise)."""
    ev = make_eval_once(res, task, n_train=48, n_eval=48)
    rng = np.random.default_rng(seed)
    best_g, best_r = None, -np.inf
    for _ in range(n_random):
        g = res.random_gene(rng)
        r = ev(g, rng)
        if r > best_r:
            best_r, best_g = r, g
    return best_g, float(best_r)


def _honest_ceiling(res, task, n_random: int, seed: int) -> float:
    """random-search best を fresh-seed honest 再評価した R² (水増し排除)."""
    best_g, _ = _best_random_gene(res, task, n_random, seed)
    return honest_reevaluate(
        make_eval_once(res, task, n_train=48, n_eval=48),
        best_g, n_trials=HONEST_N, rng=np.random.default_rng(seed + 7_000_000)
    )


def _classify(test_mean: float) -> str:
    if test_mean > 0.9:
        return "saturated"
    if test_mean < 0.5:
        return "floor"
    return "medium"


def calibrate(res, behavior, bounds, dim) -> dict:
    """(delay, noise_std) を数点走査し各 config の regime 別 random ceiling を測る.

    medium 帯 (regime mean が概ね 0.5–0.9、かつ regime 間に差) の config を選ぶ。
    """
    print("\n--- (1) 難易度キャリブレーション: (delay, noise_std) 走査 ---", flush=True)
    # 遅延を増やす / ノイズを増やすほど難化。複数候補を測って medium を選ぶ。
    candidates = [
        {"delay": 3, "noise_std": 0.05},
        {"delay": 6, "noise_std": 0.10},
        {"delay": 10, "noise_std": 0.15},
        {"delay": 14, "noise_std": 0.20},
    ]
    seq_len = 40
    results = []
    for cfg in candidates:
        regimes = make_regimes(FREQS, seq_len=seq_len, **cfg)
        per_regime = {}
        for f, task in zip(FREQS, regimes):
            vals = np.array([
                _honest_ceiling(res, task, N_RANDOM, 100 + int(f * 1000) + s * 13)
                for s in range(N_SEEDS)
            ])
            per_regime[f] = float(vals.mean())
        means = np.array(list(per_regime.values()))
        overall = float(means.mean())
        spread = float(means.max() - means.min())
        cls = _classify(overall)
        results.append({"cfg": cfg, "per_regime_mean": per_regime,
                        "overall_mean": overall, "regime_spread": spread,
                        "class": cls})
        pr = "  ".join(f"f{f}={per_regime[f]:.3f}" for f in FREQS)
        print(f"  delay={cfg['delay']:2d} noise={cfg['noise_std']:.2f}: "
              f"{pr}  overall={overall:.3f} spread={spread:.3f} [{cls}]", flush=True)

    # medium を優先、その中で regime spread (niche 兆候) が大きいものを選ぶ。
    medium = [r for r in results if r["class"] == "medium"]
    if medium:
        chosen = max(medium, key=lambda r: r["regime_spread"])
    else:
        # medium が無ければ overall が 0.7 に最も近い config (honest: medium 不在も報告)
        chosen = min(results, key=lambda r: abs(r["overall_mean"] - 0.7))
    print(f"  → 採用 config: delay={chosen['cfg']['delay']} "
          f"noise_std={chosen['cfg']['noise_std']} "
          f"(class={chosen['class']}, spread={chosen['regime_spread']:.3f})", flush=True)
    return {"sweep": results, "chosen": chosen}


def per_regime_ceiling(res, regimes) -> list[dict]:
    """選んだ config で各 regime の random-search 天井 (mean/max, solvable)."""
    print("\n--- (2) per-regime random-search 天井 (held-out R²) ---", flush=True)
    out = []
    for f, task in zip(FREQS, regimes):
        vals = []
        for s in range(N_SEEDS):
            vals.append(_honest_ceiling(res, task, N_RANDOM, 5000 + int(f * 1000) + s * 17))
        vals = np.array(vals)
        rec = {"regime": f"f={f}", "mean_r2": float(vals.mean()),
               "max_r2": float(vals.max()), "solvable": bool(vals.max() > 0.5)}
        out.append(rec)
        print(f"  f={f:<5}: mean={vals.mean():.3f} max={vals.max():.3f} "
              f"solvable={rec['solvable']}", flush=True)
    return out


def niche_signal(res, regimes) -> dict:
    """各 regime で最良 random gene が好む leak (時定数) の散らばりを測る.

    behavior descriptor は (eff_mem_norm, leak_std)。各 regime の最良 gene の
    eff_mem_norm が regime ごとに違えば「regime ごとに最適時定数が違う」= niche 構造。
    leak_spread = regime 別 eff_mem_norm の標準偏差 (= behavior descriptor 軸上の散らばり)。
    """
    print("\n--- (4) niche_signal: regime 別 最良 gene の leak/時定数 散らばり ---", flush=True)
    behavior = make_behavior(res)
    eff_mems, leak_stds, mean_leaks = [], [], []
    for f, task in zip(FREQS, regimes):
        # 複数 seed の最良 gene を集め、その behavior descriptor を平均する。
        regime_eff, regime_leakstd, regime_meanleak = [], [], []
        for s in range(N_SEEDS):
            g, _ = _best_random_gene(res, task, N_RANDOM, 9000 + int(f * 1000) + s * 19)
            bd = behavior(g)  # (eff_mem_norm, leak_std)
            leak = _sigmoid(g[: res.n_taps])
            regime_eff.append(float(bd[0]))
            regime_leakstd.append(float(bd[1]))
            regime_meanleak.append(float(leak.mean()))
        eff_mems.append(float(np.mean(regime_eff)))
        leak_stds.append(float(np.mean(regime_leakstd)))
        mean_leaks.append(float(np.mean(regime_meanleak)))
        print(f"  f={f:<5}: eff_mem_norm={eff_mems[-1]:.3f} "
              f"leak_std={leak_stds[-1]:.3f} mean_leak={mean_leaks[-1]:.3f}", flush=True)

    # niche 兆候 = regime 間で好む eff_mem_norm / mean_leak がどれだけ散らばるか。
    eff_spread = float(np.std(eff_mems))
    meanleak_spread = float(np.std(mean_leaks))
    # leak_spread_across_regimes: regime ごとの最良 gene が好む leak(=mean_leak) の標準偏差。
    leak_spread = meanleak_spread
    print(f"  → eff_mem_norm spread={eff_spread:.4f}  "
          f"mean_leak spread={meanleak_spread:.4f}", flush=True)
    return {
        "per_regime_eff_mem_norm": {f"f={f}": e for f, e in zip(FREQS, eff_mems)},
        "per_regime_mean_leak": {f"f={f}": l for f, l in zip(FREQS, mean_leaks)},
        "per_regime_leak_std": {f"f={f}": s for f, s in zip(FREQS, leak_stds)},
        "eff_mem_norm_spread": eff_spread,
        "mean_leak_spread": meanleak_spread,
        "leak_spread_across_regimes": leak_spread,
    }


def generalization(res, behavior, bounds, dim, regimes) -> dict:
    """train/test を未学習 f で分離 → MAP-E train 進化 → honest train/test 再評価."""
    print("\n--- (3) generalization: MAP-E train 進化 → hold-out test ---", flush=True)
    # extrapolation hold-out: train={f 0.05,0.2}, test={f 0.1,0.4}
    # (内挿 0.1 と外挿 0.4 の両方を test に含めて補間/外挿の混合を見る)
    train_regimes, test_regimes = split_regimes(regimes, test_idx=[1, 3])
    mix_tr, mix_te = TaskMixture(train_regimes), TaskMixture(test_regimes)
    ev_tr = make_eval_once(res, mix_tr, n_train=48, n_eval=48)
    ev_te = make_eval_once(res, mix_te, n_train=48, n_eval=48)

    train_scores, test_scores = [], []
    for s in range(N_SEEDS):
        r = map_elites_full(
            ev_tr, behavior, dim=dim, bounds=bounds,
            behavior_bounds=BEHAVIOR_BOUNDS, grid_shape=GRID, n_evals=N_EVALS,
            init_batch=max(20, N_EVALS // 10), sigma=SIGMA,
            rng=np.random.default_rng(11000 + s))
        tr = honest_reevaluate(ev_tr, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(12000 + s))
        te = honest_reevaluate(ev_te, r.best_gene, n_trials=HONEST_N,
                               rng=np.random.default_rng(13000 + s))
        train_scores.append(tr); test_scores.append(te)
        print(f"  seed {s}: train R²={tr:.3f}  test(hold-out) R²={te:.3f}  "
              f"gap={tr - te:+.3f}", flush=True)

    train_arr, test_arr = np.array(train_scores), np.array(test_scores)
    gap = float(train_arr.mean() - test_arr.mean())
    return {
        "split": "train f{0.05,0.2} / test f{0.1(interp),0.4(extrap)}",
        "train_mean": float(train_arr.mean()),
        "test_mean": float(test_arr.mean()),
        "gap": gap,
        "train_per_seed": train_arr.tolist(),
        "test_per_seed": test_arr.tolist(),
    }


def main() -> None:
    print("=== freq_tracking_diag: ③検定土俵としての適格性診断 ===")
    res = LeakyDelayLineReservoir(n_taps=N_TAPS, in_dim=IN_DIM)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim

    cal = calibrate(res, behavior, bounds, dim)
    chosen_cfg = cal["chosen"]["cfg"]
    regimes = make_regimes(FREQS, seq_len=40, **chosen_cfg)

    ceiling = per_regime_ceiling(res, regimes)
    gen = generalization(res, behavior, bounds, dim, regimes)
    niche = niche_signal(res, regimes)

    diff_class = _classify(gen["test_mean"])
    viable = bool(
        diff_class == "medium"
        and gen["gap"] > 0
        and niche["leak_spread_across_regimes"] > 0.02  # niche 兆候の閾 (保守的)
    )

    out = {
        "experiment": "freq_tracking candidate diagnostic (E-A ③検定土俵適格性)",
        "config": {"n_taps": N_TAPS, "in_dim": IN_DIM, "n_random": N_RANDOM,
                   "n_seeds": N_SEEDS, "n_evals": N_EVALS, "honest_n": HONEST_N,
                   "freqs": list(FREQS), "chosen_difficulty": chosen_cfg},
        "calibration": cal,
        "per_regime_ceiling": ceiling,
        "generalization": gen,
        "niche_signal": niche,
        "difficulty_class": diff_class,
        "viable_for_third_factor": viable,
    }
    (_HERE / "freq_tracking_diag_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 診断まとめ ===")
    print(f"  chosen difficulty: {chosen_cfg}")
    print(f"  generalization: train_mean={gen['train_mean']:.3f} "
          f"test_mean={gen['test_mean']:.3f} gap={gen['gap']:+.3f}")
    print(f"  difficulty_class: {diff_class}")
    print(f"  niche leak_spread_across_regimes: "
          f"{niche['leak_spread_across_regimes']:.4f}")
    print(f"  viable_for_third_factor: {viable}")
    print(f"\n  results JSON: {(_HERE / 'freq_tracking_diag_results.json')}")


if __name__ == "__main__":
    main()
