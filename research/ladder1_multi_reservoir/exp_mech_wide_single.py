# SPDX-License-Identifier: Apache-2.0
"""機構 wide_single quick 測定 — 幅スケーリングで parity の床が外れるか.

Step C verdict: 単一 leaky reservoir + 線形 ridge readout は delayed_parity (5-bit XOR)
を解けない (基質の床、全 method held-out R²≈0.016)。

本実験は **同一基質 (LeakyDelayLineReservoir) の幅 n_taps だけ** を 8 → 24 → 48 → 64 と
振り、random search の到達天井 (held-out max R²) を測る。baseline は同条件の単層
n_taps=8 (= 幅スイープの最小構成そのもの)。

公平性 (誤帰属の回避):
- gene は ``default_rng(GENE_BASE + seed_idx)`` から連続 draw (seed 固定で再現)。
- 全 gene を **同一 train/eval データ** で評価 (各 gene で eval rng を張り直す) → gene 間公平。
- train/eval を別 draw する held-out 評価でデータリークを作らない (make_eval_once の契約)。

判定:
- floor_lifted = (機構 best 構成の held-out max R²) > 0.5 (parity chance=0, 完全解=1)。
- 構成間比較 (各幅 vs n_taps=8) を strict_compare (片側 Wilcoxon + δ) で出して
  「幅で有意に上がるか」を honest に判定。

帰属 (attribution):
- readout は線形 ridge のまま固定。よって床が外れたら要因は reservoir のダイナミクス
  表現力 (幅由来ランダム射影の豊かさ) = reservoir_expressivity / width。
- 幅をいくら増やしても床が外れないなら、線形 readout の Minsky-Papert 限界が幅では
  超えられないことの経験的確認 (陰性知見)。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # mech_wide_single
sys.path.insert(0, str(_HERE.parent / "step_c_memory_tasks"))  # memory_tasks, strict_compare

from mech_wide_single import WideSingleConfig, random_search_ceiling  # noqa: E402
from memory_tasks import DelayedParityTask  # noqa: E402
from strict_compare import strict_compare  # noqa: E402

WIDTHS: tuple[int, ...] = (8, 24, 48, 64)  # 8=床基準 baseline、24/48/64=超ワイド
N_RANDOM = 300       # random search 1 seed あたりの gene 本数 (到達天井の推定)
N_SEEDS = 8          # 固定 seed の per-seed 試行数
N_TRAIN = 48
N_EVAL = 48
GENE_BASE = 700_001
EVAL_BASE = 900_001
FLOOR_LIFT_THRESHOLD = 0.5
OUT_JSON = _HERE / "exp_mech_wide_single_results.json"


def main() -> None:
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    results: dict[str, object] = {
        "task": "delayed_parity(seq_len=20,window=5)",
        "mechanism": "wide_single",
        "n_random": N_RANDOM, "n_seeds": N_SEEDS,
        "n_train": N_TRAIN, "n_eval": N_EVAL,
        "floor_lift_threshold": FLOOR_LIFT_THRESHOLD,
        "widths": list(WIDTHS),
        "ceilings": {},
    }

    print("=== 機構 wide_single: 幅スケーリング天井測定 ===")
    print(f"task=delayed_parity(seq_len=20,window=5)  n_random={N_RANDOM} "
          f"n_seeds={N_SEEDS}  held-out R² (chance=0, solved=1)\n")

    ceilings: dict[int, np.ndarray] = {}
    for w in WIDTHS:
        cfg = WideSingleConfig(n_taps=w, in_dim=task.in_dim)
        t0 = time.time()
        vals = np.array([
            random_search_ceiling(
                cfg, task,
                n_random=N_RANDOM, seed_idx=s,
                n_train=N_TRAIN, n_eval=N_EVAL,
                gene_base=GENE_BASE, eval_base=EVAL_BASE,
            )
            for s in range(N_SEEDS)
        ], dtype=np.float64)
        dt = time.time() - t0
        ceilings[w] = vals
        results["ceilings"][str(w)] = {
            "n_taps": w, "gene_dim": cfg.gene_dim, "total_taps": cfg.total_taps,
            "per_seed_max_r2": vals.tolist(),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "max": float(vals.max()), "min": float(vals.min()),
        }
        print(f"[{cfg.label:12s}] n_taps={w:2d} gene_dim={cfg.gene_dim:3d}  "
              f"max R²: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"(min={vals.min():.3f} max={vals.max():.3f})  {dt:.1f}s")

    base = ceilings[8]
    base_max = float(base.max())
    base_mean = float(base.mean())

    # best 構成 (= 機構の到達 held-out max R²)。pooled max over 全幅 (8 含む) の最大単一 R²。
    mech_best_single_max = max(float(v.max()) for w, v in ceilings.items())
    # 機構を「8 を除く超ワイド」の最良 mean としても要約 (幅由来の効果分離)。
    wide_only = {w: v for w, v in ceilings.items() if w != 8}
    best_wide_w = max(wide_only, key=lambda w: float(wide_only[w].mean()))
    mech_wide_best_max = float(wide_only[best_wide_w].max())

    floor_lifted = mech_best_single_max > FLOOR_LIFT_THRESHOLD
    results["baseline_1L8_max_r2"] = base_max
    results["baseline_1L8_mean_r2"] = base_mean
    results["mechanism_best_max_r2"] = mech_best_single_max
    results["mechanism_best_wide_only_max_r2"] = mech_wide_best_max
    results["best_wide_width"] = best_wide_w
    results["floor_lifted"] = bool(floor_lifted)

    # --- 判定: 幅で有意に上がるか (各幅 vs n_taps=8) ---
    print("\n=== 判定: 幅増は n_taps=8 を有意に上回るか (strict gate) ===")
    print("  (注: N_SEEDS=8 < strict_compare min_seeds=15 なので passes は機械的に False。"
          " diff/p/δ を記述統計として見る)")
    results["vs_baseline"] = {}
    for w in WIDTHS:
        if w == 8:
            continue
        r = strict_compare(ceilings[w], base, f"1L-{w}wide", "1L-8")
        results["vs_baseline"][str(w)] = {
            "diff": r.diff, "wilcoxon_p": r.wilcoxon_p,
            "paired_sign_delta": r.paired_sign_delta,
            "win_rate": r.win_rate, "passes": r.passes,
        }
        print(f"  1L-{w}wide vs 1L-8: diff={r.diff:+.4f} p={r.wilcoxon_p:.4g} "
              f"δ={r.paired_sign_delta:+.2f} win={r.win_rate:.2f} passes={r.passes}")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- honest 要約 ---
    print("\n=== honest 要約 ===")
    print(f"  baseline 1L-8 (床)        : max R² = {base_max:.4f} (mean={base_mean:.4f})")
    print(f"  mechanism wide_single best: max R² = {mech_best_single_max:.4f}")
    print(f"  超ワイドのみ best (n_taps={best_wide_w}): max R² = {mech_wide_best_max:.4f}")
    floor_note = ("床が外れた (held-out max R² > 0.5)" if floor_lifted
                  else "床は外れていない (held-out max R² <= 0.5 = parity 未解決)")
    print(f"  floor_lifted = {floor_lifted}  → {floor_note}")
    if base_max < 0.2:
        print("  → 1L-8 は Step C の床 R²≈0 を再現")
    if not floor_lifted:
        print("  → 線形 readout のまま幅だけ増やしても XOR の床は超えない "
              "(Minsky-Papert 限界の経験的確認 / 陰性知見)")
    print(f"\n結果を保存: {OUT_JSON}")

    # 機械可読 1 行サマリ (orchestrator 回収用)。
    print(f"\nSUMMARY baseline_1L8_max_r2={base_max:.4f} "
          f"mechanism_max_r2={mech_best_single_max:.4f} "
          f"floor_lifted={floor_lifted}")


if __name__ == "__main__":
    main()
