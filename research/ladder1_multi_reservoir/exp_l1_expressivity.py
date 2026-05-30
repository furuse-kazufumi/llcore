# SPDX-License-Identifier: Apache-2.0
"""梯子段1-①: 表現力 sanity — 複数 reservoir 結合で「基質の床」が外れるか.

Step C verdict: 単一 leaky reservoir + ridge は delayed_parity (5-bit XOR) を解けない
(基質の床, 全 method R²≈0)。本実験は random search で各 reservoir 構成の **到達可能
天井 (max held-out R²)** を測り、複数 reservoir 結合 (DeepESN) で床が外れるかを判定する。

公平比較の肝 (誤帰属の回避):
- 1L-8   : Step C と同じ単一層 (床の再現確認)
- 1L-16  : taps 増のみ・深さ 1
- 2L-8x8 : total_taps=16 で 1L-16 と同規模・深さ 2  ← 深さ (層間非線形合成) の効果を分離
- 3L-8x8x8: 深さ 3

判定:
- 「床が外れた」= 多層 max R² が 1L-8 を強化 honest 基準で有意に上回る
- 「深さの寄与」= 2L-8x8 が 1L-16 (同規模) を有意に上回る
  → 上回らなければ「単に大きい reservoir」で説明でき、深さは誤帰属

各 seed で gene を random draw し、全 gene を同一 train/eval データ (seed 固定) で評価して
max を取る → 構成間 paired 公平。per-seed max R² 配列を JSON 保存 (後解析・再現用)。
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
sys.path.insert(0, str(_HERE))  # multi_reservoir
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # memory_tasks, strict_compare

from multi_reservoir import DeepReservoir, make_eval_once
from memory_tasks import DelayedParityTask
from strict_compare import strict_compare

# 構成: (label, layer_taps)
CONFIGS = [
    ("1L-8", (8,)),
    ("1L-16", (16,)),
    ("2L-8x8", (8, 8)),
    ("3L-8x8x8", (8, 8, 8)),
]
N_RANDOM = 400      # random search 1 seed あたりの gene 本数 (到達天井の推定)
N_SEEDS = 10        # paired 比較用 seed 数 (strict_compare min_seeds=15 は最終③で、ここは天井推定)
GENE_BASE = 700_001
EVAL_BASE = 900_001
OUT_JSON = _HERE / "exp_l1_results.json"


def random_search_ceiling(res: DeepReservoir, task, n_random: int, seed_idx: int) -> float:
    """1 seed の random search で到達した max held-out R² を返す.

    全 gene を同一 eval データ (default_rng(EVAL_BASE+seed_idx)) で評価 → gene 間公平。
    """
    eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
    gene_rng = np.random.default_rng(GENE_BASE + seed_idx)
    best = 0.0
    for _ in range(n_random):
        gene = res.random_gene(gene_rng)
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)  # 全 gene 同一 train/eval
        best = max(best, eval_once(gene, eval_rng))
    return best


def main() -> None:
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    results: dict[str, dict] = {"task": "delayed_parity(seq_len=20,window=5)",
                                "n_random": N_RANDOM, "n_seeds": N_SEEDS, "ceilings": {}}

    ceilings: dict[str, np.ndarray] = {}
    for label, taps in CONFIGS:
        res = DeepReservoir(layer_taps=taps, in_dim=task.in_dim)
        t0 = time.time()
        vals = np.array([random_search_ceiling(res, task, N_RANDOM, s) for s in range(N_SEEDS)])
        dt = time.time() - t0
        ceilings[label] = vals
        results["ceilings"][label] = {
            "layer_taps": list(taps), "total_taps": res.total_taps,
            "gene_dim": res.gene_dim, "per_seed_max_r2": vals.tolist(),
            "mean": float(vals.mean()), "std": float(vals.std()),
        }
        print(f"[{label:10s}] taps={res.total_taps:2d} gene_dim={res.gene_dim:3d} "
              f"max R² mean={vals.mean():.4f} std={vals.std():.4f} "
              f"(min={vals.min():.3f} max={vals.max():.3f})  {dt:.1f}s")

    # --- 判定 1: 床が外れたか (各多層 vs 1L-8) ---
    print("\n=== 判定1: 床が外れたか (vs 1L-8 単一層) ===")
    results["floor_lifted"] = {}
    for label in ("1L-16", "2L-8x8", "3L-8x8x8"):
        r = strict_compare(ceilings[label], ceilings["1L-8"], label, "1L-8")
        results["floor_lifted"][label] = {
            "diff": r.diff, "wilcoxon_p": r.wilcoxon_p,
            "paired_sign_delta": r.paired_sign_delta, "passes": r.passes,
        }
        print(f"  {label} vs 1L-8: diff={r.diff:+.4f} p={r.wilcoxon_p:.4g} "
              f"δ={r.paired_sign_delta:+.2f} passes={r.passes}")

    # --- 判定 2: 深さの寄与か単なる規模か (2L-8x8 vs 1L-16, 同 total_taps=16) ---
    print("\n=== 判定2: 深さの寄与 (2L-8x8 vs 1L-16, 同規模 total_taps=16) ===")
    r_depth = strict_compare(ceilings["2L-8x8"], ceilings["1L-16"], "2L-8x8", "1L-16")
    results["depth_effect"] = {
        "diff": r_depth.diff, "wilcoxon_p": r_depth.wilcoxon_p,
        "paired_sign_delta": r_depth.paired_sign_delta, "passes": r_depth.passes,
    }
    print(f"  2L-8x8 vs 1L-16: diff={r_depth.diff:+.4f} p={r_depth.wilcoxon_p:.4g} "
          f"δ={r_depth.paired_sign_delta:+.2f} passes={r_depth.passes}")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を保存: {OUT_JSON}")

    # --- honest 要約 ---
    print("\n=== honest 要約 ===")
    base_mean = ceilings["1L-8"].mean()
    print(f"  1L-8 (単細胞床) max R² = {base_mean:.4f}  "
          f"{'← Step C の床 R²≈0 を再現' if base_mean < 0.2 else '← 床が単一層でも外れている (要検討)'}")
    lifted = [l for l in ("1L-16", "2L-8x8", "3L-8x8x8") if results['floor_lifted'][l]['passes']]
    print(f"  床が外れた構成 (strict gate pass): {lifted if lifted else 'なし'}")
    print(f"  深さの寄与 (2L>1L 同規模): {'あり' if r_depth.passes else 'なし (規模で説明可)'}")


if __name__ == "__main__":
    main()
