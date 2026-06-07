# SPDX-License-Identifier: Apache-2.0
"""梯子段1 機構 parallel_gated quick 測定 — XOR の床が要素積ゲーティングで外れるか.

Step C verdict: 単一 leaky reservoir + 線形 ridge readout は delayed_parity (5-bit XOR)
を解けない (Minsky-Papert の床、held-out R²≈0.016)。本スクリプトは機構 parallel_gated
(K 個の独立 reservoir + ペア間要素積 h^i⊙h^j を ridge readout の追加特徴) で床が外れるかを
held-out max R² で測る。floor_lifted の閾値は held-out max R² > 0.5 (parity chance=0, 完全解=1)。

公平性 (誤帰属の回避):
- baseline = **同条件の単層 n_taps=8** (Step C の床の再現)。
- 全 gene を同一 train/eval データ (seed 固定) で評価して max を取る → 構成間 paired 公平。
- held-out 評価を厳守 (train/eval を別 draw) → データリークなし。
- 構成別 ablation で「要素積の寄与か / 単なる reservoir 数・特徴次元増か」を切り分ける:
    * parallel_gated(K=4)   … 機構そのもの (積項あり)
    * linear_only(K=4)      … 同じ K reservoir だが積項を捨て線形連結のみ
      → これと機構の差 = 「積項 (乗法的相互作用)」の純寄与。
    * 1L-8 baseline         … Step C の床
- evolved_search: 機構 gene を random でなく **ES (μ+λ Gaussian mutation hill-climb)** で
  探索した到達天井。同 budget の random search と比べ、構造ある探索で天井が上がるか。

random search の探索 budget は n_random=300、n_seeds=8 固定 seed。
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
sys.path.insert(0, str(_HERE))  # mech_parallel_gated, multi_reservoir
sys.path.insert(0, str(_HERE.parent / "step_c_memory_tasks"))  # memory_tasks, reservoir, strict_compare

from mech_parallel_gated import ParallelGatedReservoir, gene_bounds as pg_bounds, make_eval_once as pg_eval
from reservoir import LeakyDelayLineReservoir, make_eval_once as single_eval
from memory_tasks import DelayedParityTask
from strict_compare import strict_compare

# 探索 budget (タスク指定)。env で上書き可 (データ飢餓 confound の切り分け用に
# N_TRAIN を feature_dim より大きく取った再測定を別 run で行うため)。
import os as _os

N_RANDOM = int(_os.environ.get("PG_N_RANDOM", "300"))   # random search 1 seed あたりの gene 本数
N_SEEDS = int(_os.environ.get("PG_N_SEEDS", "8"))       # 固定 seed 数
N_TRAIN = int(_os.environ.get("PG_N_TRAIN", "48"))      # held-out 評価の train 本数
N_EVAL = int(_os.environ.get("PG_N_EVAL", "48"))        # held-out 評価の eval 本数 (train と別 draw)
_TAG = _os.environ.get("PG_TAG", "")                    # 出力 JSON 名のサフィックス
GENE_BASE = 730_001
EVAL_BASE = 930_001
OUT_JSON = _HERE / (f"exp_mech_parallel_gated_results{('_' + _TAG) if _TAG else ''}.json")

# ES (evolved_search) ハイパラ — random と同 budget (評価回数) になるよう揃える。
ES_POP = 20          # λ (子個体数)
ES_GENS = 15         # 世代数 → 20*15 = 300 評価 = N_RANDOM と同 budget
ES_SIGMA = 0.6       # gaussian mutation σ (gene 値域 [-4,4]/[-2,2] に対し中程度)
ES_ELITE = 4         # μ (親として残す上位)


def _make_eval(res, kind: str):
    """構成 kind に応じた held-out eval_once を返す."""
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    if kind == "parallel_gated":
        return pg_eval(res, task, n_train=N_TRAIN, n_eval=N_EVAL)
    if kind == "single":
        return single_eval(res, task, n_train=N_TRAIN, n_eval=N_EVAL)
    raise ValueError(kind)


def _make_linear_only_eval(res: ParallelGatedReservoir):
    """K reservoir だが積項を捨てた線形連結のみの held-out eval (ablation).

    機構 features() のうち先頭 K*n_taps (= 線形連結) だけを ridge にかける。
    機構との差が「積項の純寄与」を与える。実装は features を計算し前半を slice する。
    """
    import numpy as _np
    sys.path.insert(0, str(_HERE.parent.parent / "src"))
    from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402

    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    lin_dim = res.n_reservoirs * res.n_taps

    def _collect(gene, n, rng):
        feats, targets = [], []
        for _ in range(n):
            inputs, target = task.generate(rng)
            f = res.features(gene, inputs)[:lin_dim]  # 積項を捨て線形項のみ
            feats.append(f)
            targets.append(_np.atleast_1d(_np.asarray(target, dtype=_np.float64)))
        return _np.array(feats, dtype=_np.float64), _np.array(targets, dtype=_np.float64)

    def eval_once(gene, rng):
        s_tr, y_tr = _collect(gene, N_TRAIN, rng)
        readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=1e-2)
        s_ev, y_ev = _collect(gene, N_EVAL, rng)
        pred = _np.atleast_2d(readout(s_ev))
        mse = float(_np.mean((pred - y_ev) ** 2))
        var = float(_np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        return float(_np.clip(r2, 0.0, 1.0))

    return eval_once


def random_search_ceiling(res, eval_once, gene_sampler, n_random: int, seed_idx: int) -> float:
    """1 seed の random search で到達した max held-out R² を返す.

    全 gene を同一 eval データ (default_rng(EVAL_BASE+seed_idx)) で評価 → gene 間公平。
    """
    gene_rng = np.random.default_rng(GENE_BASE + seed_idx)
    best = 0.0
    for _ in range(n_random):
        gene = gene_sampler(gene_rng)
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)  # 全 gene 同一 train/eval
        best = max(best, eval_once(gene, eval_rng))
    return best


def es_search_ceiling(res, eval_once, lo, hi, seed_idx: int) -> float:
    """μ+λ ES (Gaussian mutation hill-climb) で到達した max held-out R² を返す.

    random と同 budget (ES_POP*ES_GENS = 300 評価) になるよう設定。全個体評価は random と
    同一 eval データ (EVAL_BASE+seed_idx 固定) を使い、探索方式以外を公平に保つ。
    決定論性のため gene 生成 rng を seed 固定。
    """
    gene_rng = np.random.default_rng(GENE_BASE + 50_000 + seed_idx)

    def _eval(g):
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)  # 全 gene 同一 train/eval
        return eval_once(g, eval_rng)

    # 初期集団 (random)。
    pop = [lo + (hi - lo) * gene_rng.random(len(lo)) for _ in range(ES_POP)]
    scored = sorted(((_eval(g), g) for g in pop), key=lambda t: -t[0])
    best = scored[0][0]
    span = hi - lo

    for _ in range(ES_GENS - 1):
        parents = [g for _, g in scored[:ES_ELITE]]
        children = []
        while len(children) < ES_POP:
            p = parents[gene_rng.integers(0, len(parents))]
            child = p + gene_rng.normal(0.0, ES_SIGMA, size=len(p)) * (span / span.max())
            child = np.clip(child, lo, hi)  # bounds 制約
            children.append(child)
        scored = sorted(((_eval(g), g) for g in children), key=lambda t: -t[0])
        best = max(best, scored[0][0])
    return best


def main() -> None:
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    results: dict = {
        "task": "delayed_parity(seq_len=20,window=5)",
        "mechanism": "parallel_gated",
        "n_random": N_RANDOM, "n_seeds": N_SEEDS,
        "n_train": N_TRAIN, "n_eval": N_EVAL,
        "floor_threshold_max_r2": 0.5,
        "configs": {},
    }

    # ---- 機構本体 (parallel_gated K=4 n_taps=6) ----
    pg = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    pg_lo, pg_hi = pg_bounds(pg)
    pg_eval_fn = _make_eval(pg, "parallel_gated")

    # ---- baseline 単層 n_taps=8 (Step C の床) ----
    single = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    single_eval_fn = _make_eval(single, "single")

    # ---- ablation: 同 K reservoir だが積項なし (線形連結のみ) ----
    lin_eval_fn = _make_linear_only_eval(pg)

    ceilings: dict[str, np.ndarray] = {}

    def run_random(label, sampler, eval_fn, meta):
        t0 = time.time()
        vals = np.array([
            random_search_ceiling(None, eval_fn, sampler, N_RANDOM, s)
            for s in range(N_SEEDS)
        ])
        dt = time.time() - t0
        ceilings[label] = vals
        results["configs"][label] = {
            **meta, "search": "random",
            "per_seed_max_r2": vals.tolist(),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "max": float(vals.max()), "seconds": round(dt, 1),
        }
        print(f"[{label:22s}] (random) max R² mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.3f} max={vals.max():.3f}  {dt:.1f}s", flush=True)

    # baseline 単層 n_taps=8。
    run_random("1L-8(baseline)", single.random_gene, single_eval_fn,
               {"n_taps": 8, "gene_dim": single.gene_dim, "feature_dim": 8})
    # 機構 parallel_gated (積項あり)。
    run_random("parallel_gated(K4t6)", pg.random_gene, pg_eval_fn,
               {"n_reservoirs": 4, "n_taps": 6, "gene_dim": pg.gene_dim,
                "feature_dim": pg.feature_dim})
    # ablation 線形連結のみ (積項なし) — 同 gene 空間・同 reservoir。
    run_random("linear_only(K4t6)", pg.random_gene, lin_eval_fn,
               {"n_reservoirs": 4, "n_taps": 6, "gene_dim": pg.gene_dim,
                "feature_dim": pg.n_reservoirs * pg.n_taps})

    # ---- evolved_search (ES) for 機構 ----
    t0 = time.time()
    es_vals = np.array([
        es_search_ceiling(pg, pg_eval_fn, pg_lo, pg_hi, s) for s in range(N_SEEDS)
    ])
    dt = time.time() - t0
    ceilings["parallel_gated(K4t6)-ES"] = es_vals
    results["configs"]["parallel_gated(K4t6)-ES"] = {
        "n_reservoirs": 4, "n_taps": 6, "gene_dim": pg.gene_dim,
        "feature_dim": pg.feature_dim, "search": "evolved_search(ES)",
        "es_pop": ES_POP, "es_gens": ES_GENS, "es_sigma": ES_SIGMA, "es_elite": ES_ELITE,
        "per_seed_max_r2": es_vals.tolist(),
        "mean": float(es_vals.mean()), "std": float(es_vals.std()),
        "max": float(es_vals.max()), "seconds": round(dt, 1),
    }
    print(f"[parallel_gated(K4t6)-ES] (ES)   max R² mean={es_vals.mean():.4f} "
          f"std={es_vals.std():.4f} min={es_vals.min():.3f} max={es_vals.max():.3f}  {dt:.1f}s",
          flush=True)

    # ---- 主要メトリクス ----
    baseline_max = float(ceilings["1L-8(baseline)"].max())
    pg_random_max = float(ceilings["parallel_gated(K4t6)"].max())
    pg_es_max = float(es_vals.max())
    mech_best_max = max(pg_random_max, pg_es_max)
    lin_max = float(ceilings["linear_only(K4t6)"].max())

    results["baseline_1L8_max_r2"] = baseline_max
    results["mechanism_max_r2"] = mech_best_max
    results["linear_only_max_r2"] = lin_max
    results["floor_lifted"] = bool(mech_best_max > 0.5)

    # ---- 構成間 strict_compare (per-seed max R² 配列で paired、参考値) ----
    print("\n=== 構成間 strict_compare (per-seed max R², N_SEEDS=8<15 なので passes は常に False; diff/p/δ 参照用) ===", flush=True)
    comparisons = {}
    for a_label, b_label in [
        ("parallel_gated(K4t6)", "1L-8(baseline)"),
        ("parallel_gated(K4t6)", "linear_only(K4t6)"),
        ("parallel_gated(K4t6)-ES", "parallel_gated(K4t6)"),
    ]:
        r = strict_compare(ceilings[a_label], ceilings[b_label], a_label, b_label)
        comparisons[f"{a_label}_vs_{b_label}"] = {
            "mean_a": r.mean_a, "mean_b": r.mean_b, "diff": r.diff,
            "wilcoxon_p": r.wilcoxon_p, "paired_sign_delta": r.paired_sign_delta,
            "win_rate": r.win_rate, "passes": r.passes,
        }
        print(f"  {a_label} vs {b_label}: diff={r.diff:+.4f} p={r.wilcoxon_p:.4g} "
              f"δ={r.paired_sign_delta:+.2f} win={r.win_rate:.2f}", flush=True)
    results["comparisons"] = comparisons

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- honest 要約 (stdout に主要 R² を print) ----
    print("\n=== 主要メトリクス (stdout) ===", flush=True)
    print(f"baseline_max_r2 (1L-8 単層, 同条件)        = {baseline_max:.4f}", flush=True)
    print(f"mechanism_max_r2 (parallel_gated, best)    = {mech_best_max:.4f} "
          f"(random={pg_random_max:.4f}, ES={pg_es_max:.4f})", flush=True)
    print(f"linear_only_max_r2 (積項なし ablation)     = {lin_max:.4f}", flush=True)
    print(f"floor_lifted (max R² > 0.5)                = {results['floor_lifted']}", flush=True)
    print("\n=== 構成別 max R² 要約 ===", flush=True)
    for label, info in results["configs"].items():
        print(f"  {label:24s}: mean={info['mean']:.4f} max={info['max']:.4f} "
              f"feat_dim={info.get('feature_dim','?')}", flush=True)
    print(f"\n結果を保存: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
