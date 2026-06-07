# SPDX-License-Identifier: Apache-2.0
"""Step C — BG6 (specialist 出現 / task→best-kernel 写像が非定数か) smoke driver.

BREAK_GATES.md BG6 準拠。memory_tasks 各 task (FlipFlop / DelayedRecall / DelayedParity) を
**単独 fitness** にして KernelGenome 集団を進化させ、best gene の ``kernel_index()`` を集計、
**task→best-kernel 写像表**を作る。

- **pass 閾値 (BG6)**: 写像が非定数 = ≥2 種の kernel_id が少なくとも 1 task で best
  (集約 best across seeds で判定)。
- **honest 留保 (厳守)**: 全 task 同一 kernel = specialist 不在 = 「memory_tasks は kernel 中立 =
  多様化の土俵でない」という **valid な honest negative** (捏造して positive に倒さない,
  feedback_benchmark_honest_disclosure)。各 task を複数 seed (smoke 3-5) 回し:
    - best kernel が seed 間で安定か (mode 一致率 / per-seed 内訳)
    - 勝った fitness 差 (best vs 2nd-best kernel の held-out R² 差) が僅差か
  を report する。僅差なら「写像は非定数だが脆い」と明記。

進化器 = selection_lab.panmictic_ga (単目的, dim=5 連続ベクトル, GA operator 無改変流用)。
BG6 は「各 task の best gene の kernel」を見るだけなので、MAP-Elites/baseline 比較 (BG7/BG8) は
本 driver の対象外 (別段)。固定予算 panmictic GA で各 kernel に公平に best を探させる。

実行 (単独可・seed 固定・UTF-8):
    py -3.11 research/kernel_diversification/bg6_specialist.py [--quick]
出力:
    research/kernel_diversification/bg6_specialist_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

# research adapter + sibling research labs を import path に
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))

from kernel_fitness import (  # noqa: E402
    bridge_sanity_check,
    kernel_ga_bounds,
    make_kernel_eval_once,
)
from kernels import KERNEL_NAMES, N_KERNELS  # noqa: E402
from memory_tasks_import import MEMORY_TASKS  # noqa: E402
from selection_lab import panmictic_ga  # noqa: E402
from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402


def _ensure_utf8_stdout() -> None:
    """Windows cp932 console 対策 (feedback_cli_utf8_stdout_pattern)."""
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:  # pragma: no cover - already wrapped
        pass


# smoke 既定 (小 budget)。--quick で更に削減。
DEFAULTS = dict(
    n_seeds=4,        # 3-5 seed 程度 (smoke)
    n_evals=480,      # pop*gen 相当の固定予算
    pop_size=24,      # STAGE_3B 指定 pop ~24
    sigma=0.20,       # 連続ベクトル mutation 幅 (kernel_id 1.0 を跨ぐ余地)
    dim=8,            # kernel channel 数 (memory_tasks 既定の小さめ)
    honest_n_trials=12,
    base_seed=20260602,
)
QUICK = dict(n_seeds=3, n_evals=240, pop_size=16, honest_n_trials=8)


def _best_kernel_for_task(
    task_name: str,
    *,
    n_evals: int,
    pop_size: int,
    sigma: float,
    dim: int,
    honest_n_trials: int,
    base_seed: int,
    n_seeds: int,
) -> dict:
    """1 task を単独 fitness にして n_seeds 回進化 → best gene の kernel を集計.

    各 seed:
      1. panmictic_ga で dim=5 ベクトルを進化 (固定予算)。
      2. best gene を進化と独立な fresh seed で honest 再評価 (artifact 排除)。
      3. best gene の kernel_index を記録。
      4. **各 kernel 別の "best-of-budget honest fitness"** も測り、best vs 2nd の差 (margin) を出す。
         (= GA が見つけた best が本当にその kernel の優位かを per-kernel に切り分ける)
    """
    from kernel_fitness import gene_vec_to_genome  # 局所 import (循環回避)

    task = MEMORY_TASKS[task_name]()
    lo, hi = kernel_ga_bounds()
    bounds = (lo, hi)
    ga_dim = lo.shape[0]  # 5

    per_seed: list[dict] = []
    for s in range(n_seeds):
        # 射影 P は seed ごとに変える (固定構造の自由度に対する写像頑健性を見る)。
        eval_once = make_kernel_eval_once(task, dim=dim, projection_seed=s)
        evo_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 1, s]))
        res = panmictic_ga(
            eval_once, dim=ga_dim, bounds=bounds, n_evals=n_evals,
            pop_size=pop_size, tournament_k=3, sigma=sigma, elitism=1, rng=evo_rng,
        )
        best_genome = gene_vec_to_genome(res.best_gene)
        best_kidx = best_genome.kernel_index()

        # honest 再評価 (進化と独立な fresh seed)
        honest_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 2, s]))
        best_honest = honest_reevaluate(
            eval_once, res.best_gene, n_trials=honest_n_trials, rng=honest_rng)

        # per-kernel 切り分け: 各 kernel を強制した best-of-small-budget honest fitness。
        # GA の best が偶然その kernel に落ちただけでないか、kernel 別の到達 fitness を出す。
        per_kernel_best = _per_kernel_probe(
            eval_once, bounds, ga_dim=ga_dim, dim_unused=dim,
            honest_n_trials=honest_n_trials,
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 3, s])),
        )
        # margin = best kernel vs 2nd-best kernel の honest fitness 差
        ordered = sorted(per_kernel_best.values(), reverse=True)
        margin = float(ordered[0] - ordered[1]) if len(ordered) >= 2 else float("nan")
        probe_best_kernel = max(per_kernel_best, key=per_kernel_best.get)

        per_seed.append({
            "seed": s,
            "ga_best_kernel_index": int(best_kidx),
            "ga_best_kernel_name": KERNEL_NAMES[best_kidx],
            "ga_best_honest_fitness": float(best_honest),
            "ga_best_kernel_id_continuous": float(best_genome.kernel_id),
            "per_kernel_probe_honest": per_kernel_best,
            "probe_best_kernel": probe_best_kernel,
            "probe_margin_best_vs_2nd": margin,
        })

    # 集約: GA best kernel の seed 横断 mode + 安定性、probe best kernel の mode
    ga_kernels = [d["ga_best_kernel_name"] for d in per_seed]
    probe_kernels = [d["probe_best_kernel"] for d in per_seed]
    ga_counts = {n: ga_kernels.count(n) for n in KERNEL_NAMES if ga_kernels.count(n) > 0}
    probe_counts = {n: probe_kernels.count(n) for n in KERNEL_NAMES if probe_kernels.count(n) > 0}
    ga_mode = max(ga_counts, key=ga_counts.get)
    probe_mode = max(probe_counts, key=probe_counts.get)
    ga_mode_rate = ga_counts[ga_mode] / n_seeds
    margins = [d["probe_margin_best_vs_2nd"] for d in per_seed
               if np.isfinite(d["probe_margin_best_vs_2nd"])]
    mean_margin = float(np.mean(margins)) if margins else float("nan")

    return {
        "task": task_name,
        "n_seeds": n_seeds,
        "per_seed": per_seed,
        "ga_best_kernel_counts": ga_counts,
        "probe_best_kernel_counts": probe_counts,
        "ga_best_kernel_mode": ga_mode,           # この task の代表 best kernel (GA 経由)
        "ga_best_kernel_mode_rate": float(ga_mode_rate),  # seed 安定度 [0,1]
        "probe_best_kernel_mode": probe_mode,     # 強制 probe 経由の代表 best kernel
        "mean_probe_margin": mean_margin,         # best vs 2nd の平均差 (僅差判定用)
    }


def _per_kernel_probe(
    eval_once,
    bounds,
    *,
    ga_dim: int,
    dim_unused: int,
    honest_n_trials: int,
    rng: np.random.Generator,
    n_random: int = 40,
) -> dict:
    """各 kernel_id を強制した random gene の best-of-budget honest fitness を kernel 別に出す.

    GA の best が落ちた kernel が、本当にその task で最良到達 fitness を持つかを切り分ける
    軽量 probe。各 kernel に同数 (n_random) の random theta を引き、その kernel に固定した
    best gene を honest 再評価する。GA とは別予算 (probe 専用) なので GA 結果は汚さない。
    """
    lo, hi = bounds
    result: dict[str, float] = {}
    for kidx, name in enumerate(KERNEL_NAMES):
        best_f = -np.inf
        best_vec = None
        for _ in range(n_random):
            vec = lo + (hi - lo) * rng.random(ga_dim)
            vec[0] = kidx + 0.5  # この kernel に floor で落とす
            f = eval_once(vec, rng)
            if f > best_f:
                best_f, best_vec = f, vec.copy()
        # best random gene を honest 再評価 (artifact 排除)
        honest_f = honest_reevaluate(
            eval_once, best_vec, n_trials=honest_n_trials,
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)))
        result[name] = float(honest_f)
    return result


def run(quick: bool = False) -> dict:
    cfg = dict(DEFAULTS)
    if quick:
        cfg.update(QUICK)

    # adapter 健全性 smoke (検証 a/b/c)
    sanity = bridge_sanity_check(dim=cfg["dim"])
    sanity_d = {
        "all_kernels_finite_fitness": sanity.all_kernels_finite_fitness,
        "rwkv_runseq_consistent": sanity.rwkv_runseq_consistent,
        "same_seed_reproducible": sanity.same_seed_reproducible,
        "per_kernel_fitness": sanity.per_kernel_fitness,
    }

    per_task: list[dict] = []
    for task_name in MEMORY_TASKS:
        per_task.append(_best_kernel_for_task(
            task_name,
            n_evals=cfg["n_evals"], pop_size=cfg["pop_size"], sigma=cfg["sigma"],
            dim=cfg["dim"], honest_n_trials=cfg["honest_n_trials"],
            base_seed=cfg["base_seed"], n_seeds=cfg["n_seeds"],
        ))

    # BG6 判定: task→best-kernel 写像が非定数か (≥2 種の kernel が少なくとも 1 task で best)。
    # GA 経由の mode を主写像、probe 経由を補助証拠とする。
    ga_map = {d["task"]: d["ga_best_kernel_mode"] for d in per_task}
    probe_map = {d["task"]: d["probe_best_kernel_mode"] for d in per_task}
    ga_distinct = sorted(set(ga_map.values()))
    probe_distinct = sorted(set(probe_map.values()))
    bg6_pass_ga = len(ga_distinct) >= 2
    bg6_pass_probe = len(probe_distinct) >= 2

    # honest 解釈フラグ
    seed_stable = all(d["ga_best_kernel_mode_rate"] >= 0.5 for d in per_task)
    # 僅差判定: probe margin が小さい task があるか (閾値 0.05 R² = 僅差の目安)
    fragile_tasks = [d["task"] for d in per_task
                     if np.isfinite(d["mean_probe_margin"]) and d["mean_probe_margin"] < 0.05]

    if bg6_pass_ga or bg6_pass_probe:
        if fragile_tasks:
            verdict = ("PASS (non-constant mapping) — but FRAGILE: some tasks win by "
                       f"<0.05 R² margin ({fragile_tasks}). 写像は非定数だが脆い。")
        elif not seed_stable:
            verdict = ("PASS (non-constant mapping) — but SEED-UNSTABLE: best kernel が "
                       "seed 間で過半数一致しない task あり。")
        else:
            verdict = "PASS (non-constant mapping, seed-stable, non-trivial margins)."
    else:
        verdict = ("HONEST NEGATIVE — task→best-kernel 写像が定数 (全 task 同一 kernel)。"
                   "= memory_tasks は kernel 中立 = 多様化の土俵でない (valid negative, "
                   "捏造して positive に倒さない)。")

    return {
        "meta": {
            "gate": "BG6 (specialist 出現 / task->best-kernel 写像が非定数か)",
            "config": cfg,
            "n_kernels": N_KERNELS,
            "kernel_names": list(KERNEL_NAMES),
            "evolver": "selection_lab.panmictic_ga (single-objective, dim=5)",
            "honest_note": (
                "smoke 水準 (小 budget, 3-5 seed)。本実験 (>=15 seed paired Wilcoxon = BG8) は "
                "次段。fitness は固定射影 P + per-gene held-out ridge readout で kernel 寄与を分離。"
                "kernel dynamics は対角 mock (full kernel 非主張)。"
            ),
        },
        "bridge_sanity": sanity_d,
        "per_task": per_task,
        "task_to_best_kernel_map_ga": ga_map,
        "task_to_best_kernel_map_probe": probe_map,
        "distinct_kernels_ga": ga_distinct,
        "distinct_kernels_probe": probe_distinct,
        "bg6_pass_ga": bool(bg6_pass_ga),
        "bg6_pass_probe": bool(bg6_pass_probe),
        "seed_stable": bool(seed_stable),
        "fragile_tasks": fragile_tasks,
        "verdict": verdict,
    }


def _print_summary(res: dict) -> None:
    s = res["bridge_sanity"]
    print("=== adapter sanity (a/b/c) ===")
    print(f"  (a) all_kernels_finite_fitness = {s['all_kernels_finite_fitness']}")
    print(f"  (b) rwkv_runseq_consistent     = {s['rwkv_runseq_consistent']}")
    print(f"  (c) same_seed_reproducible     = {s['same_seed_reproducible']}")
    print(f"      per-kernel sample fitness  = "
          + ", ".join(f"{k}={v:.3f}" for k, v in s["per_kernel_fitness"].items()))
    print("=== BG6 task -> best-kernel map ===")
    for d in res["per_task"]:
        print(f"  {d['task']:15s} GA-mode={d['ga_best_kernel_mode']:16s} "
              f"(stab={d['ga_best_kernel_mode_rate']:.2f}) "
              f"probe-mode={d['probe_best_kernel_mode']:16s} "
              f"margin={d['mean_probe_margin']:.4f} "
              f"counts={d['ga_best_kernel_counts']}")
    print(f"distinct kernels (GA)    = {res['distinct_kernels_ga']}")
    print(f"distinct kernels (probe) = {res['distinct_kernels_probe']}")
    print(f"bg6_pass_ga={res['bg6_pass_ga']} bg6_pass_probe={res['bg6_pass_probe']}")
    print(f"VERDICT: {res['verdict']}")


if __name__ == "__main__":
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="BG6 specialist smoke driver")
    ap.add_argument("--quick", action="store_true", help="更に小 budget で高速 smoke")
    args = ap.parse_args()

    res = run(quick=args.quick)
    out = Path(__file__).resolve().parent / "bg6_specialist_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(res)
    print(f"written: {out}")
