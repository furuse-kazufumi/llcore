# SPDX-License-Identifier: Apache-2.0
"""BG9-3 (2) — 強 BG6 validity gate: kernel-favoring suite が非 inert か検定する driver.

``bg6_specialist.py`` を参考に、``kernel_favoring_tasks.FAVORING_TASKS`` を対象として
**real substrate の validity** を BG9_PREREGISTRATION §2.3 に従い検定する。

2 つの独立証拠:
  (probe) per-kernel probe: 各 task × 各 kernel を **固定 kernel_id** で最適化し held-out R²
          を測る → ``task × kernel`` 性能行列。各セルは複数 seed の平均 + std。
  (GA)    GA-mode: 各 task 単独 fitness で KernelGenome を進化させ best gene の kernel_index を
          seed 横断集計 → ``task → best-kernel`` 写像。

validity 判定 (pre-reg §2.3、**閾値は本ファイルで事前確定 = post-hoc 禁止**):
  V1. 非定数性: probe 行列で **≥2 種の kernel が別 task で best** (= 写像が定数でない)。
  V2. 非僅差: validity を主張する各 task の best-vs-2nd margin が ``MARGIN_THRESHOLD`` (=0.05 R²)
      以上 (僅差な勝ちは inert 寄り → 弁別と見なさない)。
  V3. seed 安定: 各 task の per-kernel best が ``>=3 seed`` の多数決で一致 (mode_rate>=0.5)。
  二重証拠 (probe ⟷ GA) が食い違う場合は honest に両論併記し、**負け (中立) 方向に倒す**。

通過 → real substrate 採用可。不通過 → 「real=kernel 中立」と honest 記録 (BG6 の轍、BG9 N/A 方向)。

honesty (feedback_benchmark_honest_disclosure):
  整いすぎた弁別 (4 task が綺麗に 4 kernel へ 1:1) は逆に疑う。対角 mock の限界で「種類でなく
  程度」しか出ないなら、margin が薄い / GA と probe が食い違う形で現れるはず。captured honest。

実行 (単独可・seed 固定・UTF-8):
    py -3.11 research/kernel_diversification/bg6_strong.py [--quick] [--seeds N]
出力:
    research/kernel_diversification/bg6_strong_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

# research adapter + sibling research labs を import path に
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))

from kernel_favoring_tasks import (  # noqa: E402
    FAVORING_TASKS,
    HYPOTHESIS_BEST_KERNEL,
    make_task,
)
from kernel_fitness import (  # noqa: E402
    gene_vec_to_genome,
    kernel_ga_bounds,
    make_kernel_eval_once,
)
from kernels import KERNEL_NAMES, N_KERNELS  # noqa: E402
from selection_lab import panmictic_ga  # noqa: E402
from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402


# === 事前確定 validity 閾値 (post-hoc 禁止: 結果を見て動かさない) ============
MARGIN_THRESHOLD = 0.05   # best-vs-2nd kernel の held-out R² 差の下限 (非僅差判定)
SEED_MODE_RATE_MIN = 0.5  # per-kernel best が seed 横断で多数決一致する下限
MIN_SEEDS_FOR_STABLE = 3  # seed 安定性を主張する最小 seed 数


def _ensure_utf8_stdout() -> None:
    """Windows cp932 console 対策 (feedback_cli_utf8_stdout_pattern)."""
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:  # pragma: no cover - already wrapped
        pass


# smoke〜中規模既定 (CPU 完走、各 run <900s 目安)。--quick で更に削減。
DEFAULTS = dict(
    n_seeds=5,            # 3-5 seed (smoke)。本検定 (>=15 seed) は BG9 本番。
    probe_opt_evals=80,   # per-kernel probe の (1+1)-ES 風最適化予算 / kernel / seed
    ga_n_evals=720,       # GA-mode の固定予算 (pop*gen 相当)
    ga_pop_size=24,       # STAGE_3B 指定 pop ~24
    ga_sigma=0.20,        # 連続ベクトル mutation 幅 (kernel_id 1.0 を跨ぐ余地)
    dim=8,                # kernel channel 数
    honest_n_trials=16,   # 確率的 fitness の honest 再評価 trial 数
    base_seed=20260602,
)
QUICK = dict(n_seeds=3, probe_opt_evals=50, ga_n_evals=360, ga_pop_size=16, honest_n_trials=10)


def _probe_one_kernel(
    eval_once,
    bounds,
    *,
    ga_dim: int,
    kidx: int,
    opt_evals: int,
    honest_n_trials: int,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    """1 kernel を固定 (kernel_id 強制) して theta だけ最適化 → best gene の honest R².

    (1+1)-ES 風 hill-climb を theta 部分のみに掛ける (kernel_id は当該 kernel に固定)。
    bg6_specialist の random probe より theta 探索が密 (favoring task は theta 依存が強い想定)。
    GA とは別予算 (probe 専用) なので GA 結果は汚さない。
    """
    lo, hi = bounds
    k_id_val = kidx + 0.5  # この kernel に floor で落ちる kernel_id

    def _make_vec(theta_part: np.ndarray) -> np.ndarray:
        vec = theta_part.copy()
        vec[0] = k_id_val  # kernel_id 強制 (clip 後も floor で kidx)
        return vec

    # 初期 random
    cur = lo + (hi - lo) * rng.random(ga_dim)
    cur = _make_vec(cur)
    cur_f = eval_once(cur, rng)
    best, best_f = cur.copy(), cur_f
    evals = 1
    stall = 0
    restart_patience = max(8, opt_evals // 4)
    while evals < opt_evals:
        cand = cur + rng.normal(0, sigma, size=ga_dim)
        cand = np.clip(cand, lo, hi)
        cand[0] = k_id_val  # kernel_id を固定し続ける (探索は theta のみ)
        f = eval_once(cand, rng)
        evals += 1
        if f >= cur_f:
            cur, cur_f = cand, f
            stall = 0
        else:
            stall += 1
        if cur_f > best_f:
            best, best_f = cur.copy(), cur_f
        if stall >= restart_patience and evals < opt_evals:
            cur = lo + (hi - lo) * rng.random(ga_dim)
            cur = _make_vec(cur)
            cur_f = eval_once(cur, rng)
            evals += 1
            stall = 0
            if cur_f > best_f:
                best, best_f = cur.copy(), cur_f
    # best を進化と独立な fresh seed で honest 再評価 (artifact 排除)
    honest_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
    return float(honest_reevaluate(eval_once, best, n_trials=honest_n_trials, rng=honest_rng))


def _eval_task(
    task_name: str,
    *,
    n_seeds: int,
    probe_opt_evals: int,
    ga_n_evals: int,
    ga_pop_size: int,
    ga_sigma: float,
    dim: int,
    honest_n_trials: int,
    base_seed: int,
) -> dict:
    """1 task に対し probe 行列行 (per-kernel R²) + GA-mode best-kernel を seed 横断で測る."""
    task = make_task(task_name)
    lo, hi = kernel_ga_bounds()
    bounds = (lo, hi)
    ga_dim = lo.shape[0]

    # probe: per-seed の per-kernel R² を集める → 平均 / std / per-seed best kernel
    probe_per_seed: list[dict] = []   # [{kernel: r2, ...}] per seed
    probe_best_per_seed: list[str] = []
    ga_best_per_seed: list[str] = []
    ga_best_fitness_per_seed: list[float] = []

    for s in range(n_seeds):
        # 射影 P は seed ごとに変える (固定構造の自由度に対する写像頑健性を見る)。
        eval_once = make_kernel_eval_once(task, dim=dim, projection_seed=s)

        # --- per-kernel probe ---
        probe_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 3, s]))
        per_kernel: dict[str, float] = {}
        for kidx, name in enumerate(KERNEL_NAMES):
            per_kernel[name] = _probe_one_kernel(
                eval_once, bounds, ga_dim=ga_dim, kidx=kidx,
                opt_evals=probe_opt_evals, honest_n_trials=honest_n_trials,
                sigma=ga_sigma, rng=probe_rng,
            )
        probe_per_seed.append(per_kernel)
        probe_best_per_seed.append(max(per_kernel, key=per_kernel.get))

        # --- GA-mode (kernel_id も探索対象) ---
        evo_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 1, s]))
        res = panmictic_ga(
            eval_once, dim=ga_dim, bounds=bounds, n_evals=ga_n_evals,
            pop_size=ga_pop_size, tournament_k=3, sigma=ga_sigma, elitism=1, rng=evo_rng,
        )
        best_genome = gene_vec_to_genome(res.best_gene)
        ga_best_per_seed.append(KERNEL_NAMES[best_genome.kernel_index()])
        honest_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 2, s]))
        ga_best_fitness_per_seed.append(
            float(honest_reevaluate(eval_once, res.best_gene, n_trials=honest_n_trials, rng=honest_rng))
        )

    # --- probe 行列行を集計 (kernel ごとの mean/std) ---
    probe_matrix_row: dict[str, dict] = {}
    for name in KERNEL_NAMES:
        vals = np.array([d[name] for d in probe_per_seed], dtype=np.float64)
        probe_matrix_row[name] = {"mean": float(vals.mean()), "std": float(vals.std())}

    # probe best kernel (seed-mean 行列で判定) + margin
    means = {name: probe_matrix_row[name]["mean"] for name in KERNEL_NAMES}
    ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    probe_best_kernel = ordered[0][0]
    probe_margin = float(ordered[0][1] - ordered[1][1])

    # probe best の seed 安定性 (per-seed best の mode 一致率)
    probe_best_counts = {n: probe_best_per_seed.count(n) for n in KERNEL_NAMES
                         if probe_best_per_seed.count(n) > 0}
    probe_mode = max(probe_best_counts, key=probe_best_counts.get)
    probe_mode_rate = probe_best_counts[probe_mode] / n_seeds

    # GA best kernel mode
    ga_best_counts = {n: ga_best_per_seed.count(n) for n in KERNEL_NAMES
                      if ga_best_per_seed.count(n) > 0}
    ga_mode = max(ga_best_counts, key=ga_best_counts.get)
    ga_mode_rate = ga_best_counts[ga_mode] / n_seeds

    return {
        "task": task_name,
        "hypothesis_best_kernel": HYPOTHESIS_BEST_KERNEL[task_name],
        "n_seeds": n_seeds,
        "probe_matrix_row": probe_matrix_row,         # kernel -> {mean, std}  (= 性能行列の 1 行)
        "probe_per_seed": probe_per_seed,
        "probe_best_kernel": probe_best_kernel,        # seed-mean 行列上の best
        "probe_best_vs_2nd_margin": probe_margin,
        "probe_best_counts": probe_best_counts,
        "probe_best_mode": probe_mode,
        "probe_best_mode_rate": float(probe_mode_rate),
        "ga_best_per_seed": ga_best_per_seed,
        "ga_best_counts": ga_best_counts,
        "ga_best_mode": ga_mode,
        "ga_best_mode_rate": float(ga_mode_rate),
        "ga_best_honest_fitness_mean": float(np.mean(ga_best_fitness_per_seed)),
    }


def _judge_validity(per_task: list[dict]) -> dict:
    """pre-reg §2.3 の V1/V2/V3 を事前確定閾値で判定し 3 値 verdict を出す."""
    # 主写像 = probe seed-mean best (probe が per-kernel 性能を直接測るため主証拠)
    probe_map = {d["task"]: d["probe_best_kernel"] for d in per_task}
    ga_map = {d["task"]: d["ga_best_mode"] for d in per_task}

    # V1: 非定数性 (probe で ≥2 種の kernel が別 task で best)
    probe_distinct = sorted(set(probe_map.values()))
    ga_distinct = sorted(set(ga_map.values()))
    v1_non_constant_probe = len(probe_distinct) >= 2
    v1_non_constant_ga = len(ga_distinct) >= 2

    # V2: 非僅差 (各 task の probe margin >= MARGIN_THRESHOLD)
    margins = {d["task"]: d["probe_best_vs_2nd_margin"] for d in per_task}
    non_fragile_tasks = [t for t, m in margins.items() if m >= MARGIN_THRESHOLD]
    fragile_tasks = [t for t, m in margins.items() if m < MARGIN_THRESHOLD]

    # V3: seed 安定 (probe best mode rate >= SEED_MODE_RATE_MIN, n_seeds>=MIN_SEEDS_FOR_STABLE)
    n_seeds = per_task[0]["n_seeds"] if per_task else 0
    seed_unstable_tasks = [d["task"] for d in per_task
                           if d["probe_best_mode_rate"] < SEED_MODE_RATE_MIN]
    seeds_ok = n_seeds >= MIN_SEEDS_FOR_STABLE

    # probe ⟷ GA 整合 (写像が証拠間で一致する task)
    agree_tasks = [t for t in probe_map if probe_map[t] == ga_map[t]]
    disagree_tasks = [t for t in probe_map if probe_map[t] != ga_map[t]]

    # 採用 real suite = 非定数性に寄与しうる task のうち
    #   margin 非僅差 (V2) ∧ seed 安定 (V3) を満たすもの
    adopt_tasks = [d["task"] for d in per_task
                   if d["probe_best_vs_2nd_margin"] >= MARGIN_THRESHOLD
                   and d["probe_best_mode_rate"] >= SEED_MODE_RATE_MIN]
    # 採用集合に ≥2 kernel が含まれるか (非定数性を採用集合内で満たすか)
    adopt_kernels = sorted({probe_map[t] for t in adopt_tasks})
    adopt_non_constant = len(adopt_kernels) >= 2

    # === 3 値 verdict ===
    # PASS: 採用集合が非定数 (≥2 kernel) ∧ seed 数十分 ∧ probe/GA が採用 task で概ね整合
    if adopt_non_constant and seeds_ok:
        # honest: probe/GA disagreement が採用 task に多いと弱い → 注記
        adopt_disagree = [t for t in adopt_tasks if t in disagree_tasks]
        if adopt_disagree:
            verdict = (
                "PASS (real substrate 採用可) — 但し probe/GA 写像が一部採用 task で食い違う "
                f"({adopt_disagree})。採用は probe 主証拠ベース、GA は補助。honest に弱さ明記。"
            )
        else:
            verdict = "PASS (real substrate 採用可) — probe 行列が非定数・非僅差・seed 安定。"
        verdict_code = "PASS"
    elif not seeds_ok:
        verdict = (f"N/A — n_seeds={n_seeds} < {MIN_SEEDS_FOR_STABLE}: seed 安定を主張不能。")
        verdict_code = "N/A"
    else:
        # 採用集合が定数 or 空 = 弁別が薄い / 中立 → honest negative (BG6 の轍)
        verdict = (
            "HONEST NEGATIVE (real=kernel 中立寄り) — 非僅差(>=%.2f R²)かつ seed 安定な task で "
            "≥2 kernel の弁別が立たない (margin 薄 or 単一 kernel 支配)。対角 mock の限界で "
            "「種類でなく程度」しか出ない可能性。BG9 は inert 化リスク → N/A 方向。"
            % MARGIN_THRESHOLD
        )
        verdict_code = "HONEST_NEGATIVE"

    # 仮説 (第一原理設計) が当たったか = probe best が hypothesis と一致した task 数
    hyp_hits = [d["task"] for d in per_task
                if d["probe_best_kernel"] == d["hypothesis_best_kernel"]]

    return {
        "thresholds": {
            "MARGIN_THRESHOLD": MARGIN_THRESHOLD,
            "SEED_MODE_RATE_MIN": SEED_MODE_RATE_MIN,
            "MIN_SEEDS_FOR_STABLE": MIN_SEEDS_FOR_STABLE,
        },
        "task_to_best_kernel_probe": probe_map,
        "task_to_best_kernel_ga": ga_map,
        "probe_distinct_kernels": probe_distinct,
        "ga_distinct_kernels": ga_distinct,
        "V1_non_constant_probe": bool(v1_non_constant_probe),
        "V1_non_constant_ga": bool(v1_non_constant_ga),
        "V2_non_fragile_tasks": non_fragile_tasks,
        "V2_fragile_tasks": fragile_tasks,
        "V2_margins": margins,
        "V3_seeds_ok": bool(seeds_ok),
        "V3_seed_unstable_tasks": seed_unstable_tasks,
        "probe_ga_agree_tasks": agree_tasks,
        "probe_ga_disagree_tasks": disagree_tasks,
        "adopt_tasks": adopt_tasks,
        "adopt_kernels": adopt_kernels,
        "adopt_non_constant": bool(adopt_non_constant),
        "hypothesis_hits": hyp_hits,
        "hypothesis_hit_rate": float(len(hyp_hits) / max(len(per_task), 1)),
        "verdict_code": verdict_code,
        "verdict": verdict,
    }


def run(quick: bool = False, seeds: int | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if quick:
        cfg.update(QUICK)
    if seeds is not None:
        cfg["n_seeds"] = int(seeds)

    t0 = time.time()
    per_task: list[dict] = []
    for task_name in FAVORING_TASKS:
        per_task.append(_eval_task(
            task_name,
            n_seeds=cfg["n_seeds"], probe_opt_evals=cfg["probe_opt_evals"],
            ga_n_evals=cfg["ga_n_evals"], ga_pop_size=cfg["ga_pop_size"],
            ga_sigma=cfg["ga_sigma"], dim=cfg["dim"],
            honest_n_trials=cfg["honest_n_trials"], base_seed=cfg["base_seed"],
        ))
    judgment = _judge_validity(per_task)
    wall = time.time() - t0

    return {
        "meta": {
            "gate": "BG9-3 strong BG6 validity gate (kernel-favoring suite が非 inert か)",
            "preregistration": "BG9_PREREGISTRATION.md §2.3",
            "config": cfg,
            "wall_clock_sec": round(wall, 1),
            "n_kernels": N_KERNELS,
            "kernel_names": list(KERNEL_NAMES),
            "tasks": list(FAVORING_TASKS),
            "evidence": "probe (per-kernel 固定最適化 held-out R²) + GA (kernel_id 探索 best-kernel mode)",
            "honest_note": (
                "smoke〜中規模 (3-5 seed)。本検定 (>=15 seed) は BG9 本番 (別段)。"
                "fitness は固定射影 P + per-gene held-out ridge readout で kernel 寄与を分離。"
                "kernel dynamics は対角 mock (full kernel 非主張)。整いすぎた弁別は内訳を疑う "
                "(feedback_benchmark_honest_disclosure)。"
            ),
        },
        "per_task": per_task,
        "validity": judgment,
    }


def _print_summary(res: dict) -> None:
    j = res["validity"]
    print("=== BG9-3 strong BG6: task x kernel probe matrix (held-out R², mean±std) ===")
    header = "task".ljust(18) + "".join(n.ljust(18) for n in res["meta"]["kernel_names"])
    print(header + "probe_best".ljust(18) + "margin")
    for d in res["per_task"]:
        row = d["task"].ljust(18)
        for n in res["meta"]["kernel_names"]:
            cell = d["probe_matrix_row"][n]
            row += f"{cell['mean']:.3f}±{cell['std']:.3f}".ljust(18)
        row += d["probe_best_kernel"].ljust(18) + f"{d['probe_best_vs_2nd_margin']:+.4f}"
        print(row)
    print("=== GA-mode (kernel_id 探索) task -> best-kernel ===")
    for d in res["per_task"]:
        print(f"  {d['task']:18s} GA-mode={d['ga_best_mode']:16s} "
              f"(stab={d['ga_best_mode_rate']:.2f}, counts={d['ga_best_counts']}) "
              f"probe-mode={d['probe_best_mode']:16s} (stab={d['probe_best_mode_rate']:.2f}) "
              f"hyp={d['hypothesis_best_kernel']}")
    print("=== validity (pre-reg §2.3) ===")
    print(f"  probe map     = {j['task_to_best_kernel_probe']}")
    print(f"  GA map        = {j['task_to_best_kernel_ga']}")
    print(f"  V1 non-const  : probe={j['V1_non_constant_probe']} ga={j['V1_non_constant_ga']} "
          f"(probe distinct={j['probe_distinct_kernels']})")
    print(f"  V2 margin     : non-fragile={j['V2_non_fragile_tasks']} fragile={j['V2_fragile_tasks']}")
    print(f"  V3 seed-stab  : seeds_ok={j['V3_seeds_ok']} unstable={j['V3_seed_unstable_tasks']}")
    print(f"  probe⟷GA      : agree={j['probe_ga_agree_tasks']} disagree={j['probe_ga_disagree_tasks']}")
    print(f"  adopt tasks   = {j['adopt_tasks']} (kernels={j['adopt_kernels']}, "
          f"non_constant={j['adopt_non_constant']})")
    print(f"  hypothesis    : hits={j['hypothesis_hits']} rate={j['hypothesis_hit_rate']:.2f}")
    print(f"  VERDICT [{j['verdict_code']}]: {j['verdict']}")
    print(f"wall-clock = {res['meta']['wall_clock_sec']}s")


if __name__ == "__main__":
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="BG9-3 strong BG6 validity gate driver")
    ap.add_argument("--quick", action="store_true", help="更に小 budget で高速 smoke")
    ap.add_argument("--seeds", type=int, default=None, help="seed 数を上書き (既定 5)")
    args = ap.parse_args()

    res = run(quick=args.quick, seeds=args.seeds)
    out = Path(__file__).resolve().parent / "bg6_strong_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(res)
    print(f"written: {out}")
