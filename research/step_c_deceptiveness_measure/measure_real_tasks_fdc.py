# SPDX-License-Identifier: Apache-2.0
"""Phase A CrossMetric: apply the FDC-behavior metric to the SAME 3 real tasks.

== なぜこのファイル ==
``measure_real_tasks.py`` は behavior_elite_fitness_dip メトリックで 3 実 task を測り、
「全 task の dip が d* 相当の raw 値未満寄り (= ③ 不要寄り)」を示した。1 メトリックの結論は
脆いため、**別メトリック (FDC-behavior, metric_fdc_behavior.deceptiveness_with_ci)** で
**完全に同じ task wiring** を測り直し、below/above 判定が再現するか (クロスメトリック合意) を
検証する。結論が割れたらそれ自体が重要な発見。

== 同一性の担保 (循環論法・wiring drift を避ける) ==
本 script は measure_real_tasks.py の task 構築コードを **そのまま複製** し、メトリック呼び出し
だけを差し替える:
  - reservoir tasks: LeakyDelayLineReservoir(n_taps=8, in_dim=2),
    behavior=make_behavior(res), eval=make_eval_once(res, task, 48, 48),
    train_regimes=[VariableDelayRecallTask(D, 0.2, in_dim=2) for D in (15,30)] の TaskMixture,
    flip_flop=FlipFlopTask().
  - step6: exp7 §(A) 3-param ESN, behavior=(rho,leak), deterministic fitness.
FDC の API は ``deceptiveness_with_ci(eval_once, behavior_fn, gene_bounds, dim, rng,
n_samples, honest_n_trials, n_repeats)``。behavior_fn(gene)->vector / eval_once(gene,rng)->float
の規約は reservoir.make_behavior / make_eval_once と完全一致するため、橋渡しは直結でよい。

== 閾値 ==
FDC の校正 (calibrate_fdc_behavior_results.json) から metric_at_dstar (d=0.16 での値) をロード。
behavior_elite_dip と同じ「同一メトリック・同一推定器で測った値が、合成校正で d* に対応した
raw 値を下回るか」という operational 比較。RANK は転送するが CALIBRATED MAGNITUDE は転送しない
(behavior 軸が合成 behavior=mean と異なるため; behavior_elite_dip の honest_disclosure を継承)。

== honest disclosure ==
- 各 task は 5 seed で mean ± std / 95% CI を報告 (単一値で判断しない)。
- FDC は high-d collapse (校正で観測; metric が d>=0.3 で反転) を持つ。実 task の欺瞞性が
  浅いと想定される領域 (d<=0.20, 単調域) では問題ないが、万一 FDC が low/負なら collapse 域の
  両義性に注意。degenerate (分散ゼロ) flag を必ず記録。
- 閾値未満なら「③ 不要」と正直に結論。割れたら割れたと報告 (捏造禁止)。
- 実 task / src / selection_lab は read-only。numpy のみ。CPU 完結。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# measure_real_tasks.py と同一の sys.path 構成 (task module を read-only import)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "step_c_memory_tasks"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask" / "candidates"))
sys.path.insert(0, str(_HERE.parents[0] / "step6_real_proxy"))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from metric_fdc_behavior import deceptiveness_with_ci  # noqa: E402


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_utf8()


def _load_threshold() -> dict:
    """FDC 校正結果から d*=0.16 に対応する metric_at_dstar をロード."""
    p = _HERE / "calibrate_fdc_behavior_results.json"
    info = {
        "metric_at_dstar": None,
        "d_star": None,
        "spearman_with_d": None,
        "spearman_with_d_low_range_le_0p20": None,
        "high_d_collapse_observed": None,
        "source": str(p.name),
    }
    if p.exists():
        c = json.loads(p.read_text(encoding="utf-8"))
        info["metric_at_dstar"] = c.get("metric_at_dstar")
        info["d_star"] = c.get("config", {}).get("d_star")
        info["spearman_with_d"] = c.get("spearman_with_d")
        info["spearman_with_d_low_range_le_0p20"] = c.get(
            "spearman_with_d_low_range_le_0p20")
        info["high_d_collapse_observed"] = c.get("high_d_collapse", {}).get("observed")
    return info


def _measure(name: str, eval_once, behavior, bounds, dim, *,
             n_samples: int, honest_n_trials: int, n_repeats: int,
             rng_seed: int, threshold: float) -> dict:
    """1 task に FDC-behavior を適用し below_threshold 判定込みの dict を返す."""
    rng = np.random.default_rng(rng_seed)
    res = deceptiveness_with_ci(
        eval_once, behavior, bounds, dim, rng,
        n_samples=n_samples, honest_n_trials=honest_n_trials, n_repeats=n_repeats,
    )
    mean = res["deceptiveness_mean"]
    d = {
        "task": name,
        "deceptiveness_mean": mean,
        "deceptiveness_std": res["deceptiveness_std"],
        "ci95_lo": res["ci95_lo"],
        "ci95_hi": res["ci95_hi"],
        "fdc_mean": res["fdc_mean"],
        "fdc_std": res["fdc_std"],
        "any_degenerate": res["any_degenerate"],
        "per_repeat_deceptiveness": [r["deceptiveness"] for r in res["per_repeat"]],
        "per_repeat_fdc": [r["fdc"] for r in res["per_repeat"]],
        "n_samples": n_samples,
        "honest_n_trials": honest_n_trials,
        "n_repeats": n_repeats,
        "metric_at_dstar": threshold,
        "below_threshold": (bool(mean < threshold) if threshold is not None else None),
        "ci_strictly_below": (bool(res["ci95_hi"] < threshold)
                              if threshold is not None else None),
        "ci_strictly_above": (bool(res["ci95_lo"] > threshold)
                              if threshold is not None else None),
    }
    if res["any_degenerate"]:
        d["warning"] = ("degenerate sample(s): fitness or behavior-distance variance ~0; "
                        "FDC set to 0 for those repeats; interpret below/above with care.")
    print(f"  {name}: dec={mean:.4f} ± {d['deceptiveness_std']:.4f} "
          f"95%CI [{d['ci95_lo']:.4f}, {d['ci95_hi']:.4f}] FDC={res['fdc_mean']:+.4f} "
          f"vs d*={threshold:.4f} -> below={d['below_threshold']} "
          f"(CI below={d['ci_strictly_below']}, above={d['ci_strictly_above']}) "
          f"degen={res['any_degenerate']}", flush=True)
    return d


def measure_reservoir_tasks(threshold: float) -> dict:
    """E-A multitask (variable_delay_recall) + flip_flop を同 reservoir で測る."""
    from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,
                           make_eval_once)
    from memory_tasks import FlipFlopTask
    from task_mixture import TaskMixture
    from variable_delay_recall import VariableDelayRecallTask

    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim

    # behavior_elite_dip と比較可能にするため n_samples/honest_n_trials を踏襲。
    # n_repeats=5。確率的 fitness は honest_n_trials=10 で noise 平均化。
    # 環境変数 RAPTOR_FDC_FAST=1 で軽量予算 (CI 完走用; honest に budget を記録)。
    if _FAST:
        n_samples, honest_n, n_repeats = _FAST_NS_RES, _FAST_HT_RES, _FAST_NR
    else:
        n_samples, honest_n, n_repeats = 1600, 10, 5
    out: dict = {}

    print("=== reservoir tasks (behavior = (eff_mem_norm, std(leak))) [FDC-behavior] ===",
          flush=True)
    print(f"    n_samples={n_samples} honest_n_trials={honest_n} n_repeats={n_repeats}",
          flush=True)

    t0 = time.time()
    train_regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=0.2, in_dim=2)
                     for D in (15, 30)]
    ev_ea = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    out["variable_delay_recall"] = _measure(
        "variable_delay_recall", ev_ea, behavior, bounds, dim,
        n_samples=n_samples, honest_n_trials=honest_n, n_repeats=n_repeats,
        rng_seed=12345, threshold=threshold)
    print(f"      ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    ev_ff = make_eval_once(res, FlipFlopTask(), n_train=48, n_eval=48)
    out["flip_flop"] = _measure(
        "flip_flop", ev_ff, behavior, bounds, dim,
        n_samples=n_samples, honest_n_trials=honest_n, n_repeats=n_repeats,
        rng_seed=12345, threshold=threshold)
    print(f"      ({time.time()-t1:.0f}s)", flush=True)

    out["_reservoir_config"] = {"n_samples": n_samples, "honest_n_trials": honest_n,
                                "n_repeats": n_repeats, "rng_seed": 12345}
    return out


def measure_step6_text_proxy(threshold: float) -> dict:
    """step6 ESN×実テキスト next-char proxy (exp7 §(A) と同 wiring; deterministic fitness)."""
    from esn_landscape import ESN, load_corpus, next_char_accuracy

    idx, V, _ = load_corpus(max_chars=24000)
    esn = ESN(n_reservoir=40, vocab=V, seed=0)
    n_train, n_eval, washout = 3000, 1500, 80

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        rho = 0.1 + 1.4 * float(gene[0])
        leak = 0.05 + 0.95 * float(gene[1])
        in_s = 0.3 + 1.2 * float(gene[2])
        return next_char_accuracy(esn, idx, np.array([rho, leak, in_s]),
                                  n_train=n_train, n_eval=n_eval, washout=washout)

    def behavior(gene: np.ndarray) -> np.ndarray:
        return np.array([float(gene[0]), float(gene[1])], dtype=np.float64)

    bounds = (np.zeros(3), np.ones(3))
    # deterministic fitness -> honest_n_trials=1。per-eval ~30-40ms なので n_samples 控えめ。
    if _FAST:
        n_samples, honest_n, n_repeats = _FAST_NS_S6, 1, _FAST_NR
    else:
        n_samples, honest_n, n_repeats = 800, 1, 5

    print("=== step6 text proxy (ESN x real text; behavior=(rho,leak)) [FDC-behavior] ===",
          flush=True)
    print(f"    corpus={len(idx)} chars vocab={V} N=40; n_samples={n_samples} "
          f"honest_n_trials={honest_n} n_repeats={n_repeats}", flush=True)
    t0 = time.time()
    res = _measure("step6_text_proxy", eval_once, behavior, bounds, 3,
                   n_samples=n_samples, honest_n_trials=honest_n, n_repeats=n_repeats,
                   rng_seed=12345, threshold=threshold)
    res["wiring"] = ("exp7 §(A) 3-param ESN: gene[0,1,2]->(rho in[0.1,1.5], leak in[0.05,1.0], "
                     "in_scale in[0.3,1.5]); behavior=(rho,leak); deterministic fitness "
                     "(next-char held-out accuracy), honest_n_trials=1.")
    print(f"      ({time.time()-t0:.0f}s)", flush=True)
    return {"step6_text_proxy": res, "_step6_config": {
        "n_samples": n_samples, "honest_n_trials": honest_n, "n_repeats": n_repeats,
        "rng_seed": 12345}}


def main() -> int:
    thr_info = _load_threshold()
    threshold = thr_info["metric_at_dstar"]
    if threshold is None:
        print("ERROR: could not load metric_at_dstar from FDC calibration; abort.",
              flush=True)
        return 1

    print("Phase A CrossMetric: REAL-task deceptiveness with FDC-behavior "
          f"(metric_at_dstar={threshold:.4f})", flush=True)
    print(f"  calibration: spearman_with_d={thr_info['spearman_with_d']} "
          f"(low-range d<=0.20: {thr_info['spearman_with_d_low_range_le_0p20']}), "
          f"high_d_collapse={thr_info['high_d_collapse_observed']}", flush=True)
    print("=" * 90, flush=True)

    reservoir = measure_reservoir_tasks(threshold)
    step6 = measure_step6_text_proxy(threshold)

    tasks: dict = {}
    tasks.update({k: v for k, v in reservoir.items() if not k.startswith("_")})
    tasks.update({k: v for k, v in step6.items() if not k.startswith("_")})

    below = {k: v["below_threshold"] for k, v in tasks.items()}
    ci_below = {k: v["ci_strictly_below"] for k, v in tasks.items()}
    all_below = all(below.values())

    # behavior_elite_dip の per-task below 判定 (measure_real_tasks_results.json) を読み込み比較
    eid_below = None
    eid_path = _HERE / "measure_real_tasks_results.json"
    if eid_path.exists():
        eid = json.loads(eid_path.read_text(encoding="utf-8"))
        raw = eid.get("conclusion", {}).get("below_threshold_per_task", {})
        # behavior_elite_dip は ea task を 'ea_multitask_variable_delay_recall' と命名 → 揃える
        eid_below = {
            "variable_delay_recall": raw.get("ea_multitask_variable_delay_recall"),
            "flip_flop": raw.get("flip_flop"),
            "step6_text_proxy": raw.get("step6_text_proxy"),
        }

    # クロスメトリック agreement = per-task の below 判定が一致するか
    per_task_agreement = None
    if eid_below is not None:
        per_task_agreement = {
            k: (below.get(k) == eid_below.get(k)) for k in below
        }

    out = {
        "metric": "fdc_behavior",
        "metric_name": "FDC-behavior (Fitness-Distance Correlation in behavior space)",
        "purpose": ("Phase A CrossMetric: re-test the behavior_elite_dip per-task "
                    "below/above-threshold conclusion with an independent metric (FDC-behavior). "
                    "Same task wiring, different metric, to check cross-metric agreement."),
        "threshold_at_d_star": threshold,
        "d_star": thr_info["d_star"],
        "threshold_source": thr_info["source"],
        "calibration": {
            "spearman_with_d": thr_info["spearman_with_d"],
            "spearman_with_d_low_range_le_0p20": thr_info["spearman_with_d_low_range_le_0p20"],
            "high_d_collapse_observed": thr_info["high_d_collapse_observed"],
        },
        "real_tasks": tasks,
        "configs": {
            "reservoir_tasks": reservoir["_reservoir_config"],
            "step6_text_proxy": step6["_step6_config"],
        },
        "conclusion": {
            "all_below_threshold": bool(all_below),
            "below_threshold_per_task": below,
            "ci_strictly_below_per_task": ci_below,
            "verdict": (
                "ALL real tasks measured BELOW FDC metric_at_dstar with the FDC-behavior metric -> "
                "cross-metric consistent with ③ (MAP-Elites niching) being UNNEEDED on these tasks."
                if all_below else
                "At least one real task measured AT/ABOVE FDC metric_at_dstar -> see per-task flags; "
                "③ may be load-bearing there. Inspect CI and compare with behavior_elite_dip."
            ),
        },
        "cross_metric_comparison": {
            "behavior_elite_dip_below_per_task": eid_below,
            "fdc_behavior_below_per_task": below,
            "per_task_agreement": per_task_agreement,
            "all_tasks_agree": (None if per_task_agreement is None
                                else bool(all(per_task_agreement.values()))),
            "note": ("agreement compares the BELOW/ABOVE verdict per task across the two metrics. "
                     "Disagreement is a meaningful finding (which metric / why they diverge)."),
        },
        "honest_disclosure": {
            "axis_mismatch": ("Real-task behavior axes (reservoir: eff_mem/std(leak); step6: rho/leak) "
                              "differ from synthetic behavior=mean. below_threshold is an OPERATIONAL "
                              "same-metric/same-estimator comparison vs the raw FDC value at d*; RANK "
                              "transfers, calibrated MAGNITUDE does not."),
            "fdc_high_d_collapse": ("FDC-behavior INVERTS for d>=0.30 in calibration (operational best- "
                                    "behavior reference relocates). In the deceptive (low-d) regime "
                                    "expected for these smooth tasks this is fine, but a low/negative "
                                    "FDC must not be naively read as deep deception without checking "
                                    "the collapse regime."),
            "estimation_noise": "5 seeds; mean±std + 95% CI reported (no single-seed point claim).",
            "metric_provisional": ("FDC calibration JSON is itself flagged provisional (reproduces_"
                                   "threshold=false; spearman over full sweep=0.4, =1.0 only on d<=0.20). "
                                   "Treat the threshold as operational, not final."),
        },
    }

    out_path = _HERE / "fdc_behavior_crossmetric.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 90, flush=True)
    print(out["conclusion"]["verdict"], flush=True)
    if per_task_agreement is not None:
        print(f"cross-metric per-task agreement vs behavior_elite_dip: {per_task_agreement} "
              f"(all agree = {out['cross_metric_comparison']['all_tasks_agree']})", flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
