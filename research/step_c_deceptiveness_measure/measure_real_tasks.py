# SPDX-License-Identifier: Apache-2.0
"""Apply the BEST-CALIBRATED deceptiveness metric (behavior_elite_fitness_dip) to REAL tasks.

== なぜこのファイル ==
旧 measure_real_tasks.py は別メトリック (downhill_necessity) を使っていた。本ファイルは
metric_behavior_elite_dip.deceptiveness_estimate を **そのまま import 再利用** し、合成 knob
校正 (metric_at_dstar=0.0153, ρ=1, strictly monotone; synth_calibration_results.json) と
**同一メトリック・同一推定器**で実 task を測る。これが定性「指紋」を numeric 測定に格上げする。

== 測定対象 (各 task の NATURAL behavior descriptor を使う) ==
- E-A multitask (variable_delay_recall mixture): reservoir の make_behavior(res)
  = (eff_mem_norm, std(leak)) 2D → metric 内部で PCA 第1主成分に 1D 射影。
- flip_flop: 同 reservoir (in_dim=2) + FlipFlop 単一タスク。同 behavior。
- step6 text proxy (ESN × 実テキスト next-char): exp7 §(A) と同 wiring。
  gene∈[0,1]^3 → (rho,leak,in_scale)。behavior = (rho,leak) niche 軸 (exp7 と同じ)。
  fitness は deterministic (rng 不使用) なので honest_n_trials=1。
  → 旧 calibrate_dip_metric.py は step6 を "SKIPPED (budget)" としたが、deterministic で
     per-eval ~30-40ms と安価に wire できることを実測確認したため、本ファイルでは測定する。

== honest disclosure (synth_calibration_results.json の honest_disclosure を継承) ==
1. axis mismatch: 実 task の behavior 軸 (reservoir = eff_mem/leak、step6 = rho/leak) は
   合成 knob の behavior=mean とは **別軸**。raw dip は metric_at_dstar と同一較正スケールには
   ない (合成は behavior=mean が CLT で b≈0.5 に固着し global peak bin が未到達 →
   metric_at_dstar=0.0153 は約 8x 減衰した shadow)。よって below_threshold は
   「**同一メトリック・同一推定器で測った raw dip が、合成校正で d* に対応した raw 値
   0.0153 を下回るか**」という **operational** な比較であって、実 task の "真の d" を主張する
   ものではない。RANK は転送する (ρ=1) が CALIBRATED MAGNITUDE は転送しない。
2. finite-sample envelope max は上方バイアス。bin あたり期待標本数 >= ~50 を確保し、
   複数 seed の mean±std と t系 95%CI を必ず併記する。1 seed 点推定は信用しない。
3. 2D behavior の PCA 1D 射影は分散最大方向であって「③が効く方向」とは限らない
   (射影が landscape の欺瞞構造を取り落とす可能性)。verdict で明記。

研究隔離: src / selection_lab は read-only import。numpy のみ。CPU 完結。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "step_c_memory_tasks"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask" / "candidates"))
sys.path.insert(0, str(_HERE.parents[0] / "step6_real_proxy"))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from metric_behavior_elite_dip import deceptiveness_estimate  # noqa: E402

# metric_at_dstar = 合成校正で d*=0.16 に対応した raw dip 値 (synth_calibration_results.json)。
METRIC_AT_DSTAR = 0.015281642817850516


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_utf8()


def _summarize(name: str, est) -> dict:
    """DipEstimate を below_threshold 判定込みの dict にまとめる.

    below_threshold は「点推定 mean < metric_at_dstar」。CI 全体が下にあるか
    (= 統計的に有意に下) も別フラグで残す (honest)。
    """
    d = est.as_dict()
    d["metric_at_dstar"] = METRIC_AT_DSTAR
    d["below_threshold"] = bool(est.mean < METRIC_AT_DSTAR)
    # CI 上端が閾値を下回れば「有意に下」(片側)。CI が閾値をまたぐなら結論は弱い。
    d["ci_strictly_below"] = bool(est.ci95_hi < METRIC_AT_DSTAR)
    d["ci_strictly_above"] = bool(est.ci95_lo > METRIC_AT_DSTAR)
    d["task"] = name
    print(f"  {name}: dip={est.mean:.4f} ± {est.std:.4f} "
          f"95%CI [{est.ci95_lo:.4f}, {est.ci95_hi:.4f}]  "
          f"vs d*={METRIC_AT_DSTAR:.4f} -> below={d['below_threshold']} "
          f"(CI strictly below={d['ci_strictly_below']}, strictly above={d['ci_strictly_above']})",
          flush=True)
    return d


def measure_reservoir_tasks() -> dict:
    """E-A multitask + flip_flop を同 reservoir で測る (behavior=make_behavior, PCA 1D)."""
    from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,
                           make_eval_once)
    from memory_tasks import FlipFlopTask
    from task_mixture import TaskMixture
    from variable_delay_recall import VariableDelayRecallTask

    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim

    # 確率的 fitness (ridge held-out R²) → honest_n_trials で decision noise 抑制。
    # bin あたり期待 = n_samples/n_bins。n_samples=1600, n_bins=16 で >=100/bin を確保。
    cfg = dict(n_seeds=5, n_samples=1600, n_bins=16, honest_n_trials=10)
    out: dict = {}

    print("=== reservoir tasks (behavior = (eff_mem_norm, std(leak)) -> PCA 1D) ===", flush=True)
    print(f"    config: {cfg}", flush=True)

    t0 = time.time()
    train_regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=0.2, in_dim=2)
                     for D in (15, 30)]
    ev_ea = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    est_ea = deceptiveness_estimate(ev_ea, behavior, bounds, dim, **cfg)
    out["ea_multitask_variable_delay_recall"] = _summarize(
        "ea_multitask_variable_delay_recall", est_ea)
    print(f"      ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    ev_ff = make_eval_once(res, FlipFlopTask(), n_train=48, n_eval=48)
    est_ff = deceptiveness_estimate(ev_ff, behavior, bounds, dim, **cfg)
    out["flip_flop"] = _summarize("flip_flop", est_ff)
    print(f"      ({time.time()-t1:.0f}s)", flush=True)

    out["_reservoir_config"] = cfg
    return out


def measure_step6_text_proxy() -> dict:
    """step6 ESN×実テキスト next-char proxy を測る (exp7 §(A) と同 wiring).

    gene∈[0,1]^3 → (rho,leak,in_scale)。behavior = (rho,leak) niche 軸 (exp7 と同一)。
    fitness は deterministic → honest_n_trials=1。
    """
    from esn_landscape import ESN, load_corpus, next_char_accuracy

    # exp7 §(A) と同 corpus / reservoir / 評価窓。
    idx, V, _ = load_corpus(max_chars=24000)
    esn = ESN(n_reservoir=40, vocab=V, seed=0)
    n_train, n_eval, washout = 3000, 1500, 80

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        # exp7 _acc_3param と同一 [0,1]->物理パラメータ写像。
        rho = 0.1 + 1.4 * float(gene[0])
        leak = 0.05 + 0.95 * float(gene[1])
        in_s = 0.3 + 1.2 * float(gene[2])
        return next_char_accuracy(esn, idx, np.array([rho, leak, in_s]),
                                  n_train=n_train, n_eval=n_eval, washout=washout)

    def behavior(gene: np.ndarray) -> np.ndarray:
        # exp7 §(A) と同じ niche 軸 = (rho, leak) を表す正規化 gene 成分 [0,1]^2。
        return np.array([float(gene[0]), float(gene[1])], dtype=np.float64)

    bounds = (np.zeros(3), np.ones(3))
    # deterministic fitness → honest_n_trials=1。per-eval ~30-40ms なので n_samples 控えめ。
    cfg = dict(n_seeds=5, n_samples=800, n_bins=16, honest_n_trials=1)

    print("=== step6 text proxy (ESN x real text next-char; behavior=(rho,leak) -> PCA 1D) ===",
          flush=True)
    print(f"    corpus={len(idx)} chars vocab={V} N=40; config: {cfg}", flush=True)
    t0 = time.time()
    est = deceptiveness_estimate(eval_once, behavior, bounds, 3, **cfg)
    res = _summarize("step6_text_proxy", est)
    res["wiring"] = ("exp7 §(A) 3-param ESN: gene[0,1,2]->(rho in[0.1,1.5], leak in[0.05,1.0], "
                     "in_scale in[0.3,1.5]); behavior=(rho,leak); deterministic fitness "
                     "(next-char held-out accuracy), honest_n_trials=1.")
    print(f"      ({time.time()-t0:.0f}s)", flush=True)
    return {"step6_text_proxy": res, "_step6_config": cfg}


def main() -> int:
    print("Measure REAL-task deceptiveness with behavior_elite_fitness_dip "
          f"(metric_at_dstar={METRIC_AT_DSTAR:.4f})", flush=True)
    print("=" * 88, flush=True)

    out: dict = {
        "metric_name": "behavior_elite_fitness_dip",
        "metric_at_dstar": METRIC_AT_DSTAR,
        "d_star": 0.16,
        "calibration_source": "synth_calibration_results.json (ρ=1, strictly monotone, 5 seeds)",
    }

    reservoir = measure_reservoir_tasks()
    step6 = measure_step6_text_proxy()

    tasks = {}
    tasks.update({k: v for k, v in reservoir.items() if not k.startswith("_")})
    tasks.update({k: v for k, v in step6.items() if not k.startswith("_")})
    out["real_tasks"] = tasks
    out["configs"] = {
        "reservoir_tasks": reservoir["_reservoir_config"],
        "step6_text_proxy": step6["_step6_config"],
    }

    # NUMERIC 結論
    below = {k: v["below_threshold"] for k, v in tasks.items()}
    ci_below = {k: v["ci_strictly_below"] for k, v in tasks.items()}
    all_below = all(below.values())
    out["conclusion"] = {
        "all_below_threshold": bool(all_below),
        "below_threshold_per_task": below,
        "ci_strictly_below_per_task": ci_below,
        "verdict": (
            "ALL real tasks measured BELOW metric_at_dstar=0.0153 with the SAME metric -> "
            "numerically consistent with ③ (MAP-Elites niching) being UNNEEDED on these tasks "
            "(upgrades the prior qualitative fingerprint to a measurement)."
            if all_below else
            "At least one real task measured AT/ABOVE metric_at_dstar -> see per-task flags; "
            "③ may be load-bearing there. Inspect CI before concluding."
        ),
    }
    out["honest_disclosure"] = {
        "axis_mismatch": ("Real-task behavior axes (reservoir: eff_mem/std(leak); step6: rho/leak) "
                          "differ from synthetic behavior=mean. below_threshold is an OPERATIONAL "
                          "same-metric/same-estimator comparison of raw dip vs the raw value (0.0153) "
                          "that corresponded to d* in calibration; it is NOT a claim about each "
                          "task's true d. RANK transfers (ρ=1), calibrated MAGNITUDE does not."),
        "envelope_bias": ("finite-sample bin max is upward-biased; >=~50 samples/bin enforced; "
                          "mean±std + t-based 95%CI reported over 5 seeds (no single-seed point claim)."),
        "pca_projection": ("2D->1D PCA picks the max-variance direction, not necessarily the "
                           "deceptive direction; projection may miss niche structure (caveat)."),
        "step6_not_skipped": ("Unlike the earlier calibrate_dip_metric.py note, step6 is wired and "
                              "measured here: deterministic fitness, ~30-40ms/eval, cheap to sample."),
    }

    path = _HERE / "measure_real_tasks_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 88, flush=True)
    print(out["conclusion"]["verdict"], flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
