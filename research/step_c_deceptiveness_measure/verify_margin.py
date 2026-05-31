# -*- coding: utf-8 -*-
"""
verify_margin.py -- Phase B adversarial verification, lens=sampling_threshold_margin.

Uses the ACTUAL metric (metric_behavior_elite_dip.deceptiveness_estimate) and the
ACTUAL real-task wiring (reservoir make_behavior, step6 ESN), re-measured under
seed/sample/bin/threshold perturbations.

Tests:
 T1 metric_at_dstar (threshold) stability: re-derive the d*=0.16 raw value across
    base_seed / n_seeds / n_samples / n_bins. Is 0.0153 stable or seed-noise?
 T2 real-task value stability: re-measure each task across base_seed / n_samples /
    n_bins; report mean/std and the margin to the threshold in std units.
 T3 verdict robustness: under each perturbation, does below/above flip?
 T4 honest summary of what the on-disk conclusion actually is.

ASCII-only output. Writes verify_margin_results.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "step_c_applicability"))
sys.path.insert(0, str(HERE.parents[0] / "step_c_memory_tasks"))
sys.path.insert(0, str(HERE.parents[0] / "ea_multitask"))
sys.path.insert(0, str(HERE.parents[0] / "ea_multitask" / "candidates"))
sys.path.insert(0, str(HERE.parents[0] / "step6_real_proxy"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

for s in (sys.stdout, sys.stderr):
    rc = getattr(s, "reconfigure", None)
    if rc:
        try: rc(encoding="utf-8")
        except Exception: pass

from metric_behavior_elite_dip import deceptiveness, deceptiveness_estimate  # noqa


def synth_eval_behavior():
    from exp_knob_sweep import D as SYN_D, behavior_mean, make_corridor_eval
    return SYN_D, behavior_mean, make_corridor_eval


def reservoir_setup():
    from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,
                           make_eval_once)
    from memory_tasks import FlipFlopTask
    from task_mixture import TaskMixture
    from variable_delay_recall import VariableDelayRecallTask
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res); behavior = make_behavior(res); dim = res.gene_dim
    train_regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=0.2, in_dim=2)
                     for D in (15, 30)]
    ev_vdr = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    ev_ff = make_eval_once(res, FlipFlopTask(), n_train=48, n_eval=48)
    return bounds, behavior, dim, ev_vdr, ev_ff


def step6_setup():
    from esn_landscape import ESN, load_corpus, next_char_accuracy
    idx, V, _ = load_corpus(max_chars=24000)
    esn = ESN(n_reservoir=40, vocab=V, seed=0)
    n_train, n_eval, washout = 3000, 1500, 80
    def eval_once(gene, rng):
        rho = 0.1 + 1.4 * float(gene[0]); leak = 0.05 + 0.95 * float(gene[1])
        in_s = 0.3 + 1.2 * float(gene[2])
        return next_char_accuracy(esn, idx, np.array([rho, leak, in_s]),
                                  n_train=n_train, n_eval=n_eval, washout=washout)
    def behavior(gene):
        return np.array([float(gene[0]), float(gene[1])], dtype=np.float64)
    return (np.zeros(3), np.ones(3)), behavior, 3, eval_once


def main():
    out = {"lens": "sampling_threshold_margin", "ts": time.strftime("%Y-%m-%d %H:%M")}

    SYN_D, behavior_mean, make_corridor_eval = synth_eval_behavior()
    ev_dstar = make_corridor_eval(0.16)
    sbounds = (np.zeros(SYN_D), np.ones(SYN_D))

    # ---- T1: threshold (metric_at_dstar) stability ----
    t1 = []
    base_cfg = dict(n_samples=4000, n_bins=24, honest_n_trials=1)
    for bs in (20260531, 1, 777, 999999):
        est = deceptiveness_estimate(ev_dstar, behavior_mean, sbounds, SYN_D,
                                     n_seeds=5, base_seed=bs, **base_cfg)
        t1.append({"vary": "base_seed", "base_seed": bs, "mean": round(est.mean, 5),
                   "std": round(est.std, 5), "per_seed": [round(x, 4) for x in est.per_seed]})
    for ns in (3, 5, 10):
        est = deceptiveness_estimate(ev_dstar, behavior_mean, sbounds, SYN_D,
                                     n_seeds=ns, base_seed=20260531, **base_cfg)
        t1.append({"vary": "n_seeds", "n_seeds": ns, "mean": round(est.mean, 5),
                   "std": round(est.std, 5)})
    for nsamp in (2000, 4000, 8000):
        est = deceptiveness_estimate(ev_dstar, behavior_mean, sbounds, SYN_D,
                                     n_seeds=5, base_seed=20260531,
                                     n_samples=nsamp, n_bins=24, honest_n_trials=1)
        t1.append({"vary": "n_samples", "n_samples": nsamp, "mean": round(est.mean, 5),
                   "std": round(est.std, 5)})
    for nb in (16, 24, 32):
        est = deceptiveness_estimate(ev_dstar, behavior_mean, sbounds, SYN_D,
                                     n_seeds=5, base_seed=20260531,
                                     n_samples=4000, n_bins=nb, honest_n_trials=1)
        t1.append({"vary": "n_bins", "n_bins": nb, "mean": round(est.mean, 5),
                   "std": round(est.std, 5)})
    out["T1_threshold_stability"] = t1
    thr_vals = [r["mean"] for r in t1]
    out["T1_threshold_range"] = {"min": round(min(thr_vals), 5), "max": round(max(thr_vals), 5),
                                 "stored_metric_at_dstar": 0.015281642817850516,
                                 "spread_factor": round(max(thr_vals) / max(min(thr_vals), 1e-9), 2)}

    # ---- T2/T3: real-task value + verdict stability vs the stored threshold ----
    THR = 0.015281642817850516
    rbounds, rbeh, rdim, ev_vdr, ev_ff = reservoir_setup()
    real = {}
    rcfg = dict(n_samples=1600, n_bins=16, honest_n_trials=10)
    for name, ev in (("variable_delay_recall", ev_vdr), ("flip_flop", ev_ff)):
        rows = []
        for bs in (20260531, 1, 777):
            est = deceptiveness_estimate(ev, rbeh, rbounds, rdim,
                                         n_seeds=5, base_seed=bs, **rcfg)
            margin_std = (est.mean - THR) / (est.std + 1e-12)
            rows.append({"base_seed": bs, "mean": round(est.mean, 5), "std": round(est.std, 5),
                         "ci95_lo": round(est.ci95_lo, 5), "ci95_hi": round(est.ci95_hi, 5),
                         "below_thr": bool(est.mean < THR),
                         "margin_to_thr_in_std": round(float(margin_std), 2),
                         "ci_brackets_thr": bool(est.ci95_lo < THR < est.ci95_hi)})
        real[name] = rows
    # step6 cheaper
    sbounds6, sbeh6, sdim6, ev6 = step6_setup()
    rows6 = []
    for bs in (20260531, 1, 777):
        est = deceptiveness_estimate(ev6, sbeh6, sbounds6, sdim6,
                                     n_seeds=5, base_seed=bs,
                                     n_samples=800, n_bins=16, honest_n_trials=1)
        margin_std = (est.mean - THR) / (est.std + 1e-12)
        rows6.append({"base_seed": bs, "mean": round(est.mean, 5), "std": round(est.std, 5),
                      "ci95_lo": round(est.ci95_lo, 5), "ci95_hi": round(est.ci95_hi, 5),
                      "below_thr": bool(est.mean < THR),
                      "margin_to_thr_in_std": round(float(margin_std), 2),
                      "ci_brackets_thr": bool(est.ci95_lo < THR < est.ci95_hi)})
    real["step6_text_proxy"] = rows6
    out["T2_real_task_stability"] = real

    # verdict flip summary
    flip = {}
    for name, rows in real.items():
        verdicts = set(r["below_thr"] for r in rows)
        flip[name] = {"below_set": sorted(str(v) for v in verdicts),
                      "verdict_flips_across_seeds": len(verdicts) > 1,
                      "any_ci_brackets_thr": any(r["ci_brackets_thr"] for r in rows)}
    out["T3_verdict_flip"] = flip

    (HERE / "verify_margin_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE verify_margin_results.json")


if __name__ == "__main__":
    main()
