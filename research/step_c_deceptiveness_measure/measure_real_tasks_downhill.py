# SPDX-License-Identifier: Apache-2.0
"""Phase A CrossMetric #3: apply DOWNHILL-NECESSITY to the 3 REAL tasks.

Cross-validate the behavior_elite_dip / fdc_behavior conclusion ("real tasks are all
below_threshold = smooth = phase-3 niching unneeded") with a THIRD metric.
A 1-2 metric conclusion is fragile; this checks whether downhill_necessity agrees.

== Comparability design (same footing as the other two metrics) ==
- downhill_necessity was calibrated on the synthetic knob (exp_knob_sweep.make_corridor_eval),
  behavior=mean = 1D. Existing calibration_results.json shows it is NOISY/NON-monotone
  (spearman_with_d=0.458, metric_at_dstar=0.722, smooth d<=0.10 reads EXACTLY 0.0).
  This file RE-RUNS calibration on the synthetic knob with the same grid as the real-task
  application (PCA-1D path, n_bins, fitness_trials) so the threshold and the real-task
  numbers are produced by an IDENTICAL estimator+grid pipeline.
- Real-task behavior is 2D (reservoir=(eff_mem,std(leak)); step6=(rho,leak)). We project it
  to PCA 1st PC BEFORE downhill_necessity, matching exactly how metric_behavior_elite_dip
  (_project_1d) and metric_fdc_behavior treat real tasks, AND matching the 1D synthetic grid.
  -> same metric, same estimator, same grid dimensionality = honest operational comparison.

== honest disclosure (inherited + downhill-specific) ==
1. axis mismatch / magnitude non-transfer: real-task behavior axes differ from synthetic
   behavior=mean. below_threshold = operational same-metric/same-estimator/same-grid-dim
   comparison of raw value vs the raw value that corresponded to d* in calibration. RANK
   transfers, MAGNITUDE does not. NOT a claim about each task's true d.
2. PCA 1D projection = max-variance direction, not necessarily the deceptive direction.
3. sampling noise: report mean +/- std and t-95%CI over seeds; no single-seed point claim.
4. NOT independent: shares the sampling population and PCA-1D style with the other two
   metrics; systematic sampling error can be shared. "3-metric agreement" = different
   geometric indicators pointing the same way, not 3 independent proofs.
5. downhill grid sparsity: docstring of metric_downhill_necessity warns sparse/disconnected
   occupied cells artificially lower reach (raise deceptiveness). 1D + dense grid mitigates.

research isolation: src / selection_lab / real-task code = read-only import (NOT modified).
New files only inside step_c_deceptiveness_measure/. numpy only. CPU only.
"""
from __future__ import annotations

import json
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "step_c_memory_tasks"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask" / "candidates"))
sys.path.insert(0, str(_HERE.parents[0] / "step6_real_proxy"))
sys.path.insert(0, str(_HERE.parents[0] / "step_c_applicability"))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from metric_downhill_necessity import deceptiveness_with_ci  # noqa: E402
from exp_knob_sweep import D as SYNTH_D  # noqa: E402
from exp_knob_sweep import behavior_mean, make_corridor_eval  # noqa: E402

# Shared grid for calibration AND real-task application (identical pipeline).
N_BINS = 24       # = exp_knob_sweep MAP-Elites grid; calibration_results.json also used 24.
D_STAR = 0.16     # the dip-depth knob value where phase-3 became load-bearing (exp_knob_sweep).


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_utf8()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = np.sqrt((xr**2).sum() * (yr**2).sum())
    return float((xr * yr).sum() / denom) if denom > 0 else 0.0


def _t95_half(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    tbl = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}
    t = tbl.get(n, 1.96)
    return t * std / sqrt(n)


# ---------------------------------------------------------------------------
# PCA 1D projection adapter: project a 2D (or 1D) real-task behavior_fn to its
# 1st principal component, normalized to [0,1] so downhill's behavior_bounds=[0,1]
# grid sees the same geometry the synthetic 1D calibration sees.
# Mirrors metric_behavior_elite_dip._project_1d.
# ---------------------------------------------------------------------------


def _make_pca1d_behavior(behavior_fn, gene_bounds, dim, *, n_probe, rng):
    lo, hi = gene_bounds
    probe = lo + (hi - lo) * rng.random((n_probe, dim))
    behs = np.array([np.atleast_1d(np.asarray(behavior_fn(g), dtype=np.float64))
                     for g in probe])
    mean_vec = behs.mean(axis=0)
    if behs.shape[1] == 1:
        pc1 = np.array([1.0])
    else:
        _u, _s, vt = np.linalg.svd(behs - mean_vec, full_matrices=False)
        pc1 = vt[0]
    proj = (behs - mean_vec) @ pc1
    p_lo, p_hi = float(proj.min()), float(proj.max())
    span = max(p_hi - p_lo, 1e-12)

    def behavior1d(gene: np.ndarray) -> np.ndarray:
        b = np.atleast_1d(np.asarray(behavior_fn(gene), dtype=np.float64))
        v = float((b - mean_vec) @ pc1)
        return np.array([(v - p_lo) / span], dtype=np.float64)

    return behavior1d, (np.zeros(1), np.ones(1)), {
        "pc1": [round(x, 4) for x in np.atleast_1d(pc1).tolist()],
        "proj_lo": p_lo, "proj_hi": p_hi,
    }


def _estimate_real(name, eval_once, behavior_fn, gene_bounds, dim, *,
                   n_seeds, n_samples, fitness_trials, base_seed=20260530):
    """Apply downhill_necessity to a real task via PCA-1D projection; per-seed -> mean+/-CI."""
    seed_vals, proj_info = [], None
    for s in range(n_seeds):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, 13, s]))
        b1d, bb1d, info = _make_pca1d_behavior(
            behavior_fn, gene_bounds, dim, n_probe=max(400, n_samples // 2), rng=rng)
        if proj_info is None:
            proj_info = info
        mean, _se, _samp = deceptiveness_with_ci(
            eval_once, b1d, gene_bounds, dim, rng,
            n_samples=n_samples, n_bins=N_BINS, behavior_bounds=bb1d,
            fitness_trials=fitness_trials, n_repeats=1)
        seed_vals.append(mean)
    arr = np.array(seed_vals, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = _t95_half(std, len(arr))
    return {"task": name, "mean": mean, "std": std,
            "ci95_lo": mean - half, "ci95_hi": mean + half,
            "samples": [round(x, 4) for x in seed_vals],
            "n_seeds": n_seeds, "n_samples": n_samples, "n_bins": N_BINS,
            "fitness_trials": fitness_trials, "pca_proj": proj_info}


def _tag(d: dict, metric_at_dstar: float) -> dict:
    d["metric_at_dstar"] = metric_at_dstar
    d["below_threshold"] = bool(d["mean"] < metric_at_dstar)
    d["ci_strictly_below"] = bool(d["ci95_hi"] < metric_at_dstar)
    d["ci_strictly_above"] = bool(d["ci95_lo"] > metric_at_dstar)
    return d


# ---------------------------------------------------------------------------
# step 1: re-run synthetic-knob calibration with the SAME estimator+grid path.
# ---------------------------------------------------------------------------


def reconfirm_calibration(*, n_seeds=3, n_samples=4000, base_seed=20260530):
    d_levels = [0.0, 0.05, 0.1, 0.13, 0.16, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    gene_bounds = (np.zeros(SYNTH_D), np.ones(SYNTH_D))
    print("=== reconfirm downhill_necessity calibration on synthetic knob ===", flush=True)
    print(f"    D={SYNTH_D} n_seeds={n_seeds} n_samples={n_samples} n_bins={N_BINS}", flush=True)
    per_level, means = [], []
    for d in d_levels:
        eval_once = make_corridor_eval(d)
        sm = []
        for s in range(n_seeds):
            rng = np.random.default_rng(
                np.random.SeedSequence([base_seed, 11, s, int(d * 1000)]))
            mean, _se, _samp = deceptiveness_with_ci(
                eval_once, behavior_mean, gene_bounds, SYNTH_D, rng,
                n_samples=n_samples, n_bins=N_BINS,
                behavior_bounds=(np.zeros(1), np.ones(1)),
                fitness_trials=8, n_repeats=1)
            sm.append(mean)
        lvl = float(np.mean(sm))
        means.append(lvl)
        per_level.append({"d": d, "mean": lvl,
                          "se": float(np.std(sm, ddof=1) / sqrt(n_seeds)) if n_seeds > 1 else 0.0,
                          "samples": [round(x, 4) for x in sm]})
        print(f"    d={d:.2f}: downhill={lvl:.4f}", flush=True)
    d_arr, m_arr = np.array(d_levels), np.array(means)
    rho = _spearman(d_arr, m_arr)
    metric_at_dstar = float(np.interp(D_STAR, d_arr, m_arr))
    monotone = bool(np.all(np.diff(m_arr) >= -1e-9))
    print(f"    spearman_vs_d={rho:.4f} metric_at_dstar(d*={D_STAR})={metric_at_dstar:.4f} "
          f"monotone={monotone}", flush=True)
    return {"spearman_vs_d": rho, "metric_at_dstar": metric_at_dstar, "monotone": monotone,
            "d_levels": d_levels, "per_level": per_level,
            "n_seeds": n_seeds, "n_samples": n_samples, "n_bins": N_BINS}


# ---------------------------------------------------------------------------
# step 2: 3 real tasks
# ---------------------------------------------------------------------------


def measure_reservoir_tasks(metric_at_dstar):
    from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,
                           make_eval_once)
    from memory_tasks import FlipFlopTask
    from task_mixture import TaskMixture
    from variable_delay_recall import VariableDelayRecallTask

    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim
    # honest budget: per-eval ~8.7ms (stochastic ridge R2). 900 samples / 24 bins (1D,
    # center-concentrated) keeps >=~37/bin; fitness_trials=6 suppresses decision noise;
    # 3 seeds meets the 3+ seed requirement. ~145s/task (vs 731s at 1600x10x5 = timeout).
    cfg = dict(n_seeds=3, n_samples=900, fitness_trials=6)
    out = {}
    print("=== reservoir tasks (behavior 2D -> PCA 1D -> downhill) ===", flush=True)
    print(f"    config: {cfg} n_bins={N_BINS}", flush=True)

    t0 = time.time()
    train = [VariableDelayRecallTask(seq_len=Dly, distractor_amp=0.2, in_dim=2)
             for Dly in (15, 30)]
    ev = make_eval_once(res, TaskMixture(train), n_train=48, n_eval=48)
    d_ea = _tag(_estimate_real("variable_delay_recall", ev, behavior, bounds, dim, **cfg),
                metric_at_dstar)
    out["variable_delay_recall"] = d_ea
    print(f"    variable_delay_recall: {d_ea['mean']:.4f} "
          f"CI[{d_ea['ci95_lo']:.4f},{d_ea['ci95_hi']:.4f}] below={d_ea['below_threshold']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    ev = make_eval_once(res, FlipFlopTask(), n_train=48, n_eval=48)
    d_ff = _tag(_estimate_real("flip_flop", ev, behavior, bounds, dim, **cfg), metric_at_dstar)
    out["flip_flop"] = d_ff
    print(f"    flip_flop: {d_ff['mean']:.4f} CI[{d_ff['ci95_lo']:.4f},{d_ff['ci95_hi']:.4f}] "
          f"below={d_ff['below_threshold']} ({time.time()-t1:.0f}s)", flush=True)

    out["_config"] = {**cfg, "n_bins": N_BINS}
    return out


def measure_step6(metric_at_dstar):
    from esn_landscape import ESN, load_corpus, next_char_accuracy
    idx, V, _ = load_corpus(max_chars=24000)
    esn = ESN(n_reservoir=40, vocab=V, seed=0)
    n_train, n_eval, washout = 3000, 1500, 80

    def eval_once(gene, rng):
        rho = 0.1 + 1.4 * float(gene[0])
        leak = 0.05 + 0.95 * float(gene[1])
        in_s = 0.3 + 1.2 * float(gene[2])
        return next_char_accuracy(esn, idx, np.array([rho, leak, in_s]),
                                  n_train=n_train, n_eval=n_eval, washout=washout)

    def behavior(gene):
        return np.array([float(gene[0]), float(gene[1])], dtype=np.float64)

    # deterministic fitness (ESN next-char) -> fitness_trials=1; ~30-40ms/eval. 3 seeds.
    cfg = dict(n_seeds=3, n_samples=800, fitness_trials=1)
    print("=== step6 text proxy (ESN; behavior=(rho,leak) -> PCA 1D -> downhill) ===", flush=True)
    print(f"    corpus={len(idx)} vocab={V} N=40; config: {cfg} n_bins={N_BINS}", flush=True)
    t0 = time.time()
    d6 = _tag(_estimate_real("step6_text_proxy", eval_once, behavior,
                             (np.zeros(3), np.ones(3)), 3, **cfg), metric_at_dstar)
    d6["wiring"] = ("exp7 3-param ESN: gene->(rho[0.1,1.5],leak[0.05,1.0],in_scale[0.3,1.5]); "
                    "behavior=(rho,leak)->PCA1D; deterministic fitness, fitness_trials=1.")
    print(f"    step6_text_proxy: {d6['mean']:.4f} CI[{d6['ci95_lo']:.4f},{d6['ci95_hi']:.4f}] "
          f"below={d6['below_threshold']} ({time.time()-t0:.0f}s)", flush=True)
    return {"step6_text_proxy": d6, "_config": {**cfg, "n_bins": N_BINS}}


def main():
    print("Phase A CrossMetric #3: DOWNHILL-NECESSITY on REAL tasks", flush=True)
    print("=" * 88, flush=True)

    calib = reconfirm_calibration(n_seeds=3, n_samples=4000)
    metric_at_dstar = calib["metric_at_dstar"]

    reservoir = measure_reservoir_tasks(metric_at_dstar)
    step6 = measure_step6(metric_at_dstar)

    tasks = {}
    tasks.update({k: v for k, v in reservoir.items() if not k.startswith("_")})
    tasks.update({k: v for k, v in step6.items() if not k.startswith("_")})

    below = {k: v["below_threshold"] for k, v in tasks.items()}
    ci_below = {k: v["ci_strictly_below"] for k, v in tasks.items()}
    all_below = all(below.values())

    eb_all_below = None
    try:
        eb = json.loads((_HERE / "measure_real_tasks_results.json").read_text(encoding="utf-8"))
        eb_all_below = bool(eb.get("conclusion", {}).get("all_below_threshold"))
    except Exception:
        eb_all_below = None
    agrees = (eb_all_below is not None and bool(all_below) == eb_all_below)

    out = {
        "metric_name": "downhill_necessity",
        "metric_at_dstar": metric_at_dstar,
        "d_star": D_STAR,
        "calibration_reconfirm": calib,
        "calibration_existing_file": "calibration_results.json",
        "real_tasks": tasks,
        "configs": {"reservoir_tasks": reservoir["_config"], "step6_text_proxy": step6["_config"]},
        "conclusion": {
            "all_below_threshold": bool(all_below),
            "below_threshold_per_task": below,
            "ci_strictly_below_per_task": ci_below,
            "behavior_elite_dip_all_below": eb_all_below,
            "agrees_with_behavior_elite_dip": agrees,
        },
        "honest_disclosure": {
            "operational_comparison": ("below_threshold = same-metric/same-estimator/same-grid-dim "
                                       "(1D) comparison of raw downhill vs the raw value at d*=0.16 in "
                                       "synthetic calibration. RANK transfers, MAGNITUDE does not."),
            "pca_projection": ("real-task 2D behavior -> PCA 1st PC before downhill grid; max-variance "
                               "direction may not be the deceptive direction."),
            "not_fully_independent": ("shares sampling population + PCA-1D style with the other two "
                                      "metrics; '3-metric agreement' != 3 independent proofs."),
            "downhill_grid_sparsity": ("sparse/disconnected occupied cells can artificially raise "
                                       "deceptiveness; 1D + n_bins=24 keeps the grid dense."),
            "calibration_noise": ("existing calibration_results.json had spearman_with_d=0.458, "
                                  "monotone_strict=false, and EXACT 0.0 for smooth d<=0.10 then a jump "
                                  "to ~0.72 -- downhill is a noisier/non-monotone instrument than "
                                  "fdc_behavior (rho=1). Interpret with care; see calibration_reconfirm."),
        },
    }

    path = _HERE / "downhill_necessity_crossmetric.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 88, flush=True)
    print(f"all_below_threshold={all_below}  agrees_with_behavior_elite_dip={agrees} "
          f"(eb_all_below={eb_all_below})", flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
