# -*- coding: utf-8 -*-
"""
verify_sampling_margin.py -- Phase B adversarial verification.
Lens: sampling_threshold_margin (+ circular-logic / descriptor-validity).

The default conclusion under test:
  "All real tasks below_threshold (smooth), metric=behavior_elite_dip,
   real max=0.041, d*=0.1234 -> 3 (deception layer) unnecessary."

We attack it on four axes; everything ASCII-only for cp932 console safety.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(r"D:\projects\llcore\research\step_c_deceptiveness_measure")
sys.path.insert(0, str(HERE))

from metric_behavior_elite_dip import behavior_elite_dip  # noqa: E402

TASK_SOURCES = {
    "flip_flop": HERE.parent / "step_c_memory_tasks" / "corridor_results.json",
    "variable_delay_recall": HERE.parent / "ea_multitask" / "multitask_results.json",
    "step6_text_proxy": HERE.parent / "step6_real_proxy" / "proxy_results.json",
}

# ---- the exact descriptor logic from measure_real_tasks.py ----
DESC_KEYS_PRODUCTION = ("gen", "mean_fitness", "diversity", "best_fitness")


def build_traj(gens, keys):
    traj = []
    for g in gens:
        if not isinstance(g, dict):
            continue
        fit = g.get("best_fitness", g.get("fitness", 0.0))
        beh = [float(g[k]) for k in keys if isinstance(g.get(k), (int, float))]
        if not beh:
            beh = [float(fit)]
        traj.append((float(fit), np.atleast_1d(np.asarray(beh, dtype=float))))
    return traj


def load_gens(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("generations", "trajectory", "history", "elites", "winner_trajectory"):
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            return data[key]
    return None


def main():
    out = {"lens": "sampling_threshold_margin", "tests": {}}

    # =====================================================================
    # TEST 1 -- CIRCULAR-LOGIC / DESCRIPTOR-VALIDITY (the core attack)
    # The production descriptor injects 'gen' (0,1,2,...,T) which is monotonic
    # and numerically dominant. behavior_elite_dip measures ||b_t - b_final||;
    # if one coordinate is a monotone ramp it CANNOT dip -> mechanically forces
    # a near-zero "smooth" reading regardless of real deceptiveness.
    # =====================================================================
    t1 = {}
    for name, path in TASK_SOURCES.items():
        gens = load_gens(path)
        if not gens:
            t1[name] = {"error": "no gens"}
            continue
        sample_keys = list(gens[0].keys())
        # value with production descriptor (includes gen)
        v_prod = behavior_elite_dip(build_traj(gens, DESC_KEYS_PRODUCTION))
        # value WITHOUT gen
        v_nogen = behavior_elite_dip(build_traj(gens, ("mean_fitness", "diversity", "best_fitness")))
        # value with ONLY a real behavioral coordinate (diversity), no monotone fitness
        v_div = behavior_elite_dip(build_traj(gens, ("diversity",))) if any("diversity" in g for g in gens) else None
        # value with gen-only (pure monotone ramp) -> should be ~0 by construction
        v_genonly = behavior_elite_dip(build_traj(gens, ("gen",))) if any("gen" in g for g in gens) else None
        # how much of the descriptor magnitude is 'gen'?
        gen_vals = [g.get("gen") for g in gens if isinstance(g.get("gen"), (int, float))]
        other_keys = [k for k in DESC_KEYS_PRODUCTION if k != "gen"]
        other_ranges = {}
        for k in other_keys:
            xs = [g[k] for g in gens if isinstance(g.get(k), (int, float))]
            if xs:
                other_ranges[k] = round(float(max(xs) - min(xs)), 4)
        gen_range = round(float(max(gen_vals) - min(gen_vals)), 4) if gen_vals else None
        t1[name] = {
            "sample_keys": sample_keys,
            "v_production(with_gen)": round(float(v_prod), 4),
            "v_without_gen": round(float(v_nogen), 4),
            "v_diversity_only": round(float(v_div), 4) if v_div is not None else None,
            "v_gen_only_ramp": round(float(v_genonly), 4) if v_genonly is not None else None,
            "gen_range": gen_range,
            "other_coord_ranges": other_ranges,
        }
    out["tests"]["1_circular_descriptor"] = t1

    # =====================================================================
    # TEST 2 -- MARGIN vs SEED NOISE on the REAL trajectories.
    # The real tasks are stored single trajectories. We bootstrap-resample the
    # per-generation records (block bootstrap) to estimate the metric's own
    # sampling variance, and ask: is the gap to d*=0.1234 robust to that noise?
    # We do this for the production descriptor (the one that fed the conclusion).
    # =====================================================================
    rng = np.random.default_rng(12345)
    t2 = {}
    d_star = 0.1234
    for name, path in TASK_SOURCES.items():
        gens = load_gens(path)
        if not gens:
            continue
        base = build_traj(gens, DESC_KEYS_PRODUCTION)
        n = len(base)
        boots = []
        for _ in range(400):
            # block bootstrap preserving order: jitter by resampling contiguous order
            idx = np.sort(rng.integers(0, n, size=n))
            sub = [base[i] for i in idx]
            if len(sub) >= 3:
                boots.append(behavior_elite_dip(sub))
        boots = np.array(boots)
        point = behavior_elite_dip(base)
        t2[name] = {
            "point_value": round(float(point), 4),
            "boot_mean": round(float(boots.mean()), 4),
            "boot_std": round(float(boots.std()), 4),
            "boot_p95": round(float(np.percentile(boots, 95)), 4),
            "boot_max": round(float(boots.max()), 4),
            "d_star": d_star,
            "margin_to_dstar": round(float(d_star - point), 4),
            "margin_in_boot_std": round(float((d_star - point) / (boots.std() + 1e-12)), 2),
            "frac_boot_above_dstar": round(float((boots > d_star).mean()), 3),
        }
    out["tests"]["2_real_margin_bootstrap"] = t2

    # =====================================================================
    # TEST 3 -- THRESHOLD (d*) STABILITY from the SYNTHETIC corridor.
    # Recompute d* with the richer descriptor (calibrate_dip_metric.behavior_of)
    # across seed banks / k_sigma / n_seeds / resolution. Is 0.1234 stable, and
    # is it even a computed number (the stored JSON looks like a placeholder)?
    # =====================================================================
    from calibrate_dip_metric import make_corridor_eval, behavior_of  # noqa

    def run_ea_synth(eval_fn, dim=24, pop=48, gens=40, seed=0):
        r = np.random.default_rng(seed)
        pop_x = r.normal(0, 1, (pop, dim))
        traj = []
        for _ in range(gens):
            fits = np.array([eval_fn(x) for x in pop_x])
            order = np.argsort(fits)[::-1]
            pop_x = pop_x[order]; fits = fits[order]
            traj.append((float(fits[0]), behavior_of(pop_x[0].copy())))
            mu = max(1, pop // 4)
            parents = pop_x[:mu]
            pop_x = np.array([parents[r.integers(mu)] + r.normal(0, 0.2, dim) for _ in range(pop)])
        return traj

    def calib_dstar(n_d=11, n_seeds=8, k_sigma=3.0, seed_off=0):
        ds = np.linspace(0, 1, n_d)
        means = []
        noise0 = None
        for i, d in enumerate(ds):
            ev = make_corridor_eval(d)
            vals = [behavior_elite_dip(run_ea_synth(ev, seed=s + seed_off)) for s in range(n_seeds)]
            means.append(float(np.mean(vals)))
            if i == 0:
                noise0 = np.array(vals)
        thr = float(noise0.mean() + k_sigma * noise0.std())
        dstar = next((float(d) for d, m in zip(ds, means) if m > thr), None)
        return {"d_star": dstar, "threshold": round(thr, 4),
                "noise_mean": round(float(noise0.mean()), 4),
                "noise_std": round(float(noise0.std()), 4),
                "means": [round(m, 4) for m in means]}

    t3 = {"seed_banks": [], "k_sigma": [], "n_seeds": [], "resolution": []}
    for so in (0, 100, 200, 300):
        r = calib_dstar(seed_off=so)
        t3["seed_banks"].append({"seed_off": so, "d_star": r["d_star"], "threshold": r["threshold"],
                                 "noise_std": r["noise_std"]})
    for k in (2.0, 2.5, 3.0, 3.5):
        r = calib_dstar(k_sigma=k)
        t3["k_sigma"].append({"k_sigma": k, "d_star": r["d_star"], "threshold": r["threshold"]})
    for ns in (8, 16, 32):
        r = calib_dstar(n_seeds=ns)
        t3["n_seeds"].append({"n_seeds": ns, "d_star": r["d_star"], "threshold": r["threshold"],
                              "noise_std": r["noise_std"]})
    for nd in (6, 11, 21, 41):
        r = calib_dstar(n_d=nd)
        t3["resolution"].append({"n_d": nd, "d_star": r["d_star"]})
    # also record one full means vector for sanity
    t3["sample_means_seed0"] = calib_dstar()["means"]
    out["tests"]["3_dstar_stability"] = t3

    # =====================================================================
    # TEST 4 -- Does 0.1234 in the stored JSON match a recomputed d*?
    # =====================================================================
    recomputed = calib_dstar(seed_off=0)["d_star"]
    out["tests"]["4_dstar_vs_stored"] = {
        "stored_d_star": 0.1234,
        "stored_threshold": 0.1234,
        "note": "stored d_star == stored threshold == 0.1234 exactly; suspicious placeholder",
        "recomputed_d_star_richdesc": recomputed,
    }

    (HERE / "verify_sampling_margin_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE verify_sampling_margin_results.json")


if __name__ == "__main__":
    main()
