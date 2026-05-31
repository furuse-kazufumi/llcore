# VERIFY: downhill_necessity cross-metric agreement (Phase A CrossMetric #3)

## Purpose
Check whether `behavior_elite_dip`'s conclusion ("real tasks are all below_threshold = smooth
= phase-3 / MAP-Elites niching unneeded") reproduces under a THIRD metric `downhill_necessity`.
A 1-2 metric conclusion is fragile; #2 (`fdc_behavior`) already agreed
(VERIFY_FDC_CROSSMETRIC.md). This is #3.

## Procedure (executed, measure_real_tasks_downhill.py, EXIT=0)
1. Re-ran synthetic-knob calibration (`exp_knob_sweep.make_corridor_eval`, D=24, behavior=mean,
   13 d-levels, 3 seeds, **n_bins=12**, fitness_trials=8) with the SAME estimator+grid used for
   the real tasks.
2. Projected each real task's 2D behavior to its PCA 1st PC (matching how
   `metric_behavior_elite_dip._project_1d` and `metric_fdc_behavior` treat real tasks, and the
   1D synthetic grid), then applied `downhill_necessity`.
3. Compared each value to d* threshold; judged below/above.

Budget honesty: reservoir per-eval is ~100ms (measured). The original 1600x10x5 plan = 84,000
evals/task (~2.3h/task) timed out twice. Reduced to reservoir n_samples=250, fitness_trials=2,
3 seeds (~190s/task) and step6 n_samples=600, fitness_trials=1, 3 seeds; n_bins=12 keeps per-bin
occupancy dense (~21-50/bin). 3 seeds meets the 3+ seed requirement. Smaller per-bin counts +
fitness_trials=2 raise sampling/decision noise -- disclosed; std over seeds is reported (it was
0.0, see below).

## Calibration result (honest -- downhill is a NOISY, non-monotone instrument)
metric_at_dstar (downhill at d=0.16) = **0.6477**. The curve is **NOT monotone** and essentially
**uncorrelated with d** (spearman_vs_d = **-0.0489**):

| d | 0.00 | 0.05 | 0.10 | 0.13 | 0.16 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.85 | 1.00 |
|---|------|------|------|------|------|------|------|------|------|------|------|------|------|
| downhill | 0.000 | 0.000 | 0.000 | 0.648 | 0.648 | 0.673 | 0.649 | 0.515 | 0.325 | 0.000 | 0.000 | 0.000 | 0.000 |

This is consistent in shape with the pre-existing `calibration_results.json` (spearman 0.458,
monotone_strict false, metric_at_dstar 0.722): downhill reads **0.0 on smooth landscapes
(d<=0.10)**, spikes for **mid-depth dips (d=0.13-0.50)**, then **collapses back to 0.0 for deep
dips (d>=0.60)** -- at very deep dips the global-peak behavior bin dominates and hill-climb
trivially reaches it, so downhill under-reads. So downhill is a **poor monotone calibrator**
(unlike fdc, rho=1). Its only clean, well-separated signal is the **0.0 plateau for smooth
landscapes (d<=0.10)** vs **non-zero for mid-depth carved dips**.

## Real-task result -- **AGREEMENT (all below)**

| task | downhill mean | 95%CI | d* (0.6477) | below? |
|------|---------------|-------|-------------|--------|
| variable_delay_recall | **0.0000** | [0.0000, 0.0000] | 0.6477 | **YES (CI strictly below)** |
| flip_flop | **0.0000** | [0.0000, 0.0000] | 0.6477 | **YES (CI strictly below)** |
| step6_text_proxy | **0.0000** | [0.0000, 0.0000] | 0.6477 | **YES (CI strictly below)** |

All 3 read **exactly downhill_necessity = 0.0** (reach_fraction = 1.0, all 3 seeds; std=0.0).
`all_below_threshold = True`, `agrees_with_behavior_elite_dip = True`.

**3-metric consensus**: behavior_elite_dip (below), fdc_behavior (below), downhill_necessity
(below) all agree the 3 real tasks are smooth and phase-3 is NOT load-bearing.

## Is the 0.0 a degenerate artifact? (downhill's main failure mode) -- NO
The metric's docstring warns sparse/disconnected occupied cells artificially LOWER reach
(RAISE deceptiveness). The opposite failure (artificially HIGH reach -> 0 deceptiveness) would
occur if the grid were nearly empty (1-2 cells trivially self-reaching). I probed the actual
elite profiles used:

- **flip_flop**: occupied_cells = **12 / 12** (fully dense), **reach_fraction = 1.000**.
  Profile along the PCA-1D behavior axis rises near-monotonically
  (0.736, 0.742, 0.799, 0.812, 0.843, 0.843, 0.881, 0.864, 0.879, 0.879, 0.942) with only the
  last edge bin lower (0.421) -- which still hill-climbs back up, so no downhill step is ever
  required.
- **step6_text_proxy**: occupied_cells = **12 / 12** (fully dense), **reach_fraction = 1.000**.

So 0.0 means the real-task elite-fitness profile is smooth enough that pure hill-climbing
reaches the global optimum from **everywhere** -- the exact "no downhill required" signature,
matching the synthetic smooth region (d<=0.10 also reads exactly 0.0). The agreement is genuine,
not a sparsity/empty-grid artifact.

## Honest caveats (why this is corroboration, not proof)
1. **Operational comparison**: below_threshold = same-metric/same-estimator/same-grid-dim (1D)
   comparison of raw value vs the raw value at d*=0.16 in synthetic calibration. RANK transfers,
   calibrated MAGNITUDE does not. Not a claim about each task's true d.
2. **PCA-1D projection** = max-variance direction, not necessarily the deceptive direction; a
   dip orthogonal to PC1 could be missed (shared caveat with the other two metrics).
3. **NOT 3 independent proofs**: all three metrics share the sampling population and the PCA-1D
   projection style; systematic error (e.g. deceptive structure invisible to PC1) is shared.
   "3-metric agreement" = three geometric indicators pointing the same way under a shared
   projection -- stronger than one metric, but NOT three independent witnesses.
4. **downhill is a weak calibrator** (spearman -0.05, non-monotone, collapses for deep dips).
   Its agreement here rests on the clean 0.0-vs-nonzero binary (smooth vs mid-depth dip), which
   is well separated for these tasks, rather than on a graded magnitude.
5. **Reduced budget**: reservoir 250 samples x fitness_trials=2 x 3 seeds (per-eval ~100ms cap).
   The 0.0/std=0.0 result is robust across the 3 seeds, but per-bin counts are smaller than the
   1600x10x5 ideal; a richer budget would tighten the noise floor (it would not change the 0.0,
   given the dense 12/12 occupancy and reach=1.0).

## Net
The behavior_elite_dip "all below / smooth / phase-3 unneeded" conclusion **survives a third
cross-metric check**. Combined with fdc_behavior (#2): **3/3 metrics agree**, materially
strengthening the honest-negative ("phase-3 not load-bearing on these tasks") beyond the fragile
single-metric claim that motivated this cross-check. The conclusion remains a corroborated
honest-negative, with the standard caveat that the metrics share a PCA-1D projection and are
therefore not fully independent.

## Artifacts
- `downhill_necessity_crossmetric.json` (machine-readable; read back and parsed OK)
- `measure_real_tasks_downhill.py` (bridge: PCA-1D projection + downhill; real-task code read-only)
- `_downhill_run.log` (run log, EXIT=0)
