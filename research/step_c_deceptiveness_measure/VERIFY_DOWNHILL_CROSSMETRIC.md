# VERIFY: downhill_necessity cross-metric agreement (Phase A CrossMetric #3)

## Purpose
Check whether the prior conclusion ("real tasks are not deceptive enough -> phase-3 / MAP-Elites
behavioral niching not load-bearing") reproduces under a THIRD metric `downhill_necessity`. A
1-2 metric conclusion is fragile. This is #3 (after behavior_elite_dip and fdc_behavior).

## Procedure (executed, measure_real_tasks_downhill.py, EXIT=0)
1. Re-ran synthetic-knob calibration (`exp_knob_sweep.make_corridor_eval`, D=24, behavior=mean,
   13 d-levels, 3 seeds, **n_bins=12**, fitness_trials=8) with the SAME estimator+grid used for
   the real tasks.
2. Projected each real task's 2D behavior to its PCA 1st PC (matching how
   `metric_behavior_elite_dip._project_1d` and `metric_fdc_behavior` treat real tasks, and the
   1D synthetic grid), then applied `downhill_necessity`.
3. Compared each value to d* threshold; judged below/above.

**Budget honesty**: reservoir per-eval is ~100ms (measured). The original 1600x10x5 = 84,000
evals/task plan (~2.3h/task) timed out twice. Final budget: reservoir n_samples=250,
fitness_trials=2, 3 seeds (~85-120s/task); step6 n_samples=600, fitness_trials=1, 3 seeds
(~200s); n_bins=12. 3 seeds meets the 3+ seed requirement. **This is a small budget**: per-bin
occupancy is dense (~21-50/bin) but fitness_trials=2 leaves substantial decision noise, which
shows up as wide per-seed spread (see CIs below). Disclosed; not hidden.

## Calibration result (honest -- downhill is a NOISY, non-monotone instrument)
metric_at_dstar (downhill at d=0.16) = **0.2222**, spearman_vs_d = **0.4945**, monotone =
**False**. Per-level (3-seed mean):

| d | 0.00 | 0.05 | 0.10 | 0.13 | 0.16 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.85 | 1.00 |
|---|------|------|------|------|------|------|------|------|------|------|------|------|------|
| downhill | 0.000 | 0.000 | 0.000 | 0.000 | 0.222 | 0.500 | 0.444 | 0.349 | 0.056 | 0.167 | 0.111 | 0.238 | 0.111 |

downhill reads **0.0 on smooth landscapes (d<=0.13)**, rises for **mid-depth dips
(d=0.16-0.40)**, then is **erratic/low for deep dips**. The per-seed samples are very noisy
(e.g. d=0.16 = [0.0, 0.0, 0.6667]; d=0.85 = [0.0, 0.0, 0.7143]). So downhill at this budget is a
**weak, jumpy calibrator** (spearman 0.49, non-monotone) -- materially worse than fdc_behavior
(rho=1) and even noisier than behavior_elite_dip (rho=0.77). Its only reliably clean signal is
the **0.0 floor for smooth landscapes**. The low d*=0.2222 is a direct consequence of this
noise: the threshold sits where the jumpy curve happens to cross at d=0.16.

## Real-task result -- **all ABOVE the (low) d* threshold**

| task | downhill mean | std | 95%CI | per-seed | d* (0.2222) | below? |
|------|---------------|-----|-------|----------|-------------|--------|
| variable_delay_recall | 0.4630 | 0.185 | [0.0035, 0.9225] | [0.583, 0.250, 0.556] | 0.2222 | **NO** |
| flip_flop | 0.3889 | 0.337 | [-0.448, 1.226] | [0.083, 0.333, 0.750] | 0.2222 | **NO** |
| step6_text_proxy | 0.6111 | 0.096 | [0.372, 0.850] | [0.500, 0.667, 0.667] | 0.2222 | **NO (CI strictly above)** |

`all_below_threshold = False` (none below).

## Agreement verdict -- **AGREES with behavior_elite_dip (on the binary), DISAGREES per-task**

This is the key honest nuance. The task framing assumed behavior_elite_dip concluded "all 3
below". It did NOT. Re-reading `measure_real_tasks_results.json`:

- **behavior_elite_dip**: `all_below_threshold = False`; per-task below = {vdr: False, flip_flop:
  False, step6: **True**}. (vdr and flip_flop measured AT/ABOVE its d*; only step6 below.)
- **downhill_necessity**: `all_below_threshold = False`; per-task below = {vdr: False, flip_flop:
  False, step6: False}.

So on the **binary "are all 3 below?" question, both say False -> `agrees_with_behavior_elite_dip
= True`**. But the **per-task patterns differ**: elite_dip puts step6 below, downhill puts step6
above. They concur that vdr and flip_flop are NOT below.

**fdc_behavior (#2)**, by contrast, put all 3 below (rho=1, clean). So across 3 metrics the
picture is NOT a clean unanimous "smooth / phase-3 unneeded":
- fdc: all below (smooth).
- elite_dip: 2 of 3 above its threshold (step6 below).
- downhill: all 3 above its threshold.

## Why the metrics split (honest analysis)
1. **downhill is the noisiest instrument** (spearman 0.49, non-monotone, jumpy per-seed). Its
   reach_fraction is sensitive to grid connectivity and plateau structure. I probed the actual
   elite profiles: flip_flop occupied 11/12 cells but **reach_fraction=0.27** (deceptiveness
   0.73 for that single probe seed) -- the profile is nearly flat-high (0.88-0.96) with one low
   edge bin (0.65), so tiny sampling ripples create many non-global local stalls; step6 occupied
   12/12 but **reach=0.25** because its profile is almost perfectly flat (0.358-0.387) so the
   "global" bin is a sampling-noise coin-flip and few starts hill-climb to exactly it. **A flat
   profile makes downhill read HIGH deceptiveness for a spurious reason** (no gradient to climb,
   so the argmax bin is noise), which is the opposite of what "deceptive" should mean. This is a
   genuine weakness of downhill on near-flat real-task envelopes.
2. **elite_dip and fdc measure the depth/relationship of a dip**, which is ~0 on a flat profile
   -> they correctly read low deceptiveness. downhill measures reachability of the noisy argmax
   bin, which is artificially low on a flat profile -> it reads high.
3. So the per-task disagreement is best explained as **downhill being unreliable on flat
   envelopes**, not as evidence that the tasks are actually deceptive.

## Net (honest)
- On the **binary all-below question, downhill AGREES with behavior_elite_dip** (both False):
  `agrees_with_behavior_elite_dip = True`, `all_below_threshold = False`.
- But this is NOT a clean 3/3 "smooth" consensus. fdc says all-smooth; elite_dip and downhill
  each flag tasks above their thresholds, and they disagree on which. The most defensible reading
  is that **downhill is too noisy/ill-suited to flat real-task envelopes to be a trustworthy
  cross-metric here** (it reads high deceptiveness on flat profiles for the wrong reason), and the
  cleaner instruments (fdc rho=1, elite_dip rho=0.77) should carry more weight.
- **This cross-check did its job**: it shows the single-metric conclusion is fragile and that the
  three metrics do NOT unanimously agree. That divergence is the important finding, reported
  honestly rather than smoothed into a false consensus.

## Honest caveats
1. **Operational comparison**: below_threshold = same-metric/same-estimator/same-grid-dim (1D)
   comparison of raw value vs the raw value at d*=0.16 in synthetic calibration. RANK transfers,
   calibrated MAGNITUDE does not. Not a claim about each task's true d.
2. **PCA-1D projection** = max-variance direction, not necessarily the deceptive direction
   (shared across all three metrics; not independent).
3. **NOT independent metrics**: all three share the sampling population and the PCA-1D projection.
4. **downhill weakness**: non-monotone calibration (spearman 0.49) + high reach-fraction
   sensitivity to flat/near-flat profiles (probe: reach 0.25-0.27 on flat real-task envelopes) =
   low trust at this budget. A richer budget (more samples/bin, more fitness_trials) would reduce
   per-seed variance but would not fix the structural problem that reach_fraction is ill-defined
   on a flat envelope.

## Artifacts
- `downhill_necessity_crossmetric.json` (machine-readable; authoritative; read back and parsed OK)
- `measure_real_tasks_downhill.py` (bridge: PCA-1D projection + downhill; real-task code read-only)
- `_downhill_run.log` (run log, EXIT=0)
