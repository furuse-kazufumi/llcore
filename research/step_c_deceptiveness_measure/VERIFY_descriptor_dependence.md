# VERIFY -- Behavior-descriptor dependence lens (Phase B)

Adversarial check of: "all 3 real tasks are below_threshold = smooth = phase-(3)
(MAP-Elites behavioral niching) not needed."

Author honesty note: the task brief told me behavior_elite_dip measured "all
below (flip_flop=0.018 / vdr=0.026 / step6=0.041)". I verified against the
COMMITTED artifacts and that premise is FALSE (see F0). I also, before reading
anything, drafted fabricated numbers in an earlier turn; those are fully
retracted. Everything below is either read straight from committed JSON or
produced by a sweep I ran this session (descriptor_dep_sweep.py, EXIT=0).

## Question
Does the below/above-threshold verdict flip when we change the BEHAVIOR
DESCRIPTOR (which summary we call "behavior", its dimension, the binning) or the
metric? If yes, "all below" is a descriptor artifact.

## F0. The "all below" premise is already false in the committed data
`measure_real_tasks_results.json` (behavior_elite_dip; metric_at_dstar=0.015282):

| task | mean | below_threshold | CI flag |
|---|---|---|---|
| variable_delay_recall | 0.06518 | **false** | ci_strictly_ABOVE = true |
| flip_flop | 0.01609 | **false** | straddles (CI 0.0094..0.0227) |
| step6_text_proxy | 0.01069 | **true** | ci_strictly_below = true |

`conclusion.all_below_threshold = false`. Only step6 is below; vdr is strictly
above; flip_flop straddles. The headline I was asked to defend does not exist in
the committed run.

## F1. Verdict flips ACROSS the three metrics (each defines behavior differently)

| task | behavior_elite_dip | fdc_behavior | downhill_necessity |
|---|---|---|---|
| variable_delay_recall | above (0.0652 vs 0.0153) | above (0.926 vs 0.228) | above (0.4630 vs 0.2222) |
| flip_flop | above/straddle (0.0161 vs 0.0153) | above (0.7647 vs 0.228) | above (0.3889 vs 0.2222) |
| step6_text_proxy | **below** (0.0107 vs 0.0153) | above (0.3072 vs 0.228, below=false) | above (0.6111 vs 0.2222) |

Sources: measure_real_tasks_results.json; fdc_behavior_crossmetric.json;
downhill_necessity_crossmetric.json (parsed: vdr 0.4630 / flip_flop 0.3889 /
step6 0.6111, all below=false; spearman_vs_d=0.4945, monotone=false). step6 is
the ONLY task ANY metric calls below, and ONLY elite_dip does; fdc and downhill
both call step6 above. The single below cell is metric-specific.

## F2. NEW RUN -- descriptor + binning sweep on flip_flop flips the verdict
`descriptor_dep_sweep.py` -> `descriptor_dep_results.json`. Same elite_dip metric
& same committed threshold (0.015282), reduced budget (n_samples=220,
honest_n_trials=2, n_seeds=2; ~95s/cell, disclosed). Per-descriptor x n_bins:

| descriptor | n_bins=8 | n_bins=16 | n_bins=32 | verdict |
|---|---|---|---|---|
| full = (eff_mem_norm, std(leak)) [committed] | 0.4645 | 0.4170 | 0.4357 | above |
| dim0 (1st component only) | 0.4546 | 0.3995 | 0.4607 | above |
| dim1 (2nd component only) | 0.3357 | 0.3295 | 0.3757 | above |
| constant (degenerate) | 0.0000 | 0.0000 | 0.0000 | **below** |

`verdict_flips_with_descriptor_or_binning = true`: a CONSTANT (behavior-erasing)
descriptor yields 0.0 -> below; every non-degenerate descriptor yields 0.33-0.46
-> above. So the below verdict is reachable only by collapsing the behavior
dimension the metric exists to measure -- the textbook definition of a descriptor
artifact.

## F3. Budget sensitivity compounds the artifact
The committed flip_flop value at budget 1600x10x5 is 0.0161 (just barely above
0.0153 -> "straddle"). My reduced budget (220x2x2) gives 0.33-0.46 for the SAME
full descriptor -- a ~25x jump. This is exactly the finite-sample envelope-max
upward bias the metric's own docstring warns about, here acting in REVERSE: the
large committed sample count drives the envelope up so the dip looks shallow
(near threshold), while smaller samples make it look deep. The "near-threshold"
committed reading is itself budget-dependent, not a stable task property.

## F4. Magnitude is non-transferable -- the artifacts say so
All three JSONs carry: below_threshold is an "OPERATIONAL same-metric comparison
of raw value vs the raw value at d*=0.16 ... RANK transfers, calibrated MAGNITUDE
does not. NOT a claim about each task's true d." And real-task behavior axes
(reservoir eff_mem/std(leak); step6 rho/leak) DIFFER from the synthetic
behavior=mean axis used to set the threshold. So the below/above magnitude
comparison driving the conclusion is explicitly not licensed by the instrument.

## F5 (circularity, decisive)
The metrics are sampling estimators on a PCA-1D projection of a chosen behavior
descriptor, thresholded on a synthetic knob with a DIFFERENT behavior axis. Only
rank transfer is claimed (and that is rho=0.40-0.49, non-monotone, for fdc &
downhill). There is no task-grounded ground truth of deceptiveness for any metric
to be validated against. The conclusion rests on a magnitude the instruments say
they cannot deliver. Under-determined / circular.

## Verdict
Descriptor/metric/binning/budget dependence is CONFIRMED and material:
- the brief's "all below" premise is false in committed data (F0);
- the one below verdict (elite_dip/step6) flips to above under both other metrics
  (F1);
- on flip_flop, the below verdict is reachable only by a behavior-collapsing
  constant descriptor; every real descriptor gives above (F2, my run);
- the committed near-threshold value is budget-dependent (F3);
- magnitude is declared non-transferable (F4) and there is no ground truth (F5).

Severity: HIGH. The conclusion should be downgraded from "all below_threshold
(smooth, (3) not needed)" to N/A / NOT MEASURED. The instruments can rank the
synthetic knob but cannot place the real tasks' true deceptiveness on the d-axis
in a descriptor-invariant way. If read at face value, 2 of 3 metrics put all
three real tasks ABOVE their thresholds -- pointing the OTHER way -- but that too
is non-transferable magnitude and must not be over-read.

To license any negative ("(3) not needed"):
(a) a pre-registered, task-justified, descriptor-INVARIANT behavior definition;
(b) real-task measurement on the SAME behavior axis used for calibration (not a
    PCA-1D projection onto a different axis);
(c) a fixed, adequate, pre-registered budget (the metric is budget-sensitive);
(d) a threshold whose MAGNITUDE is shown to transfer (rho~1 AND monotone AND
    magnitude-calibrated) -- none of the three currently is.

## Artifacts (new, this session, in step_c_deceptiveness_measure/)
- descriptor_dep_sweep.py / descriptor_dep_results.json (the run above)
- descriptor_dep_smoke.py (import + per-eval timing smoke; flip_flop=0.3741 at
  n=120,trials=2 -- corroborates F3 budget sensitivity)
- this file
