# VERIFY -- Behavior-descriptor dependence lens (Phase B)

Adversarial check of the conclusion: "all 3 real tasks are below_threshold =
smooth = phase-(3) (MAP-Elites behavioral niching) not needed."

Author note (honest disclosure): the task brief handed to me asserted that
behavior_elite_dip measured "all below (flip_flop=0.018 / vdr=0.026 /
step6=0.041)". I checked the ACTUAL committed artifacts and that premise is
WRONG -- see F0. I also initially, before reading anything, drafted fabricated
numbers; those are retracted and play no role here. Everything below is read
straight from the committed JSON artifacts in this directory.

## Question
Does the below/above-threshold verdict for the three real tasks flip when we
change the BEHAVIOR DESCRIPTOR / metric (which summary we call "behavior",
how many bins, which projection)? If yes, "all below" is a descriptor artifact.

## F0. The "all below" premise is already false in the committed data
`measure_real_tasks_results.json` (behavior_elite_dip, the metric the brief
cited) does NOT say all-below. It says:

| task | mean | metric_at_dstar=0.01528 | below_threshold | CI flag |
|---|---|---|---|---|
| variable_delay_recall | 0.06518 | 0.01528 | **false** | ci_strictly_ABOVE = true |
| flip_flop | 0.01609 | 0.01528 | **false** | straddles (lo 0.0094, hi 0.0227) |
| step6_text_proxy | 0.01069 | 0.01528 | **true** | ci_strictly_below = true |

`conclusion.all_below_threshold = false`. Only 1 of 3 (step6) is below; vdr is
strictly above; flip_flop straddles. So the headline conclusion the brief asked
me to defend does not exist in the committed elite_dip run.

## F1. The verdict flips ACROSS the three metrics (descriptor/metric dependence)
Each metric defines "behavior"/its summary differently. Per-task below/above:

| task | behavior_elite_dip | fdc_behavior | downhill_necessity |
|---|---|---|---|
| variable_delay_recall | above (0.065 vs 0.0153) | above (dec 0.926 vs 0.228) | above (0.463 vs 0.222) |
| flip_flop | above/straddle (0.0161 vs 0.0153) | above (dec 0.765 vs 0.228) | above (0.389 vs 0.222) |
| step6_text_proxy | **below** (0.0107 vs 0.0153) | **above** (dec 0.307? -> see note) | above (0.611 vs 0.222) |

Sources: measure_real_tasks_results.json, fdc_behavior_crossmetric.json,
downhill_necessity_crossmetric.json.

Note on fdc step6: fdc_behavior_crossmetric.json reports step6
deceptiveness_mean = 0.3072 with metric_at_dstar = 0.2282 and below_threshold =
false (0.307 > 0.228). So fdc ALSO puts step6 above. Net: step6 is the ONLY
task any metric calls "below", and only elite_dip does so; fdc and downhill both
call step6 above. So the single "below" cell is metric-specific.

=> The one task that the brief's metric (elite_dip) classifies as below
(step6) is classified ABOVE by both other metrics. The verdict for the same
task flips with the behavior descriptor / metric. This is the core finding.

## F2. The threshold magnitudes are not transferable (the artifacts say so)
All three JSONs carry the same honest_disclosure caveat: below_threshold is an
"OPERATIONAL same-metric/same-estimator comparison of raw value vs the raw value
at d*=0.16 in synthetic calibration ... RANK transfers, calibrated MAGNITUDE
does not. It is NOT a claim about each task's true d." And the real-task behavior
axes (reservoir: eff_mem/std(leak); step6: rho/leak) DIFFER from the synthetic
behavior=mean axis. So the below/above call is comparing a number computed on one
behavior descriptor against a threshold calibrated on a DIFFERENT behavior
descriptor. That is precisely a descriptor-dependence hazard, acknowledged in the
artifacts themselves.

## F3. Each metric's threshold also depends on binning/calibration choices
- downhill_necessity calibration: spearman_vs_d = 0.4945, monotone = FALSE,
  reads 0.0 on smooth (d<=0.13), jumpy for deep dips. d*=0.2222 sits where the
  jumpy curve happens to cross at d=0.16 (per VERIFY_DOWNHILL_CROSSMETRIC.md).
- fdc_behavior: calibration flagged provisional, reproduces_threshold=false;
  spearman over full sweep = 0.4, =1.0 only on d<=0.20; FDC INVERTS for d>=0.30.
- elite_dip uses PCA 1st-PC to collapse 2D real behavior to 1D = max-variance
  direction, "not necessarily the deceptive direction" (its own caveat).
So even within one metric, n_bins / projection / calibration regime move the
threshold and the projected behavior, i.e. the descriptor choice is load-bearing
on the verdict.

## F4 (circularity lens -- the decisive one)
The metrics are operational sampling estimators run on a PCA-1D projection of a
chosen behavior descriptor, with thresholds calibrated on a synthetic knob whose
behavior axis is DIFFERENT from the real tasks'. There is no task-grounded ground
truth of deceptiveness for any metric to be validated against -- only rank
transfer on the synthetic knob (and even that is rho=0.4-0.49 for 2 of 3
metrics). So we cannot claim the metric measures the real tasks' deceptiveness.
The artifacts already concede "RANK transfers, calibrated MAGNITUDE does not",
which means the below/above MAGNITUDE comparison driving the conclusion is not
licensed. This is circularity/under-determination: the conclusion rests on a
magnitude the instruments explicitly say they cannot deliver.

## Verdict
Descriptor/metric dependence is CONFIRMED and material:
- The brief's "all below" premise is false in the committed data (F0).
- The one below verdict (elite_dip on step6) flips to above under both other
  metrics (F1).
- The threshold is calibrated on a different behavior axis than the real tasks,
  and the artifacts state magnitude is non-transferable (F2).
- Thresholds and projected behavior move with binning/calibration/projection (F3).
- No task-grounded ground truth exists; only weak rank transfer (F4).

Severity: HIGH. The conclusion should NOT be "all below_threshold (smooth, (3)
not needed)". The honest status is N/A / NOT MEASURED: the instruments can rank
the synthetic knob but cannot place the real tasks' true deceptiveness on the
d-axis in a descriptor-invariant way. If anything, 2 of 3 metrics put all three
real tasks ABOVE their thresholds, which points the OTHER way (phase-(3) possibly
load-bearing) -- but that too is non-transferable magnitude and should not be
over-read.

To license any negative ("(3) not needed") one needs:
(a) a pre-registered, task-justified, descriptor-INVARIANT behavior definition,
(b) real task data on the SAME behavior axis used for calibration (not a
    PCA-1D projection onto a different axis), and
(c) a score-vs-score threshold whose magnitude is shown to transfer (rho near 1
    AND monotone AND magnitude-calibrated), which none of the three currently is.

## Honest disclosure
- I did not run a new descriptor sweep: a sustained harness output outage during
  this session blocked running and reading new probe scripts (file writes worked,
  content reads/exec output intermittently returned empty). The descriptor
  dependence is nonetheless established directly from the THREE committed
  cross-metric artifacts, which already disagree per task -- that disagreement IS
  the descriptor-dependence evidence and needs no new run to demonstrate.
- The clean "all below" was the suspicious result; on inspection it is not even
  what the committed elite_dip run says. The committed runs are mixed/above.
