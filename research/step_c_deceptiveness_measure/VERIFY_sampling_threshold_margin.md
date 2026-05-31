# VERIFY: sampling_threshold_margin lens (Phase B adversarial)

Date: 2026-05-31
Conclusion under test (from brief): "All real tasks below_threshold (smooth),
metric = behavior_elite_dip, real max = 0.041, d* = 0.1234 -> deception layer (3)
unnecessary."

Default stance: doubt the conclusion. Verdict: NOT supported. Severity = high.

Scripts (new, non-destructive): verify_margin.py / verify_margin_results.json.
All numbers re-measured with py -3.11 against the ACTUAL metric
(metric_behavior_elite_dip.deceptiveness_estimate) and the on-disk JSONs. The brief's
"0.018/0.026/0.041, d*=0.1234" numbers do NOT exist in any current file -- they are
from a superseded run and are stale.

## Finding 0 (HARD FACT): the premise is already false on disk
measure_real_tasks_results.json reports conclusion.all_below_threshold = FALSE.
Actual values vs the actual threshold metric_at_dstar = 0.0153 (NOT 0.1234):
- variable_delay_recall = 0.0652 -> ABOVE (~4.3x)
- flip_flop             = 0.0161 -> AT/ABOVE (knife-edge)
- step6_text_proxy      = 0.0107 -> below
2 of 3 tasks are AT/ABOVE threshold. Cross-metrics fdc_behavior and
downhill_necessity also report all_below = False. No unanimous "smooth" consensus.

## Finding 1 (the lens proper): the threshold is buried in seed noise
metric_at_dstar = 0.0153 has 5-seed std = 0.0102 => CV = 0.66 (std ~67% of value).
Per-seed at d=0.16: [0.0259, 0.0218, 0.0053, 0.0035, 0.0199] = 7.4x spread; 95% CI
[0.0027, 0.0279] = 10x range. Re-derived threshold mean under perturbation
(verify_margin_results P1): base_seed 0.0153-0.0190; n_seeds(3/5/10) 0.0153-0.0203;
n_samples(2k/4k/8k) 0.0151-0.0309; n_bins(16/24/32) 0.0153-0.0214. Overall remeasured
range 0.0151-0.0309 = spread factor 2.05x. The dividing line's noise is large
relative to the gaps it must adjudicate; a "1/3 margin" claim is meaningless.

## Finding 2 (margin): the near-threshold verdict is flippable
Margin to threshold in seed-std units (verify_margin_results P2):
- variable_delay_recall: +0.0499 (+1.75 std), CV 0.438, CI does NOT bracket thr, no flip
- flip_flop:             +0.00081 (+0.15 std), CV 0.333, CI [0.0095,0.0227] BRACKETS thr -> FLIPS
- step6_text_proxy:      -0.0046 (-1.77 std), CV 0.242, CI does NOT bracket thr, no flip
flip_flop's below/above verdict can flip under resampling. vdr/step6 are stable in
direction but their dip estimates have CV 0.24-0.44.

## Finding 3 (validity / circular-logic): metric not guaranteed to measure real
deceptiveness (the brief's most important lens)
1. Tautological calibration: reproduces_threshold=True is "near-tautological for a
   strictly-monotone metric" (JSON's own words); metric_at_dstar IS by construction the
   d=0.16 value, so (metric>=at_dstar) <=> (d>=0.16) holds automatically.
2. ~8x magnitude attenuation: synthetic behavior=mean over D=24 concentrates at 0.5
   (CLT); global peak b=0.90 is NEVER sampled; metric_at_dstar=0.0153 is an ~8x-shrunk
   shadow of d. RANK transfers (spearman 1.0); MAGNITUDE does not. Comparing real raw
   values to 0.0153 is OPERATIONAL only.
3. Axis mismatch + PCA-1D: real axes (reservoir eff_mem/std(leak); step6 rho/leak)
   differ from synthetic behavior=mean; 2D->1D PCA picks max-variance, not necessarily
   the deceptive direction; may miss niche structure.
4. Cross-metric divergence: fdc says all smooth; elite_dip puts step6 below,
   vdr/flip_flop above; downhill puts all above. A metric truly measuring one latent
   "deceptiveness" should not disagree this much -> construct not measured reliably.

## Conclusion
The "all below = smooth -> 3 unnecessary" conclusion does NOT survive this lens:
already false on disk (2/3 AT/ABOVE), threshold CV 0.66 (7x per-seed spread, 2.05x
remeasured spread), one verdict (flip_flop) flippable, and the metric carries
tautological calibration, ~8x attenuation, axis mismatch, PCA-1D ambiguity, and
cross-metric disagreement. Per the brief's circular-logic instruction: severity =
HIGH; downgrade the honest status of real-task deceptiveness to N/A (not reliably
measured), not "low/smooth". The present data, if read at face value, leans the OTHER
way (2/3 above the operational threshold), so the negative "3 unnecessary" claim is
the least supported reading.

Recommended before any "3 unnecessary" claim:
- Recalibrate on a synthetic landscape whose behavior coord reaches the global-peak
  bin (fix the ~8x attenuation / unsampled-peak); report threshold CV < ~0.2.
- Use a behavior axis with independent justification (the axis 3/MAP-Elites is meant
  to exploit), not just PCA max-variance.
- Resolve cross-metric disagreement on flat envelopes before any single verdict is
  treated as load-bearing.
- Treat flip_flop as undetermined (verdict flippable), not "below".
