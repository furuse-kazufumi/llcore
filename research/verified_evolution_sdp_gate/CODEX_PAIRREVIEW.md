# Codex pair-review — SDP-gated coupled evolution (2026-06-03)

Codex (gpt-5.4, `codex exec -s read-only`) adversarial review. Per
[[feedback_codex_pair_review_for_llcore]] + [[feedback_external_ai_verify]]: each finding
verified against real code, then fixed or disclosed. 9 findings (1 blocker, 5 high, 3 med).

**Headline robustness:** the G4 rotation payoff (sdp ≫ inf, p=3.1e-5, mechanism-attributed
via the exp1 *certified* inf/2norm/sdp region ceilings) survives every finding — the issues
concern the soundness *framing*, the *non_certified* frontier number, the nonnormal task's
status, fallback transparency, and statistical disclosure. Dispositions:

| # | sev | finding | verified? | disposition |
|---|---|---|---|---|
| 1 | blocker | G1 measured final-pop @2500, not admitted children @≥20k | TRUE | **FIXED**: added proper admitted-child soundness check @20k (`verify_g1_admitted.py`); reframed G1. |
| 2 | high | empirical oracle is from-below → cannot *certify* soundness, only falsify | TRUE | **FIXED (reframe)**: soundness rests on the certifier **theorems** (‖J‖_∞<1⟹ρ<1; σ_max<1⟹contraction; LMI⟹P-contraction, Track C/D-proven); the oracle is a *falsification/consistency* check (found 0). Stated as such. |
| 3 | high | Lens B weaker than prereg (sdp+two only, 50k not 100k, no edge search) | TRUE | **DISCLOSED**: scope reduction for tractability; covers the richest gates (main risk). Full prereg Lens B = future. |
| 4 | high | fallback pressure recorded but not analyzed; inf low-rotation could be fallback-flooding | TRUE | **FIXED**: fallback/rejection counts now reported; inf 0.41 clarified (caps + ungated init-elites). exp1 *certified* region ceiling (inf=0.38) is independent of the GA, so the mechanism holds regardless. |
| 5 | med | Lens A is gene-KEYED random (not gene-independent scalar); hash() salting | PARTLY | **DISCLOSED + checked**: numeric-tuple hash is NOT salted (verified deterministic) so it is reproducible; but it is gene-keyed, not pure-noise. It still controls the artifact (fitness uncorrelated with dynamics → sdp gets no edge, p=0.91). Noted. |
| 6 | high | stats broader than prereg (n=15 not 20, no power, no multiplicity) | TRUE | **DISCLOSED**: deviations in §7; added multiplicity note (rotation p=3.1e-5 ≪ Bonferroni 0.05/9) + post-hoc power (psd=1.00 = complete separation). |
| 7 | med | nonnormal is circular by construction but verdict promotes "PASS" | TRUE | **FIXED (demote)**: nonnormal reframed as a *reachability demonstration* (optimum SDP-only by design), NOT co-equal payoff evidence. Headline = rotation only (non-circular). |
| 8 | med | exp1 per_region max not contracting-filtered; non_certified 0.99 may be divergent | TRUE | **FIXED**: code bug (mask not intersected with contracting) corrected; frontier ceiling recomputed on contracting-only genes. |
| 9 | med | sdp_only thin shell — no margin audit disclosed | TRUE | **DISCLOSED**: sdp_only genes are thin-margin (min-eig ~1e-7); the Track-D certifier independently re-checks LMI eigenvalues (not solver-blind), but near-boundary robustness is a stated caveat. |

No finding overturns the headline; all are rigor/honesty improvements to secondary claims —
exactly what the pair-review discipline is for. Corrections applied to VERDICT.md §5–§9.
