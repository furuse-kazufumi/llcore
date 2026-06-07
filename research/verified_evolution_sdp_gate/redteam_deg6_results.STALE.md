# STALE — `redteam_deg6_results.json` is a PRE-CORRECTION (SCS-era) artifact (2026-06-03)

**Status: SUPERSEDED. Do not cite its complementarity / coverage numbers.**

`redteam_deg6_results.json` was produced by an early run of `redteam_deg6.py` **before** the
SCS→CLARABEL solver-artifact correction (DEG6_VERDICT.md §0, AUDIT_SCS_CLARABEL_2026-06-03.md). It
still reports the **retracted** "complementary" degree-4/6 ladder:

| field in `redteam_deg6_results.json` | reported (stale) | corrected truth |
|---|---|---|
| `lens1_soundness.n_deg_certified` | **54** | 4 deg-certified residual under CLARABEL (`exp_deg6_ladder_results.json`) |
| `lens4_numerical.by_margin[*].complement_both_pos` | **true** (1e-6 / 1e-7 / 1e-8) | **false → NESTED** under CLARABEL |
| `lens4_numerical.PASS_complement_robust` | **true** | the margin sweep was **structurally blind** to the artifact |
| `all_pass` | true | the ladder-non-nested premise it "confirmed" is retracted |

## Why it was not re-run

Two reasons, both honest:

1. **Over the 8-minute budget.** The recorded `elapsed_s = 2406.9` (~40 min) — dominated by lens1's
   50 000-sample soundness oracle over every certified gene and lens2's 8-seed evolution sweep. A
   faithful re-run is far outside the sub-8-minute window allotted, so per the audit task it is
   annotated rather than regenerated.
2. **Its fixture was overwritten.** `redteam_deg6.py` (lens1) reads the live
   `exp_deg6_residual_genes.json`, which the later CLARABEL ladder run shrank from the SCS-era
   54 deg_certified / 53 residual_uncert to the post-correction **4 / 10**. A re-run would therefore
   no longer exercise the original SCS-era battery at all, so the JSON it produced is not directly
   reproducible without first repointing the script at `exp_deg6_residual_genes_scs_era.json`.

## What supersedes it

- **`exp_deg6_ladder_results.json`** (CLARABEL, the canonical coverage frontier): the ladder is
  **NESTED** — `complementarity.deg4_minus_deg6 = 0`, `deg6_minus_deg4 = 1`,
  `G_A2_ladder_non_nested = false`. The quadratic SDP certifies 286/300; deg4/6 add only +4.
- **`verify_solver_artifact.py`** + **`test_solver_swap_regression.py`** (on the immutable
  `exp_deg6_residual_genes_scs_era.json` snapshot): reproduce the keystone — SCS fabricates
  complementarity (deg4_only=23, deg6_only=13, both=18), CLARABEL is nested (0/0/54), and CLARABEL
  recovers 42–43 of the 53 SCS false negatives.
- **`verify_certifier_jsr_soundness.py`** and DEG6_VERDICT.md §3 confirm certifier soundness
  independently of the solver bug (the artifact was false *negatives*; soundness was never at risk).

## Honest note on lens4

The lens4 "margin sweep" reported `PASS_complement_robust = true`, i.e. it **confirmed** the
artifact rather than catching it — exactly the lesson in DEG6_VERDICT.md §7: a margin sweep cannot
detect an SCS false-negative; only a solver swap can. That is why the standing guard is the
solver-swap regression test, not a margin sweep.
