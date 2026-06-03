# RUMP+OR Hardening — Verdict (2026-06-03)

PART 2 of the contraction-certifier hardening. ADDITIVE: a new Rump-verified-PD + OR-of-solvers
verifier factory layered beside the existing SDP certifiers. NO default behaviour was changed.

## What was built

`rump_hardened_verifier.make_sdp_verifier_rump_or()` — an additive `VerifierBackend` (n=2 coupled
gene) that admits a gene iff:

- the solver-independent **inf-norm or 2-norm** closed-form certificate holds (sound fast path,
  identical to the existing certifiers), **OR**
- the **OR-of-{CLARABEL, SCS}** SDP gate finds a Rump-VERIFIED common-Lyapunov `P`: for ANY solver,
  the SDP is solved (with the SAME `margin=1e-7` constraint the existing pipeline uses) and the
  returned `P` passes `rump_pd.verified_pd` on the SAME no-margin matrices the float path tests.

It reuses the validated, adversarially-proven-sound `rump_pd.verified_pd` (11 tests, 36860-trial
skeptic battery, 0 false positives) and `rump_pd.rump_verify_certificate`.

## CRITICAL INVARIANT — matched the float path's ACTUAL recheck (NO margin)

The audit's trap: the SDP **constraint** carries a strict margin (`P - JᵀPJ >> margin·I`), but the
certifier's internal **float recheck** in `lyapunov_sdp_certifier.certify_common_lyapunov` tests the
**no-margin** matrices:

```
eig_P   = min eig(P)            > 0      # P strictly PD, NO margin
min_dec = min eig(P - JᵀPJ)     > 0      # decrease LMI strictly PD, NO margin subtracted
```

The Rump recheck here verifies the **identical** matrices — `P` and every `P - JᵀPJ` — at the SAME
(zero) margin, via `rump_verify_certificate(P, verts, margin=0.0)`. Subtracting `margin·I` would
test a strictly HARDER condition and could spuriously SHRINK the admit set — the exact failure the
spec warns against.

> ⚠️ **CORRECTION (pair-review F3, 2026-06-04 — the direction was inverted):** an earlier draft argued
> the admit set is "preserved-or-grown, never shrunk" because `verified_pd` *dominates* the float
> test. That reasoning is **backwards**. `verified_pd` returns a **sound LOWER bound** on `λ_min`
> (≤ the float-computed `λ_min`), so on the SAME matrix it is the **STRICTER** test: `{Rump-accepted}
> ⊆ {float-accepted}` per-`P`. In principle the Rump recheck could therefore *reject* a barely-PD `P`
> the float test accepts. The observed **286 == 286 (0 lost)** is **EMPIRICAL**, not a domination
> theorem — explained by a large headroom: the SDP constraint enforces a `margin = 1e-7` decrease
> while `verified_pd`'s Cholesky backward-error bound is `~n·u·maxdiag ≈ 1e-15`, so every admitted
> `P` clears the verified-PD test by **~8 orders of magnitude** (confirmed numerically: 4000
> barely-PD matrices with `λ_min ∈ [1e-9,1e-5]`, 0 Rump rejections, 0 unsound lower bounds). The
> OR-of-{CLARABEL,SCS} adds a second solver's `P` as a fallback chance. **Soundness (no false
> positives) is unaffected** — that part of `verified_pd` is correct and proven. Net: the hardening
> strengthens the *basis of trust* (a machine-checked PD proof across two solvers) and no shrinkage
> is observed on this pool at this margin, but "never shrunk" is an **empirical** statement here, not
> a guaranteed invariant.

## Comparison result — EXP-A 300-contracting-gene pool

Pool regenerated EXACTLY as `exp_deg6_ladder.py` does (seed=2024, empirical-contraction filter
ρ<1 at n_samples=4000 from the same seeded stream). Script: `exp_rump_hardening_compare.py`.
Results JSON: `rump_hardening_compare_results.json`.

| Quantity | Value |
|---|---|
| Pool size (contracting genes) | 300 (scanned 786) |
| **Float-recheck SDP admit count** | **286** |
| **Rump+OR admit count** | **286** |
| **Delta (Rump+OR − float)** | **0** |
| Lost (float-only, NOT Rump+OR) — must be 0 | 0 |
| Extra (Rump+OR-only, NOT float) | 0 |
| Extra admits all sound (jsr_lb < 1) | true (vacuously — 0 extra) |
| Extra unsound (jsr_lb ≥ 1) | 0 |
| Admit set preserved-or-grown (no shrink) | true |

The admit count is **byte-for-identical** at 286/300. The invariant held: ZERO genes were lost
(no shrinkage) and ZERO extra (so the JSR soundness check on extras is vacuously satisfied — there
were no extras to expand). The Rump+OR gate reproduces the float gate's admit set EXACTLY on this
pool while resting its verdict on a machine-checked PD proof + a two-solver OR instead of a single
solver's float `eigvalsh` claim.

## DECISION RULE (human decision — NOT flipped here)

Pre-registered rule: **if Rump+OR admit count == float count (expected 286/300), the default
recheck CAN later be promoted to Rump+OR with the invariant exactly preserved.**

Measured: **286 == 286 (delta 0).** → The equal-count branch holds. **Promotion to the default is
SAFE with the invariant exactly preserved** (admit set unchanged on this pool; the hardened gate
only strengthens the basis of trust from a single solver's float recheck to a verified-PD proof
across {CLARABEL, SCS}).

This verdict does NOT flip the default — `make_sdp_verifier_rump_or` remains an additive, opt-in
factory. Promotion is a separate, explicit human decision. No EXTRA (sound) delta exists to report
for a human judgement call; the recommendation is simply: promotion is safe whenever desired.

## Honest caveats

- Result is on the n=2 coupled substrate, the seed=2024 EXP-A pool (300 genes), margin=1e-7,
  t_domain=`tmin1`, with CLARABEL **and** SCS both installed (cvxpy 1.9.1). A different pool / seed
  / dimension / solver set could in principle surface a non-zero (sound, by the invariant) delta.
- "All extra admits sound" is **vacuously** true here (0 extras). The JSR oracle (max product
  length 6, Gripenberg lower bound) was wired and would flag any `jsr_lb ≥ 1` extra as unsound; it
  was simply never exercised because there were no extras.
- The OR's value (CLARABEL-vs-SCS disagreement near the feasibility boundary) is measured separately
  in `exp_rump_or.py` / `rump_or_experiment_results.json`; on THIS pool the two gates agree exactly,
  so the OR is not load-bearing for the admit COUNT here — its value is robustness, not coverage.
