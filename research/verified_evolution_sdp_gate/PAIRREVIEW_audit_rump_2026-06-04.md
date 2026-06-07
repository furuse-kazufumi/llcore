# Pair-review — SCS→CLARABEL audit + Rump/OR hardening + deg6/8/JSR (2026-06-04)

The evening 2026-06-03 work (CLARABEL re-audit of the arc, the Rump-verified-PD + OR-of-solvers
hardening, the degree-6/8/JSR ladder) landed its verdicts but its **Codex pair-review was left
incomplete** when the session exited. This document completes it, per
[[feedback_codex_pair_review_for_llcore]] + [[feedback_external_ai_verify]] +
[[feedback_cvxpy_pin_accurate_solver]].

Two independent adversarial reviewers were used (the arc's own lesson: a *single* solver — and, by
analogy, a single reviewer — is a single point of failure; margin-sweep red-teams are structurally
blind to solver artifacts, so the decisive checks are **solver-swap** and **multi-perspective**):

1. **Codex** (`gpt-5.4`, `codex exec -s read-only`) — adversarial review of 6 headline claims.
   Raw log: `_codex_audit_rump_output.txt` (4852 lines); prompt: `_codex_review_prompt_audit_rump.txt`.
2. **Adversarial multi-agent Workflow** (`llcore-audit-rump-pairreview`, 6 skeptics, each instructed
   to REFUTE one claim via real-code + independent recomputation, overturning refutations re-verified
   by a second agent). Corroboration summarised in the last section.

Every finding was then re-verified against the actual code by the orchestrator before action.

## Verification of the landed milestones (evidence, not assertion)

| claim | result | evidence |
|---|---|---|
| src suite green | **255 passed** (23.7s) | `pytest tests` |
| research suite green | **312 passed**, 2 skipped (→ **313** after the F1 regression test added below) | `pytest research`; the 2 skips are z3-fallback tests skipped *because z3 is present* (not masked failures) |
| 0 unsound | **CERTIFIER_SOUND_by_JSR: true**, 0 genes with jsr_lb≥1 (certified jsr_lb ≤ 0.966) | `verify_certifier_jsr_soundness_results.json` |
| no reachable SCS default | all REACHABLE cvxpy solves pinned to CLARABEL; `src/llcore/verifier/backends.py` is correctly fail-closed (`:216-217` refuses before solving when CLARABEL absent) | grep of every `.solve(`; one residual footgun fixed below (F1) |

## Codex findings — disposition (each re-verified, then fixed or disclosed)

| # | sev | claim challenged | verified? | overturns thesis? | disposition |
|---|---|---|---|---|---|
| F1 | high | "every cvxpy `.solve()` is pinned" | **TRUE** | no | **FIXED** — see below |
| F2 | high | "soundness confirmed" (JSR + float recheck framing) | **TRUE** | no (refines) | **DISCLOSED** — language narrowed |
| F3 | high | "Rump+OR preserved-or-grown, never shrunk (by domination)" | **TRUE** (direction inverted) | no | **CORRECTED** — see below |
| F4 | med | `verifier_deg6.py` docstring still says "complementary (non-nested)" | **TRUE** | no | **FIXED** — comment corrected |
| F5 | low | compare script doesn't prove the SDP+Rump branch is non-vacuous | **TRUE** | no | **DISCLOSED** — 149/286 noted |

**Codex's bottom line (confirmed):** *no finding overturns the central thesis* — the right contraction
verifier is SDP/Lyapunov and it is ~95% complete on this CPU pool — *but the publication over-claimed
fail-closed completeness and soundness-proof strength; those were narrowed before release.*

### F1 (high, FIXED) — falsy-solver footgun → fail-closed
`research/spectral_lyapunov_contraction/lyapunov_sdp_certifier.py` had
`prob.solve(solver=_eff_solver) if _eff_solver else prob.solve()`. The `solver is None` guard at
`:107-110` does **not** catch a falsy non-`None` value such as `solver=""`, which would set
`_eff_solver=""` and fall into the bare `prob.solve()` (= cvxpy's SCS default, the artifact-prone
solver). Production callers never pass `""`, so this was a latent footgun, not an active bug — but
"every solve is pinned" was false as written.
- **Fix:** the bare `else prob.solve()` is removed; a falsy resolved solver now fail-closes
  (`available=False`, `solver_status="no_accurate_solver_fail_closed"`). The only remaining ways to
  run the SDP are an explicit accurate solver or the CLARABEL default.
- **Note (not a footgun):** the OR gate deliberately runs `solver="SCS"` (`rump_pd.py:286-301`,
  `rump_hardened_verifier.py:117-126`) — an *explicit* adversarial probe whose result is then
  Rump-re-verified; that is the OR's whole point, not an unpinned default.
- **Regression test added:** `test_fail_closed.py::test_lyapunov_certifier_falsy_solver_string_fail_closed`
  (asserts `solver=""` fail-closes with CLARABEL present). Suite: 9 → **10 passed**.

### F2 (high, DISCLOSED) — soundness wording narrowed
The certifier accepts `optimal`/`optimal_inaccurate` then gates with a float `eigvalsh>0` recheck (a
numerical check, not a float-level proof), and the JSR oracle is a **one-sided, finite-length (≤6)
lower bound**: `jsr_lb≥1` would prove unsoundness, but `0 such genes` is *necessary, not sufficient*.
The audit's "0 false admits / soundness intact" wording is now narrowed to **"0 *observed* false
admits"**, with soundness explicitly attributed to the certifier **theorems** (quadratic: vertex-LMI ⇒
hull stability by convexity; deg≥2: the lifted SOS sufficient condition) + the independent eigen
recheck. The Rump verified-PD gate is the only component giving a machine-checked float-level proof,
and it is additive (not the default). Edits: `AUDIT_SCS_CLARABEL_2026-06-03.md` (blockquote + §soundness).

### F3 (high, CORRECTED) — the preserve-or-grow rationale was inverted
The RUMP verdict argued the admit set is "preserved-or-grown, never shrunk" because `verified_pd`
*dominates* the float test. **Backwards.** `verified_pd` returns a **sound LOWER bound** on `λ_min`
(≤ the float-computed `λ_min`), so on the SAME matrix it is the **STRICTER** test
(`{Rump} ⊆ {float}` per-`P`) — it could in principle *reject* a barely-PD `P` the float test accepts.
- **Why 286==286 holds anyway (empirical, with headroom):** the SDP constraint enforces a
  `margin=1e-7` decrease while `verified_pd`'s Cholesky backward-error bound is `~n·u·maxdiag ≈ 1e-15`,
  so admitted `P`s clear the verified-PD test by **~8 orders of magnitude**.
- **Independent numerical confirmation (orchestrator):** 4000 barely-PD symmetric matrices with true
  `λ_min ∈ [1e-9,1e-5]` → **0** Rump rejections (so no shrink in the relevant regime) and **0** unsound
  lower bounds (`lb ≤ true λ_min` always). This both confirms the direction (verified_pd is the
  conservative/stricter test) *and* that the practical impact is negligible at this margin.
- **Soundness (no false positives) is unaffected** — that property of `verified_pd` is correct and
  proven. Edit: `RUMP_HARDENING_VERDICT.md` (CRITICAL INVARIANT § gets a ⚠️ CORRECTION block).

### F4 (med, FIXED) — stale "complementary/non-nested" docstring
`verifier_deg6.py::make_deg6_verifier_n2` still justified the union as "the lifted degree-4/degree-6
certificates are complementary (non-nested)" — contradicting the corrected, *nested* CLARABEL result
(`deg4∖deg6=0`). The union is still sound (a union of sound certificates is sound), but the *reason*
was the retracted SCS-era story. Docstring corrected to the nested framing. (The module-level
docstring already carried the retraction note; only this function docstring lagged.)

### F5 (low, DISCLOSED) — SDP+Rump branch is non-vacuously exercised
The compare script doesn't instrument how many admits used the SDP+Rump branch vs the inf/2-norm fast
path. Read off the same seed=2024 ladder (`exp_deg6_ladder_results.json`): inf=88 + two_only=49 = 137
fast-path, so **149 of 286** admits require the genuine SDP solve + Rump recheck. The 286==286 equality
reflects agreement on a real SDP-branch workload, not an all-fast-path artifact. Noted in the verdict's
honest caveats.

## Orchestrator-side independent verifications (not from Codex)

- **F3 direction + magnitude** — numerical battery above (0 rejections / 0 unsound LB).
- **backends.py is genuinely fail-closed** — read `:198-234`: `available` requires cvxpy AND CLARABEL;
  the SDP solve at `:224` is unreachable unless `_CLARABEL_AVAILABLE` (so `_SDP_SOLVER="CLARABEL"`).
  Codex correctly did NOT flag it.
- **F5 count 149/286** — read directly from `exp_deg6_ladder_results.json` `counts.sdp_only=149`.
- **test integrity** — the 2 research skips are z3-fallback tests (skipped *because* z3 is installed);
  no xfail/xpass masking; F1/F4 changes re-run clean (`verified_evolution_sdp_gate`: 73 passed).

## Adversarial multi-agent Workflow — corroboration

6 skeptics (each REFUTE one claim via independent recomputation; 564k tokens, 160 tool-uses, 45 min).
**Outcome: every Codex finding and disposition corroborated; no new thesis-overturning finding.**

| skeptic | claim | verdict | independent evidence |
|---|---|---|---|
| S1 | no reachable SCS default | **partially_refuted (low)** = Codex F1 | **reproduced** `solver=""` → bare `prob.solve()` → `solver_stats.solver_name=='SCS'`; traced all callers — none pass `""`/`0` (footgun latent, now fixed). Inaccuracy warnings are CLARABEL (SCS emitted 0), each followed by the independent eigen recheck. |
| S2 | no CLARABEL false positives | **upheld (none)** | 300-pool 286 admits (149 SDP-only) + 791 boundary SDP solves → **0 false admits, 0 eigen-recheck disagreements**, worst jsr_lb 0.9999<1 (stable at product length 8). |
| S3 | ladder retraction correct, not over-corrected | **upheld (none)** | reproduced SCS 23/13/18 vs CLARABEL 0/0/54 nested; SCS-era snapshot SHA-256 == git `cd400ef` (not overwritten); over-correction probe refuted (54 certified all jsr_lb<1; the 23 SCS-deg4_only all genuinely contracting); Track-D 1291/1363, 0 false positives. |
| S4 | Rump+OR 286==286, non-vacuous | **upheld (none)** | recomputed 286==286, 0 lost; **137 fast-path + 149 SDP+Rump branch** (rump recheck ran 298×); 610 actual certificate matrices: float-PD ⊆ rump-PD (0 lost). **Independently flagged the same F3 wording inversion** ("a sound LOWER bound is ≤ λ_min, not ≥") — confirms the correction. OR: CLARABEL 149 vs SCS 56 (93 disagreements), but CLARABEL alone covers the count → doc's "OR = robustness not coverage" caveat is accurate. |
| S5 | verified_pd no false positives | **upheld (none)** | ~7M adversarial matrices (300k + 3M lemma + 300k + 200k + 1.5M + 2M) → **0 false positives, 0 unsound lower bounds**; empirical ‖E‖₂ ≤ 0.59× the bound (40% headroom); γ_{n+1}/envelope/maxdiag/symmetrization derivation audited sound. |
| S6 | suite green + JSR framing honest | **upheld (none)** | src 255/0-skip, research 312+2skip (the 2 are z3-fallback, z3 confirmed installed); **0 xfail/xpass** (grep + `-rxX`, `--strict-markers`); JSR oracle re-run byte-matches; framing correct (`VERDICT.md:175-180` "falsify but not certify"). Minor: `DEG6_VERDICT.md:67`'s inline "→ CERTIFIER SOUND" arrow is loosely worded (not dishonest given context) → **tightened** in this pass for F2 consistency. |

Note: a *concurrent* full-suite run (launched alongside this 6-agent Workflow) showed 1 failure —
`gnn/test_gnn.py::test_verify_latency_under_threshold` (a wall-clock `mean<10ms` assertion). It is a
**CPU-contention flake**: it passed in the earlier isolated run (312 passed) and again in isolation
after the Workflow finished (`1 passed in 0.66s`). Not a regression; unrelated to the edited files.

## Net result

Central thesis **intact and unchanged**: on this CPU pool a common quadratic Lyapunov (SDP) certifies
~95% of contracting evolved dynamics; the degree ladder adds a tiny near-boundary residual; soundness
shows 0 observed false admits. The pair-review's value was **narrowing three over-claims**
(fail-closed completeness, soundness-proof strength, the preserve-or-grow rationale) and **closing one
latent footgun** (F1) — exactly what the discipline is for. No verdict's headline number changed.

Files touched: `lyapunov_sdp_certifier.py` (F1 fix), `verifier_deg6.py` (F4 doc), `test_fail_closed.py`
(F1 regression test), `AUDIT_SCS_CLARABEL_2026-06-03.md` (F2), `RUMP_HARDENING_VERDICT.md` (F3, F5),
this file. `src/` unchanged (255). Push deferred (llcore remote not created).
