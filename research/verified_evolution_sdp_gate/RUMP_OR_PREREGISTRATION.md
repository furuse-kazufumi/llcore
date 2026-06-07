# PRE-REGISTRATION — Pilot A: Rump verified-PD + OR-of-{CLARABEL,SCS} standing gate (2026-06-03)

> Pilot A from `VERIFICATION_METHODS_SURVEY_2026-06-03.md` (rank #1, ADOPT). Hardens the
> verified-evolution arc's soundness for **zero new dependencies** (pure numpy) by replacing the
> float-fragile `np.linalg.eigvalsh(... > 0)` certificate recheck with a **Rump verified
> positive-definiteness** test, and by baking "solver-swap is the decisive detector"
> (`AUDIT_SCS_CLARABEL_2026-06-03.md`, memory `feedback_cvxpy_pin_accurate_solver`) into a
> standing **OR-of-solvers** gate. This file is written BEFORE running the experiment; the
> invariants below are committed in advance.

## Scope discipline (committed)

- **research/ ONLY. src/ UNTOUCHED.** New file `rump_pd.py` + `test_rump_pd.py` only.
- **Additive.** Existing committed certifiers (`coupled_components._sdp_certifies`,
  `lyapunov_sdp_certifier.certify_common_lyapunov`, `verifier_deg4`, `verifier_deg6`) are NOT
  modified. INVARIANT: all existing verdicts/tests unchanged. The Rump gate is a NEW, stricter,
  self-contained layer that sits *beside* them; it never alters their behaviour.

## Method (committed before measurement)

**`verified_pd(M) -> (ok, lam_min_lb)`** — symmetrize `M`; attempt floating-point Cholesky of
`M - alpha*I`; using the classical RIGOROUS Cholesky backward-error bound (Demmel Thm 10.7 /
Higham Thm 10.3-4 / Rump 2006): a completed Cholesky implies `R^T R = A + E` exactly with
`||E||_2 <= gamma_{n+1} * n * max_i A_ii`, `gamma_k = k*u/(1-k*u)`, `u = 2^-53`. Since `R^T R`
is PSD, `lambda_min(M - alpha*I) >= -||E||_2`, hence `lambda_min(M) >= alpha - bound`. We search
alpha downward and return the first (largest) `alpha - bound > 0`. SOUND by construction:
`ok=True` only when `M` is PROVABLY PD; the returned `lam_min_lb` is a GUARANTEED lower bound.

**`or_certifies_lyapunov(gene, solvers=('CLARABEL','SCS')) -> bool`** — for EACH solver, solve the
common-Lyapunov SDP (reusing `certify_common_lyapunov`), take the returned `P`, and accept the
solver iff `P` AND every `(P - J_v^T P J_v - margin*I)` pass `verified_pd`. Return True if ANY
solver yields a Rump-passing certificate (**OR, NOT a vote** — a vote would preserve the SCS
false negative).

## Pre-registered invariants (falsifiable, committed before measurement)

| # | Invariant | Type | Falsified if... |
|---|---|---|---|
| **I1** | **Rump ⊆ true-PD** (SOUNDNESS). For all M: `verified_pd(M).ok == True` ⇒ exact `lambda_min(M) > 0`. ZERO false positives. | hard | any M with `eigvalsh_min <= 0` is Rump-certified |
| **I2** | **lower bound sound.** When `ok==True`, `lam_min_lb <= lambda_min(M)` (exact). | hard | any certified M has `lb > true_min_eig` |
| **I3** | **Rump ⊇ float-certified** (COVERAGE) on comfortably-PD M (`true_min >= 1e-6`). No coverage loss. | hard | a comfortably-PD M passes float `min>1e-9` but Rump rejects it |
| **I4** | **OR-gate ⊇ single-solver-Rump.** `or_certifies_lyapunov(g)` is True whenever EITHER solver's P passes Rump (the OR can only admit a superset of each single-solver Rump verdict). | hard | OR rejects a gene some single solver's P Rump-certifies |
| **I5** | **OR-gate is SOUND.** Every gene the OR-gate admits is genuinely contracting: its `jsr_lb < 1` (solver-independent Gripenberg oracle). | hard | an OR-admitted gene has `jsr_lb >= 1` (switched-expansive) |

The boundary band `(1e-9, 1e-6)` (between the float floor `1e-9` and Rump's ~`n*u*maxdiag`
guarantee threshold) is EXPLICITLY ALLOWED to be under-certified by Rump (the spec permits
under-certifying a barely-PD M). I3 only asserts no loss on *comfortably*-PD matrices.

## Experiment (committed before measurement)

**Population:** sdp-certifiable coupled genes near the SDP feasibility boundary — the "sdp_only
thin-shell" (~1e-7 margin). Generated deterministically via `coupled_components` /
`exp_deg6_capability.find_residual_reference`-style scanning, plus the committed near-boundary
genes in `exp_deg6_residual_genes.json` (`deg_certified` and `residual_uncert` lists). For each
gene we record, per t-box vertex certificate, the CLARABEL P and SCS P and run the Rump recheck.

**Pre-registered predictions (committed):**

- **P1 (coverage):** the Rump-certified set is a **superset** of the float-`eigvalsh`-certified
  set with **ZERO float-only false positives** — i.e. there is no gene whose certificate the float
  recheck accepts but whose vertex LMIs are NOT genuinely PD (after the Rump conservative
  margin). Equivalently: any "float-only" acceptance is a genuine near-boundary case, never a
  Rump under-certification on a comfortably-PD certificate. *(Honest nuance committed in advance:
  a certificate exactly AT the `margin` slack can have a vertex LMI numerically at/just-below 0;
  Rump will correctly reject THAT certificate — that is a sound rejection of a non-PD matrix, not
  a coverage loss. The OR-gate is the remedy: the OTHER solver typically supplies a certificate
  with positive slack.)*

- **P2 (solver-artifact measure):** we log how many genes **CLARABEL** certifies (Rump-verified)
  that **SCS** calls infeasible — the standing artifact-class measure. Direction of the count is
  measured, not assumed; the audit found SCS under-certifies near the boundary, so we expect
  `clarabel_recovers_over_scs >= 0` and report the real number. The OR-gate value = these
  recovered genes are admitted by the standing gate without any human re-run.

**Outputs:** `rump_or_experiment_results.json` (real measured numbers), reported in the
structured summary. If any invariant or prediction does NOT hold, it is reported honestly (the
honest self-correction is the credibility asset, per the survey's Fundability section).
