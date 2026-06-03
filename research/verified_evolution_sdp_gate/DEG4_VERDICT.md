> ⚠️ **CORRECTION (2026-06-03 audit, `research/AUDIT_SCS_CLARABEL_2026-06-03.md`):** the "D4 residual
> 172, recovered 57 (33%)" below was measured with cvxpy's **SCS** default, which false-negatives near
> the SDP feasibility boundary and **inflated the residual ~5.5×**. Under the accurate **CLARABEL**
> solver: **D4 residual = 31, recovered by deg4 = 4 (12.9%)**. The "33% recovery" is largely an SCS
> artifact; the true degree-4 contribution is marginal (the quadratic SDP already certifies ~95%).
> Soundness unaffected (0 unsound). Read with the audit.

# VERDICT — degree-4 (non-quadratic) Lyapunov VerifierBackend (2026-06-03)

> Roadmap item #2 (skeleton extension): add a richer `VerifierBackend` that reaches the
> **Track-D D4 residual** — genes ρ(J)<1 over the box that NO fixed induced norm and NO
> common *quadratic* Lyapunov P certifies. `research/`, src untouched, honest disclosure.

## 1. What was built (new VerifierBackend plug-in)

`verifier_deg4.py`: a **degree-4 homogeneous Lyapunov** certificate
V(z)=m(z)ᵀ P m(z), where m(z) is the degree-2 monomial (Veronese) vector. It certifies
contraction via a common-P LMI on the **symmetric 2nd Kronecker power** J^[2] of each t-box
vertex Jacobian (Parrilo–Jadbabaie SOS hierarchy, k=2). This is a strictly richer class than
the common quadratic (k=1) that `cert_sdp` uses. **Only the backend is new** — `evolve()` and
the codecs are untouched; `make_deg4_verifier_n2()` returns a drop-in `VerifierBackend`
(`sdp OR deg4`).

The symmetric-power construction `sym2_power(A)` was verified correct vs brute-force monomial
transforms for n=2,3,4 (max error 0). The certifier independently re-checks the solver's P
eigenvalues (never solver-blind) and pre-screens ρ(J_v)<1 at every vertex.

## 2. Result (1200 random n=2 genes)

| quantity | value |
|---|---|
| empirically contracting (ρ<1) | 471 |
| **D4 residual** (ρ<1 but inf/2-norm/quadratic-SDP all reject) | **172** |
| **recovered by degree-4** | **57 (33.1 %)** |
| degree-4 certs the quadratic class could not give | 57 |
| **degree-4 unsound certs** (cert but empirical ρ≥1) | **0** |

(n=400 replicate: 58 residual, 17 recovered = 29.3 %, 0 unsound — consistent.)
`test_deg4.py`: symmetric-power correctness, soundness, residual recovery, sdp⊆sdp_deg4.

## 3. Reading

- **The next frontier rung is reached on CPU.** The CPU-Verification arc ended with "the
  right verifier is SDP/Lyapunov (quadratic)" and the integration verdict noted SDP still
  leaves a **D4 residual** — the highest-fitness contracting dynamics it cannot prove. A
  **degree-4 (non-quadratic) Lyapunov** recovers **~1/3** of that residual, *soundly*. The
  verifier-fitness frontier now reads:

      inf-norm  →  2-norm  →  quadratic SDP  →  **degree-4 Lyapunov (+33 % of residual)**  →  …(degree-6 / JSR-exact)…  →  true contraction set

- **Honest limits:** (a) recovery is **partial (33 %, not 100 %)** — even degree-4 leaves
  a residual (degree-6+ SOS or an exact JSR routine would be the next rung). (b) The lifted
  full-space LMI is a **sufficient** (slightly conservative) form of the degree-4 SOS
  condition (it imposes the decrease on all of Rⁿ⁽ⁿ⁺¹⁾ᐟ², not only the Veronese variety), so
  the true degree-4 reach is ≥ what we certify. (c) Sound: 0 unsound certs over 1200 genes
  (consistency oracle; the guarantee itself is the Lyapunov certificate). (d) cvxpy required
  (optional `[sdp]` extra); degrades to `False` if absent (fail-closed).

## 4. Bottom line

✅ A **non-quadratic (degree-4) Lyapunov** VerifierBackend plugs into the unchanged skeleton
and **soundly recovers ~33 % of the Track-D D4 residual** the quadratic SDP misses — the next
step on the verifier-fitness frontier, on CPU. This directly extends the arc's "stronger
verifier ⇒ more reachable safe fitness" with a concrete stronger verifier.

Artifacts: `verifier_deg4.py`, `test_deg4.py`, `exp_deg4.py`, `exp_deg4_results.json`. No push.
Next roadmap item: #3 — promote a verifier backend into `src/` (Stage 3b plugin, cvxpy optional).
