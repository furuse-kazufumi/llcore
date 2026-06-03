# PRE-REGISTRATION — degree-8 SOS + JSR-bracket: closing the coverage frontier (2026-06-03)

> Roadmap **#6** (user-chosen thrust: "coverage 深掘り — degree-8 / JSR-exact"). Written **before**
> the experiments. Builds on the verified deg6 result (DEG6_VERDICT.md): of the 53 deg6-residual
> genes, **47 (89 %) are finite-degree-gap candidates** (JSR_lb<1 but degree-6 cannot certify),
> with JSR_lb up to **0.9974** (near-boundary). `research/`, src untouched, honest disclosure.

## Question

The SOS Lyapunov ladder (deg4→deg6) is a CPU approximation of the exact joint spectral radius
(JSR) of the t-box vertex set {J_v}. **How far must the ladder climb to certify the full
contracting set, and does a JSR bracket close the residual?** Two instruments:

1. **degree-8 SOS Lyapunov** = the symmetric 4th Kronecker power lift (`certify_degN(·, degree=4)`,
   already general). The next rung above degree-6.
2. **JSR bracket** per gene: lower bound `jsr_lb` = max over length-≤K vertex products of ρ(∏)^{1/k}
   (Gripenberg); upper bound `γ*_d` = smallest γ such that {J_v/γ} admits a common degree-d
   Lyapunov (bisection on the scaled lifted LMI). The true JSR ∈ [jsr_lb, γ*_d]. A gene is
   **certified contracting at the SOS-degree-d limit** iff γ*_d < 1.

## Substrate / soundness (committed, same as deg6)

- coupled n=2, t-box vertex Jacobians. Sound certificate = common lifted Lyapunov; **soundness
  re-checked by independent eigendecomposition** (P≻0 and every decrease-LMI ≻0) AND by the JSR
  oracle (every certified gene must have jsr_lb<1, else a soundness bug). cvxpy fail-closed.
- Scaling identity: `sym_power(J/γ, k) = sym_power(J, k)/γ^k`, so testing the degree-d Lyapunov on
  the γ-scaled vertices is a sound SOS upper bound on JSR (Parrilo–Jadbabaie; the lifted full-space
  form is sufficient/conservative, so γ*_d is an upper bound on the true SOS-degree-d bound, hence
  on JSR only up to the lift conservativeness — disclosed; certify only when γ*_d<1 strictly).

## Targets = the 47 finite-gap residual genes (from `exp_deg6_residual_genes.json`)

| gate | pre-registered prediction | falsifiable gate |
|---|---|---|
| **G-8A** degree-8 advances coverage | degree-8 certifies ≥1 finite-gap gene that deg6 cannot | **PASS iff** degree-8 recovers ≥1 of the 47 beyond {sdp∪deg4∪deg6} |
| **G-8B** ladder still non-nested | deg6∖deg8 and deg8∖deg6 both non-empty over the residual pool | report counts; PASS iff both>0 (else "nested at this rung") |
| **G-8C** soundness | every degree-8-certified gene has jsr_lb<1 @ max_len≥6 | **PASS iff** 0 certified with jsr_lb≥1 |
| **G-JSR** bracket closes residual | γ*_8 < 1 for a MAJORITY of the 47 finite-gap genes (the bracket [jsr_lb, γ*_8] certifies them) | report the fraction with γ*_8<1; PASS iff > 50 % |
| **G-tight** bracket tightens with d | mean (γ*_d − jsr_lb) decreases d=2→3→4 (quad→deg6→deg8) | report the trend |

## Honest-null / caveats (pre-committed)

- **Near-boundary genes may stay uncertified.** The finite-gap genes have jsr_lb up to 0.9974;
  certifying JSR<1 for a gene whose true JSR is ~0.997 needs a Lyapunov with enormous condition
  number — degree-8 may still miss them (SOS conservativeness + numerical limits). **A partial
  recovery (e.g. degree-8 closes 40–70 %, not 100 %) is the committed honest outcome**, and would
  show the coverage frontier asymptotes rather than closes at any finite CPU degree.
- **"Exact JSR" is NP-hard.** We report a *tight bracket* [jsr_lb, γ*_d], NOT an exact value.
  Genes with jsr_lb<1<γ*_d remain genuinely inconclusive at degree d (need degree-10+/branch-and-
  bound). Stated openly — no gene is called "expansive" unless jsr_lb≥1.
- **Numerical**: degree-8 = 5×5 (n=2) lifted P near the contraction boundary → solver
  "inaccurate" warnings expected; the independent eigen re-check + jsr_lb<1 cross-check, not the
  solver status, are the soundness authority. Re-run borderline γ*_8 at tightened margin.

## Red-team (committed)

1. **Soundness** — jsr_lb (max_len 6) on every degree-8/γ*_8-certified gene; 0 with jsr_lb≥1.
2. **Bracket validity** — for ≥20 genes, confirm jsr_lb ≤ γ*_d (lower ≤ upper) and that γ*_d is
   monotone non-increasing in d (deg4≥deg6≥deg8 scaled bounds) — a violation = a bisection/lift bug.
3. **No double-counting** — degree-8 "recovery" counts only genes NOT already certified by
   sdp∪deg4∪deg6 (strict residual).
4. **Complementarity artifact** — re-check G-8B genes at margins 1e-6..1e-8 (borderline flips).

## Design note (post-gates, from the `verifier_jsr` smoke — disclosed, gates unchanged)

The smoke revealed the LIFTED full-space SOS family is **NON-MONOTONE in degree**: γ* can *increase*
deg2→deg8 (0.767→0.780 on a near-normal gene), because the full-space LMI imposes the decrease on
the whole lifted space (not only the Veronese variety), and that conservativeness grows with the
lift dimension. So **G-tight (bracket shrinks monotonically with d) is EXPECTED TO FAIL** — and that
*is* the JSR-bound face of the deg4/deg6 complementarity. The tightest valid upper bound from this
family is therefore **γ*_min = min_d γ*_d**, and the lifted-union coverage is {γ*_min < 1} (this is
what `exp_jsr_bracket.py` reports). A genuinely monotone bound converging to exact-JSR needs a
**proper SOS-on-variety program** (SOS multipliers for the Veronese ideal) or **branch-and-bound JSR**
(NP-hard) — scoped as the rigorous next step, not claimed here. Gates above are unchanged; G-tight is
reported as the (committed) honest negative.

## Deliverables

`verifier_jsr.py` (γ*_d bisection bracket; degree-8 via `certify_degN(·,4)`), `exp_deg8_ladder.py`
(degree-8 recovery + complementarity + soundness on the 47 finite-gap), `exp_jsr_bracket.py`
(bracket per residual gene; fraction closed by γ*_8), `test_deg8_jsr.py`, `redteam_deg8.py`,
`DEG8_JSR_VERDICT.md`. No push (exposure avoidance). Begin only after DEG6_VERDICT.md is locked
(red-team + adversarial review incorporated).
