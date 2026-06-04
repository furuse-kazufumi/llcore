# R1 VERDICT — the quadratic-SDP contraction-completeness result DEGRADES with dimension (does NOT generalise)

> Honest-disclosure project (post-SCS-artifact). Pre-registered gates: `PREREGISTRATION_R1.md`.
> Experiment: `exp_dim_completeness.py`. Raw results: `dim_completeness_results.json`. Date: 2026-06-04.
> **CLARABEL confirmed as the pinned SDP solver** (`_CLARABEL_OK=True`, `_SOLVER=CLARABEL`,
> `installed=['CLARABEL','SCS','SCIPY','HIGHS','OSQP']`). Never SCS.

## Bottom line

The n=2 finding — a **common quadratic Lyapunov SDP certifies ~95 % of empirically-contracting genes** — does
**NOT generalise** to higher coupled dimension. It **DEGRADES sharply and monotonically** as n grows:

| n | pool (contracting) | scanned | ∞-norm | +2-norm (cum) | **+quadratic SDP (cum)** | **SDP %** | residual (contracting, no quad cert) | residual % |
|---|---|---|---|---|---|---|---|---|
| 2 | 200 | 555  | 70 | 97  | **184** | **92.0 %** | 16  | 8.0 %  |
| 3 | 200 | 1377 | 6  | 21  | **159** | **79.5 %** | 41  | 20.5 % |
| 4 | 200 | 4876 | 0  | 0   | **96**  | **48.0 %** | 104 | 52.0 % |

**SDP-completeness drops 44 points from n=2 (92.0 %) to n=4 (48.0 %).** The contracting-but-uncertified residual
grows from 8 % → 20.5 % → **52 %**. At n=4 the quadratic SDP misses **more than half** of the empirically
contracting dynamics, and the cheap ∞-norm / 2-norm certificates collapse to **0** (every contracting n=4 gene
in the pool needed the full SDP and even then half failed).

**G4 verdict = `degrades`.** This satisfies the pre-registered "DEGRADES" definition on all three counts:
p_n falls below 0.90 at both n=3 and n=4; the n=2→n=4 drop (0.44) far exceeds the 0.10 threshold; the n=4
residual fraction (0.52) far exceeds the 0.15 threshold.

## Pre-registered gates

| gate | criterion | result |
|---|---|---|
| **G1** completeness ≥ 0.80 at every n | n=4 is 0.48 | **FAIL** (n=4 below 0.80; n=3 at 0.795 also below) |
| **G2** SDP a (near-)superset of 2-norm: `two_beats_sdp == 0` ∀n | 0 / 0 / 0 | **PASS** |
| **G3** 0 unsound admits ∀n | 0 / 0 / 0 | **PASS** |
| **G4** generalises vs degrades | see above | **DEGRADES** |

G1 failing is not a soundness problem — it is exactly the *measurement* of the degradation: the quadratic SDP
class is increasingly **incomplete** (rejects genuinely-contracting dynamics) as the coordinates couple.

## Soundness — 0 unsound admits at every n (the result is honest in BOTH directions)

Soundness was checked **independently of the solver** for every SDP-certified gene via two from-below oracles:
- (a) high-sample empirical spectral radius `empirical_rho_fast(g, n_samples=20000)` (vectorized; verified to
  match `coupled_nd.empirical_rho` to 1.1e-15 on the same seed — machine epsilon, identical math, 126× faster);
- (b) an n-dim JSR product lower bound over the 2^n vertex Jacobians (`jsr_lb`, generalising
  `verify_certifier_jsr_soundness.jsr_lb`), product length capped per n (n=2:≤6/4 verts, n=3:≤4/8 verts,
  n=4:≤3/16 verts) so the product count stays at 4096.

**No SDP-certified gene at any n had ρ_sound ≥ 1 or jsr_lb ≥ 1. n_unsound = 0 / 0 / 0.** Worst observed values
over certified genes: max ρ_sound = **0.99999** (n=2, single near-boundary gene — strictly < 1, exact value
0.9999921987890266; the `1.0` in the JSON is a 4-decimal rounding of that), 0.9994 (n=3), 0.9998 (n=4); max
jsr_lb = 0.9998 (n=3,n=4). The certifier is **sound** at every dimension — it does not admit non-contracting
dynamics; it merely **rejects more and more genuinely-contracting dynamics** as n grows. That is the degradation:
a growing **false-negative** (conservatism) rate, never a false admit.

> Caveat (carried from the n=2 audit, pair-review F2): the JSR oracle is a one-sided, finite-length lower bound
> (here additionally length-capped for tractability at n≥3); `jsr_lb<1` is necessary-but-not-sufficient. The
> soundness guarantee rests on the certifier theorem (vertex-LMI ⇒ hull stability by matrix convexity for the
> quadratic class) plus the independent float-eigen re-check of the solver's P inside `cert_sdp`. The oracles are
> falsification checks, and they falsified nothing.

## Why it degrades (interpretation, consistent with the pre-registered "degrades → meaning")

As more coordinates couple (W ∈ [-2,2]^{n×n}), an increasing share of contracting dynamics are **non-normal /
rotationally-coupled**: contracting in some quadratic metric that *varies with state*, which a single common
quadratic Lyapunov P (one ellipsoid for all 2^n vertices simultaneously) cannot cover. The ∞-norm and 2-norm
(P=I) certificates collapse first (0 at n=4), then the common-P SDP itself increasingly fails. This is the
predicted failure mode of a single quadratic certificate as dimension grows, and it motivates higher-degree /
non-monotone / path-complete (multiple-Lyapunov) certificates at higher n — the residual genes are saved to
`dim_completeness_residual_genes.json` for exactly that future work.

This is the opposite of the n=2 headline, and it is reported plainly as the finding. It does **not** contradict
the n=2 audit (which stands: at n=2 the SDP is ~95 % complete and the degree-ladder beyond it is a thin tail).
It shows the **completeness** of the quadratic class is **dimension-specific**, not a dimension-robust property.

## Exact numbers, seeds, budget

- Master seed 4242; per-n seed = 4242+n (n=2→4244, n=3→4245, n=4→4246). Fully reproducible:
  `py -3.11 exp_dim_completeness.py --n-target 200 --seed 4242 --sound-samples 20000`.
- Pool 200 contracting genes per n. Pool screen: `empirical_rho_fast(g, n_samples=4000) < 1`. Soundness:
  `n_samples=20000` + JSR.
- Acceptance rate falls with n (200/555 ≈ 36 % at n=2, 200/1377 ≈ 15 % at n=3, 200/4876 ≈ 4 % at n=4) — random
  W∈[-2,2]^{n×n} contracts far less often as n grows (independent confirmation that higher-n contraction is
  rarer / harder, consistent with the certifier degradation).
- Total elapsed 507 s. One CLARABEL `optimal_inaccurate` UserWarning fired at n=3; `cert_sdp` accepts that status
  only after its independent float-eigen re-check of P, and the JSR/ρ oracles confirmed 0 unsound — so it did not
  produce a false admit. (An earlier 829 s run with the pure-Python `coupled_nd.empirical_rho` gave the identical
  n=2 coverage: inf=70, two_cum=97, sdp_cum=184 (92.0 %), residual=16, 0 unsound — coverage counts are
  deterministic and solver/throughput-independent, as pre-registered.)

## Limits (as registered)

- Pool is a *sample* (200/n) of contracting dynamics, not the full contracting set; the percentages carry a
  ±few-% sampling CI. The **44-point** drop is far larger than any plausible CI, so the *direction and magnitude*
  of the degradation are robust even if the exact percentages shift a couple of points with seed.
- JSR oracle length-capped at n≥3 for tractability (one-sided lower bound; caveat above).
- n ∈ {2,3,4} only; no claim beyond n=4 (though the monotone 92→79.5→48 trend and the collapsing acceptance rate
  both suggest continued decay).
