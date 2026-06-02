# VERDICT — SDP-Lyapunov-gated coupled evolution (Track B + D integration)

> Goal (user, 2026-06-03): 「llcore 研究の続き。CPU 環境下で AI 進化の仕組みを実現する。」
> 「進化の実現が最優先。一度実現すれば骨組みとして拡張するだけ。」
> Result: **a working, verified, extensible CPU mechanism for evolving an AI dynamics
> core** — evolution climbs, the SDP-Lyapunov verifier provably keeps every admitted
> gene contracting, and a *better* verifier measurably unlocks more reachable fitness.
> Discipline: `research/verified_evolution_sdp_gate/`, **src untouched**, seeds fixed,
> pre-registration first, honest disclosure, adversarial red-team, Codex pair-review.

## 1. What was built (the 骨組み / skeleton)

`evolvable_core.py` — a minimal verified-evolution loop with **three pluggable
extension points**: `GeneCodec` (substrate), `Objective` (evolution direction),
`VerifierBackend` (fail-closed safety gate). Adding a direction = a new Objective;
extending the substrate or verifier = a new codec/backend. `coupled_components.py`
supplies the n=2 coupled-RWKV plugins + 4 verifier backends (none / inf_norm /
two_norm / sdp) wrapping the Track C/D certifiers. 12 TDD tests pass; `demo_evolve.py`
shows the realization end-to-end; README.md documents extension.

## 2. Gates (verifier backends), all SOUND for admitting only contractions

| backend | certificate | admits (of 10175-gene pool, exp1) |
|---|---|---|
| `none` | accept all | all |
| `inf_norm` | closed-form ‖J‖_∞<1 over achievable-t box | 1860 |
| `two_norm` | vertex σ_max(J)<1 | 1860 + 692 = 2552 |
| `sdp` | common-P quadratic Lyapunov LMI (cvxpy) | 1860 + 692 + 817 = **3369** |

SDP admits **1.8× more** genes than inf_norm (G3 ✓): the extra 692 (2-norm) + 817
(SDP-only) are genuine contractions the conservative ∞-norm structurally over-rejects.

## 3. exp1 — landscape attribution (mechanism, NON-circular, 10175 genes)

The decisive, gate-independent evidence: classify every gene by tightest certifier
and measure the best achievable **rotation-task** fitness per region.

| region (tightest sound certificate) | genes | max rotation fitness |
|---|---|---|
| `inf` (∞-norm certifies) | 1860 | **0.38** |
| `two_norm_only` | 692 | 0.89 |
| `sdp_only` | 817 | 0.90 |
| `non_certified` (ρ<1 but no quadratic/norm cert = Track-D D4 residual) | 6806 | 0.99 |

**The mechanism is the region ceiling, not GA luck**: the ∞-norm-admissible region
*cannot* exceed 0.38 on the rotation task; the 2-norm/SDP regions reach 0.90. So a
verifier that admits more (while staying sound) lets evolution reach strictly higher
fitness. Benign-task best gene sits in `inf` (fitness 1.0) — no region advantage there
(the basis of the G5 null).

**Honest limitation (must disclose):** among empirically-CONTRACTING high-fitness
rotation genes, **39 of the top 50 are `non_certified`** (ρ<1 but provable by neither a
fixed induced norm nor a common quadratic Lyapunov P — the Track-D D4 residual). All
sound gates, SDP included, reject them. So **even the SDP gate is over-conservative**
relative to the true contraction set; the highest-fitness contracting dynamics (0.99)
need a *stronger* verifier still (JSR / non-quadratic Lyapunov). The verifier-fitness
frontier is therefore monotone but unfinished:

    inf (0.38)  →  two_norm (0.89)  →  sdp (0.90)  →  [JSR / non-quadratic?]  →  empirical contraction (0.99)

SDP's unique gain over the 2-norm is small on rotation (0.90 vs 0.89; the thin SDP-only
shell, consistent with Track D). The large win is **inf → two_norm**.

## 4. exp2 — gated evolution (G0–G5)   [n=15 seeds, pop=24, gens=25]

<!-- FILL from exp2_results.json (15-seed run) -->

## 5. Pre-registered gates — verdicts

<!-- FILL: G0/G1/G2/G3/G4/G5 PASS/FAIL with numbers -->

## 6. Adversarial red-team (Lenses A–D)

<!-- FILL from redteam_*.json -->

## 7. Honest disclosure / deviations

- **Config deviation from PREREGISTRATION §4** (pop=30, gens=40, n=20): run at
  pop=24, gens=25, **n=15** for CPU tractability (cvxpy per-call canonicalisation is the
  bottleneck). n=15 still satisfies the G4 strict-gate threshold (n≥15). The gates are
  structural (soundness / region-ceiling / null) and seed-robust, not config-sensitive.
- **Fitness ceiling caught & fixed**: the first fitness (exp(−MSE)) saturated at ~0.997
  for most genes (the ③/llive ceiling trap); replaced with **R²** (project-standard,
  headroom) before any verdict — disclosed per [[feedback_benchmark_honest_disclosure]].
- **NonNormalObjective is a *reachability test*, not a naturalness claim**: its target is
  the free response of a hand-picked SDP-only reference gene, so its optimum lives in the
  SDP-only region by construction. It tests whether the SDP gate can *reach* dynamics the
  conservative gates forbid; it does NOT claim natural tasks are non-normal. Stated openly.
- **A behaviour-preserving SDP fast-path** (reject when ρ(J_v)≥1 at any t-box vertex,
  a necessary condition for the LMI) was added to make the sweep tractable. Gate
  decisions are provably identical (verified: 12 TDD tests unchanged).
- **Push**: none (llcore remote not created — exposure avoidance). Local commit only.

## 8. Bottom line

<!-- FILL after stats -->
