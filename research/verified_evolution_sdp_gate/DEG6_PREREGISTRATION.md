# PRE-REGISTRATION — degree-6 Lyapunov & the two frontiers of certificate strength (2026-06-03)

> Roadmap item **#5** (skeleton extension). Written **before** the gated experiments are run
> (scouts informed design/feasibility only; the gates + honest-null below are committed here).
> `research/` only, `src/` untouched, honest disclosure ([[feedback_benchmark_honest_disclosure]]).

## Thesis under test

The arc established **"a stronger sound verifier unlocks more reachable safe fitness"** at the
inf→2-norm→SDP rungs (rotation 0.41→0.86). Roadmap #2 added a **degree-4** Lyapunov rung that
recovered 33 % of the Track-D *D4 residual* — but measured only **certification coverage on a
random pool**, never an **evolution payoff**. This pre-registration separates two distinct
frontiers of certificate strength and tests where each one moves:

- **Coverage frontier** = fraction of empirically-contracting genes a verifier can *certify*.
- **Capability frontier** = best *objective fitness* an evolution loop can *reach* when gated
  by that verifier (a stronger gate can only help if the global optimum lives in the region it
  newly admits and is unreachable by the weaker region).

## Substrate / config (committed)

- Substrate: coupled **n=2** RWKV map (`coupled_components.CoupledGeneCodec`, dim=6), decay∈[0,1]²,
  W∈[−2,2]²ˣ². Verifier ladder (nested admission sets):
  `L0=inf ⊆ L1=inf∪2norm ⊆ L2=sdp ⊆ L3=sdp∪deg4 ⊆ L4=sdp∪deg4∪deg6`.
- Determinism: every random draw threaded through a seeded `np.random.Generator`; paired/CRN
  comparisons. Empirical contraction oracle = `empirical_spectral_radius` (from-below sup of
  ρ(J) over the (s,x) box; a SOUND gate's admits must be <1).
- Soundness re-check inside every certifier: P≻0 and each decrease-LMI ≻0 verified by an
  independent eigen-decomposition (never solver-blind); ρ(J_v)<1 pre-screen at every t-box vertex.

## EXP-A — coverage frontier + complementarity (pool, n=2)

Pool: N≥300 empirically-contracting genes (seed=2024). Cumulative union sizes L0..L4.

| Gate | Pre-registered prediction | Falsifiable gate |
|---|---|---|
| **G-A1** coverage advances | L4 > L2 (deg-union certifies strictly more than quadratic class) | **PASS iff** L4 − L2 ≥ +5 genes |
| **G-A2** ladder non-nested | both deg4∖deg6 ≥ 1 **and** deg6∖deg4 ≥ 1 over the pool | **PASS iff** both > 0 |
| **G-A3** soundness | every deg6-certified gene has empirical ρ<1 at 50 000 samples | **PASS iff** 0 unsound |

Honest-null pre-committed: if L4 − L3 = 0 (deg6 adds nothing beyond deg4) → "coverage saturates
at deg4"; if G-A2 fails (nested after all) → report the ladder as nested. Both are valid outcomes.

## EXP-B — capability frontier / evolution payoff (n=2)

`evolve()` (pop=24, gens=25, elitism=1, k=3, σ=0.15, resample_cap=50, gate_initial=True) with
each ladder gate L0..L4, **n_seeds=15**, paired CRN (same per-seed rng across gates), reach =
best fitness. Two objectives:

| Objective | Role | Pre-registered prediction |
|---|---|---|
| **rotation** | POSITIVE CONTROL (known inf→2-norm payoff) | L1..L4 reach ≈0.86 and ≫ L0 (≈0.41) — validates the harness |
| **residual-reach** | the NEW test: target = free response of the **best quad-rejected, deg4/deg6-certified** reference gene found by a fixed seeded search (honest reachability framing, like `NonNormalObjective`) | tests whether L3/L4 reach strictly higher than L2 |

Strict gate (project standard): paired one-sided Wilcoxon p<0.05 **and** |paired_sign_delta|≥0.147
**and** n≥15.

| | Gate | Falsifiable |
|---|---|---|
| **G-B1** harness valid | rotation: L1−L0 passes the strict gate (positive control reproduces the known inf→2norm payoff) | PASS iff strict-gate(rotation, L1 vs L0) |
| **G-B2** capability payoff | residual-reach: L4 (or L3) reach > L2 reach | PASS iff strict-gate(residual, L4 vs L2) |

**Honest-null pre-committed (scout-anticipated):** the scout found the residual's achievable
transient amplification is weak (≤1.36) so SDP-certified genes approximate residual targets to
R²≈0.90, gap ≤ ~0 — therefore **G-B2 is expected to be NULL**. A null G-B2 is the committed,
non-rationalised conclusion: *"the capability frontier saturates at SDP; degree-4/6 advance
coverage, not capability, on this substrate."* (This is the sharper, honest finding, not a
failure to explain away.) G-B2 PASS would instead show a genuine deg-rung capability unlock.

## EXP-C — does the capability *potential* grow with substrate dimension? (coupled_nd, n=2,3,4)

Mechanism hypothesis: non-normal transient amplification grows with dimension, so the *potential*
capability gap of higher-degree verifiers should grow with n. Measure, per n∈{2,3,4}, the **max
achievable transient amplification of quad-rejected, empirically-contracting (residual) genes**.

- **G-C** dimension trend: max residual transient amp is **monotone increasing** in n.
  - PASS → the capability payoff of higher-degree verifiers is *dimension-gated* (returns in
    high-dim / full-LLM regimes — motivates the GPU bet structurally).
  - FAIL (flat/decreasing) → capability saturation is substrate-intrinsic, not dimensional.
  Either way is a committed, reportable outcome.

## JSR honesty oracle (attribution of the irreducible residual)

For genes uncertified even by L4 (deg6): bracket the true box contraction rate. JSR **lower**
bound = max over sampled length-k Jacobian products of ρ(∏J)^{1/k}. If JSR_lb ≥ 1 → the gene is
box-expansive (the empirical (s,x) oracle simply never sampled the worst corner / the t-box
over-approx is loose) — **NOT** a finite-degree limitation. If JSR_lb < 1 yet L4 fails → a genuine
finite-degree gap (degree-8+/exact-JSR would be the next rung). This attributes the residual
honestly rather than calling all of it a "verifier weakness".

## Red-team (committed lenses)

1. **Soundness** — 50 000-sample independent from-below ρ oracle over **all** deg6/deg4 admits in
   every experiment; require 0 unsound.
2. **Admission-size artifact** — re-run EXP-B with a **random (gene-keyed deterministic) fitness**;
   a real capability payoff must vanish (reach ≈ tie across gates). If a "payoff" survives random
   fitness it was an artifact of how many genes each gate admits.
3. **Circularity** — verify the residual-reach reference gene is chosen **independently** of the
   gate under test and is genuinely quad-rejected (not secretly SDP-certifiable); confirm its
   self-R²=1 and that a quad-certified gene cannot trivially reproduce its target by symmetry.
4. **Numerical robustness** — vary the SDP margin (1e-6..1e-8); G-A counts must not flip the
   verdict; the independent eigen re-check (not the solver's "may be inaccurate" status) is the
   sound authority.

## Deliverables

`verifier_deg6.py` (done, 13 tests), `exp_deg6_ladder.py` (EXP-A), `exp_deg6_capability.py`
(EXP-B), `exp_deg6_dimension.py` (EXP-C), `jsr_bracket.py` (oracle), `redteam_deg6.py`,
`DEG6_VERDICT.md`. No push (llcore remote not created — exposure avoidance). Codex pair-review
best-effort (file-read has been timing out systemically — attempt, do not block).
