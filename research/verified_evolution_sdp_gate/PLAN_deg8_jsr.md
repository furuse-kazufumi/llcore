# EXECUTION PLAN — degree-8 / JSR-exact coverage thrust (2026-06-03)

Complements `DEG8_JSR_PREREGISTRATION.md` (gates) with the *sequence, dependencies, compute,
parallelism, decision tree, and packaging*. Goal-fit: CPU, additive (`research/`), src untouched,
honest disclosure. North star: **drive the verifier COVERAGE frontier to its JSR limit on CPU and
quantify how close finite-degree SOS gets** — the coverage half of the two-frontier thesis, made
rigorous (the fundable / surpass-llive deliverable).

## Phase 0 — LOCK deg6 first (gate: do not build on unverified ground)
- [in-flight] red-team `redteam_deg6.py` (soundness 50k, admission-artifact, circularity, margins).
- [in-flight] adversarial Workflow `_review_workflow.js` (6 skeptics → confirm → completeness critic).
- On completion: fill `DEG6_VERDICT.md` §8 (red-team), apply critic corrections, re-check any
  flagged claim. **Local git commit** (no push — exposure avoidance).
- **Decision:** any reviewer BLOCKER → fix before Phase 1. Otherwise proceed.

## Phase 1 — degree-8 SOS rung  (file: exp_deg8_ladder.py)   [~3-5 min]
- Input: the **47 finite-gap residual genes** (`exp_deg6_residual_genes.json`, the jsr_lb<1 set
  deg6 misses). degree-8 = `certify_degN(vertices, degree=4)` (already general; sym 4th power).
- Measure G-8A (recovery beyond sdp∪deg4∪deg6), G-8B (complementarity deg6/deg8), G-8C (soundness:
  jsr_lb<1 for all degree-8 certified). Also re-confirm sym_power(·,4) vs brute force (n=2,3,4).
- Smoke `verifier_jsr.py` first (γ*_d bisection sanity: bracket tightens deg1→2→3→4).

## Phase 2 — JSR bracket  (file: exp_jsr_bracket.py)   [~10-15 min]  ⟂ Phase 1 (parallelizable)
- For each of the 47 finite-gap genes: bracket [jsr_lb, γ*_d] for d=2,3,4 (deg4/6/8 SOS upper).
- Measure G-JSR (fraction with γ*_8 < 1 = certified at the degree-8 limit), G-tight (mean width
  γ*_d − jsr_lb shrinks with d). Characterise the **near-boundary tail** (jsr_lb≈0.99) that stays
  uncertified at any finite CPU degree.
- Run Phase 1 + Phase 2 as two background processes (independent inputs) to halve wall-clock.

## Phase 3 — full coverage frontier  (extend exp_deg6_ladder → degree-8 + JSR)   [~6-8 min]
- Re-run the 300-contracting-gene pool ladder with the extra rungs:
  L0 inf → L1 2norm → L2 sdp → L3 deg4 → L4 deg6 → **L5 deg8 → L_JSR (γ*<1)**.
- Headline table: cumulative certified of 300, all the way to the JSR limit. This is the paper's
  coverage-frontier figure.

## Phase 4 — verification (same discipline as deg6)   [~15-20 min]
- `test_deg8_jsr.py` (TDD): γ*_d monotone non-increasing in d; jsr_lb ≤ γ*_d (bracket valid);
  degree-8 sound; no double-counting.
- `redteam_deg8.py`: soundness (jsr_lb<1 on every certified), bracket validity (≥20 genes), margin
  1e-6..1e-8 robustness, strict-residual recovery counts.
- Adversarial Workflow (reuse `_review_workflow.js` pattern, retargeted) — skeptics try to refute
  "degree-8 advances coverage", "γ*_8 closes a majority", "bracket is valid/monotone".
- `DEG8_JSR_VERDICT.md` (honest: partial recovery + asymptote to the JSR=1 boundary is a committed
  valid outcome). Local commit.

## Decision tree (after Phase 2/3)
- **γ*_8 closes >50 % of the 47** → coverage frontier nearly closed on CPU. Strong result →
  package (Phase 5). Optional: degree-10 / branch-and-bound JSR for the tail.
- **γ*_8 closes <20 %** → coverage **asymptotes** at the JSR=1 boundary; finite-degree SOS cannot
  close the near-boundary tail. Honest negative, still a clean result (the frontier has a
  fundamental near-boundary tail; exact-JSR needs B&B, NP-hard) → document + package.
- Either way the **bracket [jsr_lb, γ*_8]** quantifies exactly how close CPU SOS gets to exact-JSR.

## Phase 5 — packaging (surpass-llive / funding arc; user-gated)
- The coverage frontier (inf→…→deg8→JSR limit, complementary non-nested rungs, all sound) +
  the deg6 **two-frontier** result (capability saturates at SDP, dimension-gated) = a complete,
  novel, falsifiable narrative: *"Certificate strength has two frontiers; the coverage frontier
  reaches the JSR limit on CPU, the capability frontier is dimension-gated."*
- Repro packaging (seeds/env fixed, research/ tidy) → TMLR/workshop/arXiv draft section.
- Bridge to GPU (deferred, user-gated): the dimension-monotonicity experiment (EXP-C n=2..6, cheap
  CPU) is the go/no-go signal whenever the GPU question is revisited.

## Compute / parallelism summary
- Phases 1 & 2 parallel (independent inputs) → ~15 min wall. Phase 3 → ~8 min. Phase 4 → ~20 min.
- All n=2 lifted LMIs are tiny (deg8 = 5×5); cvxpy canonicalisation is the only cost. Background
  processes + the `2>&1` direct-to-file pattern (NOT `Where-Object`, which buffered the smoke).
- Soundness authority throughout = independent eigen re-check + jsr_lb<1 cross-check (NOT solver
  status). Pre-register → run → red-team → adversarial Workflow → verdict, per stage.

## Risks / honest guards (carry the deg6 lessons)
- "Good numbers" distrust: any recovery/closure number gets the jsr_lb<1 soundness cross-check and
  a margin-robustness pass before it enters a verdict.
- Near-boundary genes (jsr_lb→0.997): expect degree-8 to leave a tail — report it, don't hide it.
- "Exact JSR" wording: always "tight bracket", never an exact value (NP-hard); no gene called
  expansive unless jsr_lb ≥ 1.
