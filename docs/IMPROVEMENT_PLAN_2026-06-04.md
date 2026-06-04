# llcore Improvement Plan (2026-06-04) — methodical roadmap

> Goal: 「llcore の改良を計画的に試みる」. Discipline (all projects): pre-registration → smallest
> falsifiable PoC → honest measurement (CLARABEL-pinned, 0-unsound checked independently) → Codex /
> adversarial pair-review → optional-extras (base = stdlib+numpy). research/ isolated, src additive-only.
> Honest disclosure: a degradation / negative result is a valid, valuable outcome — never inflate.

## Where llcore is (baseline, so we improve from evidence not assumption)

- **Verification arc**: the right contraction verifier is a common quadratic Lyapunov **SDP/LMI**, not
  SMT/Z3. On CPU (n=2) it certifies **~95%** (286/300) of contracting evolved dynamics. Hardened
  (Rump verified-PD + OR-of-solvers) and **pair-reviewed** (Codex + 6-agent; over-claims narrowed,
  footgun closed; headline numbers unchanged). src 255 + research 313 tests pass. **This is the live,
  fertile thread.**
- **③ selection arc**: CPU-structurally closed (BG9) — niching is load-bearing only when deception lives
  in high-dimensional behavior space; CPU substrates' hard coordinates are low-dim and a strong
  random-restart baseline samples them directly. Further CPU ③ work is low-ROI; the only open venue is
  GPU full-LLM (paid, gated).
- **Paper**: verification-arc paper draft in progress (Workflow). Reproducibility/figures/bib are the gap.

## The arc's own honest limits == the highest-quality improvement targets

1. Completeness measured only at **n=2** → does ~95% **generalise with dimension**? (R1)
2. The lifted SOS family is **non-monotone**; exact-JSR is NP-hard; **2 near-boundary genes stay open**;
   coverage asymptotes to JSR=1 → a **lifted-union (γ*_min) backend** and/or exact-JSR closure. (R2)
3. Substrate is a single coupled-RWKV kernel → **richer GeneCodec** (multi-kernel / learning-rule). (R3)
4. Thesis demonstrated on one architecture → **cross-architecture verifier backends**. (R4)
5. The paper needs **reproducibility hardening** to be submittable. (R-repro)

## Candidate improvements (analysed)

### R1 — Dimension generalisation of the ~95% completeness  ★ IN PROGRESS (wf w2n07eeen)
- **What/why**: re-measure the inf → 2-norm → quadratic-SDP coverage frontier at **n=3, n=4** under
  CLARABEL (the n=2 audit number is 286/300; exp_nd only measured the evolution payoff, pre-audit).
- **Feasibility**: CPU/$0, bounded; reuses `coupled_nd.py` (CLARABEL-pinned cert_inf/two/sdp). Coverage
  counts are deterministic → concurrency-safe (won't be contaminated by the paper Workflow).
- **Falsifiable gate**: does SDP stay ~95% at n=3,4 (→ generality, strengthens the paper) or **degrade /
  residual grow** (→ honest limit: SDP less complete at higher dim, motivates higher-degree rungs)?
- **0-unsound** checked independently (empirical ρ at high samples + n-dim JSR lower bound).
- **Paper relevance**: turns the headline from "n=2" into "n=2,3,4" (or honestly bounds it).

### R2 — Lifted-SOS-union (γ*_min) backend  +  exact-JSR tail (menu B)
- **R2a (bounded, recommended next)**: a `VerifierBackend` that certifies via **γ*_min = min over degrees
  {2,4,6,8}** (since the SOS family is NON-monotone, the *union* is the correct certificate, not any single
  degree). Operationalises the DEG6 finding as a clean backend.
  - *Gate*: does the union certify the deg6_only / deg8 genes single degrees miss, at **0 unsound**? Does it
    beat each single-degree backend on coverage without false admits?
- **R2b (high-risk, gated, = menu B; user not yet selected for execution)**: close the **2 open
  near-boundary genes** (jsr_lb 0.9915 / 0.9787) via proper SOS-on-variety or branch-and-bound.
  - *Gate*: does B&B close jsr_lb→1 genes within a fixed CPU budget? Honest-negative likely (NP-hard; the
    frontier may genuinely asymptote to the JSR=1 boundary). **Do NOT auto-run; keep as an offered option.**
- **Feasibility**: R2a CPU/$0 bounded; R2b CPU but unbounded/NP-hard.

### R3 — Richer GeneCodec (multi-kernel / learning-rule genes)
- **What/why**: skeleton extension point #1 — evolve a gene that selects among kernels or encodes a learning
  rule, broadening the evolvable space.
- **Falsifiable gate**: does the verifier stack still certify a healthy fraction at **0 unsound** on the
  richer substrate? (Honest risk: non-coupled-RWKV dynamics may fall outside the affine-in-t Jacobian box →
  needs a new certifier; BG9 already showed kernel-union does NOT help ③, so frame R3 as a *verifier-
  generality* test, not a ③ revival.)
- **Feasibility**: CPU/$0; medium effort (new codec + possibly new certifier path).

### R4 — Cross-architecture verifier backends (SNN / Neural ODE / GNN)
- **What/why**: `research/other_archs/` already prototyped these with per-arch contraction/stability gates
  (e.g. the SNN Z3 spike-train symbolic construction; Neural-ODE continuous-time contraction). Promote one
  to a verified×evolvable result → extends the thesis beyond coupled RWKV.
- **Falsifiable gate**: is the contraction gate **load-bearing** (rejects divergent children) **and sound**
  on that architecture? Does a stronger verifier unlock more reachable safe fitness, as at n=2?
- **Feasibility**: CPU/$0; bounded per-arch; strong **paper extension** ("verified evolution across
  architectures: SDP/Lyapunov where it reduces, arch-specific certificates where it doesn't").

### R-repro — Paper reproducibility hardening
- **What/why**: clean repro scripts + fixed-seed harness + figures (frontier bars, SCS↔CLARABEL swap, gate
  payoff) + bib first-source verification. **Directly enables paper submission** (the current gap).
- **Feasibility**: CPU/$0; bounded; highest leverage *if shipping the paper is the priority*.

### Parked (decision recorded in memory `project_llcore_productionization_candidates`)
- **Eval Cython** (accelerate the non-native eval loops) — when eval is the measured bottleneck / at
  productionisation; profile + honest benchmark first.
- **GPU ③ full-LLM (BG10)** — the only open ③ venue; **paid**, pre-registration first, portfolio-gated.
- **IP protection / obfuscation** (Nuitka/Cython/Rust + encryption + license) — final/production stage,
  crown-jewels-only, OSS framework stays open.

## Recommended sequence + decision gates

```
R1 (running)  ──►  read R1 verdict
   │ if SDP generalises (~95% at n=3,4):  R4 (cross-arch, biggest paper payoff)  ──►  R-repro  ──► submit
   │ if SDP degrades with n:              R2a (lifted-union backend recovers the grown residual?) ──► R4
   └ in all cases: R-repro is the gate to actually shipping the paper.
R2b (exact-JSR) and GPU(BG10): OFFER ONLY — user-gated, not auto-run.
R3: opportunistic (after R2a/R4) — verifier-generality probe on a richer codec.
```
**Rationale**: prioritise CPU/$0, bounded, falsifiable, paper-strengthening improvements (R1→R4/R2a→R-repro);
gate the unbounded/paid ones (R2b/GPU). Every step pre-registered, CLARABEL-pinned, 0-unsound-checked,
pair-reviewed. Re-plan after each verdict (results steer the next step — `feedback_self_made_freely_revisable`).

---

## UPDATE 2026-06-04 (post R1 + R2a) — re-plan from results

**R1 result = DEGRADES (committed):** quadratic-SDP completeness over the inf/2-norm frontier is **n=2-
specific**: 92% (n=2) → 79.5% (n=3) → **48% (n=4)**; residual 8%→52%; induced norms collapse to ~0 at n=4.
**Soundness GENERALISES** (0 unsound at every n, independently re-verified). Folded into the paper (§3.6 +
abstract scope + §7 limitations).

**R2a result = FUNDAMENTAL, not a degree artifact (committed):** lifted higher-degree SOS (deg4/6/8) recovers
only **22% (n=3) / 29% (n=4)** of the grown residual; deg8 ≈ flat over deg6. **~63% of the still-residual is
switched-expansive** (jsr_lb≥1) = beyond ANY common-Lyapunov / box-switched SOS at any degree. **Key insight:
at higher dimension the binding constraint shifts from the Lyapunov certificate CLASS to the t-box REACHABLE-
SET over-approximation** (the box hull contains expansive corners the nonlinear map never visits).

**NEW candidate — R-reach (newly INDICATED next; directly attacks the R1/R2a bottleneck):** replace the
independent-per-coordinate achievable-t **box** with a TIGHTER **sound** reachable-set over-approximation
(the t_i are coupled through the shared input/state, so the achievable (t_1..t_n) is lower-dimensional than
the box) so the certifier need not prove contraction over box-corners the map never reaches — recovering the
switched-expansive "false residual" at higher n.
- ⚠️ **CRITICAL soundness subtlety (do NOT auto-run a naive version):** every candidate gene already CONTRACTS
  (ρ<1) — so the empirical ρ / JSR oracle is **VACUOUS** as a soundness check here (it passes regardless of
  whether the certificate is valid). A tighter reachable set that is NOT a *provable* over-approximation would
  "recover" genes with a logically INVALID proof that the standard oracle cannot catch. **R-reach requires a
  THEOREM-LEVEL soundness argument** that the tighter set contains the true reachable Jacobian set — not just
  an empirical pass. Design + pre-register the soundness argument FIRST; the measurement is only meaningful
  once that holds. This is exactly the honest-disclosure trap to avoid.

**Re-sequenced plan:**
```
R1(done: degrade) → R2a(done: fundamental, box is the bottleneck) → choose next:
  • R-reach (highest-value for high-dim completeness, but soundness-subtle → design+prove FIRST, do not auto-run)
  • R4 (cross-arch breadth — lower-risk, strong paper extension)        ← safer parallel option
  → R-repro (paper submission gate)
R2b(exact-JSR) / GPU(BG10): user-gated, offer only.
```
Recommendation: present R-reach (with the soundness-proof-first requirement) and R4 as the two next options;
R-reach is higher-value but needs a sound tighter-reachability argument before any measurement is trustworthy.
