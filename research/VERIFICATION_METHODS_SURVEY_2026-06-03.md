# SURVEY — alternative verification methods for llcore contraction certification (2026-06-03)

> Motivated by the SCS solver-artifact audit (`AUDIT_SCS_CLARABEL_2026-06-03.md`,
> `feedback_cvxpy_pin_accurate_solver`). 6-family parallel survey (web + first-principles + RAD corpus,
> 7 agents). Goal: methods that (a) HARDEN against the solver-artifact class, (b) CLOSE the residual /
> reach exact-JSR, (c) verify the NONLINEAR map directly. Ranked by value-to-llcore **on this Windows
> CPU box today** (a method needing MOSEK/Julia/WSL/torch is a project, not a next-step pilot).

**Grounded facts (verified against result JSONs):** the true residual is a **2-gene near-boundary
tail** (jsr_lb 0.9787 / 0.9915, lifted-SOS γ*_min 1.0004 / 1.0003, bracket width ~3–4e-4); the
deg6→deg8 non-monotonicity is real (`higher_degree_worse_count=1`); only 2 of 4 finite-gap genes close
via the lifted union. The residual is small and concentrated — that shapes the ranking.

## Ranking

| # | Method | Artifact-hardening | Residual / exact-JSR | Nonlinear? | Install cost NOW | Verdict |
|---|---|---|---|---|---|---|
| **1** | **Rump verified-PD + OR-of-{CLARABEL,SCS} recheck** | **Best** — removes the float-fragile 1e-7 `eigvalsh` recheck; bakes "solver-swap = decisive detector" into a standing gate | hardens soundness; makes deg8 closures *guaranteed* | no | **zero new deps** | **ADOPT** |
| **2** | **Two-sided Gripenberg + invariant-polytope exact JSR** | strong (no SDP in path = solver-independent cross-check) | **best for the 2-tail** (short SMP = polytope sweet spot; collapses the 3e-4 bracket to a point) | no | Gripenberg-ub = zero deps; polytope = 1 git-pip + wrapper | **PILOT** |
| **3** | **Path-complete max-of-quadratics LMI** (Ahmadi–Jungers) | moderate + a **monotonicity invariant** = auto artifact-detector | strictly stronger than common-P; proper SOS-on-variety restores monotonicity | no | **zero new deps** (vertex LMIs in cvxpy) | **PILOT (Tier-1)** |
| 4 | NN-verif: single-slope IQC / LipSDP-in-cvxpy | same solver (no new hardening) | collapses 2ⁿ vertices → 1 multiplier LMI | partial | IQC-LMI zero deps; auto_LiRPA = torch | pilot IQC-LMI; defer LiRPA |
| 5 | SMT / dReal / α,β-CROWN | independent 3rd leg | closes *linearization-induced* residual (true map tighter than t-box) | **yes (only family)** | dReal = WSL; CROWN = torch | **defer** (right family, wrong engine for Windows-now) |
| 6 | Contraction metrics P(t) + scenario | sound CEGIS route | parameter-dependent P(t) = cheap win | yes | P(t)-LMI zero deps; FOSSIL = WSL | Tier-1 P(t) only; **scenario = reject** |

## Top pilots (concrete, falsifiable first experiments)

**Pilot A — Rump verified-PD + OR-recheck gate (ADOPT, do first; zero deps, hours).**
Replace `np.linalg.eigvalsh` recheck on P **and** each P − Jᵥᵀ P Jᵥ (and the lifted deg-d Gram matrices)
with **Rump verified positive-definiteness** (float Cholesky of M − α·I + a rounding-error bound proves
λ_min ≥ a guaranteed positive value; ~1× Cholesky, pure numpy). Certify on **OR-of-{CLARABEL,SCS}**
(any solver whose P passes Rump) — OR, **not a vote** (a vote would preserve the SCS false negative).
*First experiment:* re-run the `sdp_only` thin-shell (~1e-7 margin, Codex #9) through float-eigvalsh vs
Rump; pre-register that Rump-certified ⊇ float-certified with no float-only false positives, and log how
many genes CLARABEL recovers that SCS calls infeasible (= the standing artifact-class measure).

**Pilot B — exact JSR on the 2-tail genes (PILOT, do second; the one most likely to CLOSE the tail).**
Step 1 (zero deps): extend `jsr_bracket.py`'s lower-bound enumerator to a branch-and-bound with a
pruning **upper** bound → a converging solver-independent [lb, ub]. Step 2: invariant-polytope algorithm
(self-port ~200 lines or `D-Tarnu/jsr`, wrapped with an independent eigen/LP recheck, fail-closed).
*First experiment:* on jsr_lb 0.9787 & 0.9915 (γ*_min ~1.0003-4), pre-register the polytope **closes**
(SMP length 1-2) returning exact JSR<1 — certifying contraction lifted SOS provably cannot. If exact
JSR ≥ 1, that is *also* a win (proves the rejection correct, upgrading "inconclusive" → authoritative).

**Pilot C — path-complete max-of-quadratics LMI backend (PILOT, Tier-1; zero deps, a day).**
De Bruijn order-1 max-of-quadratics over the 2ⁿ vertices in the existing cvxpy/CLARABEL pattern,
strictly stronger than common-P. *First experiment:* on the 2 finite-gap genes the lifted union missed,
pre-register path-complete certifies ≥1. Use the **monotonicity invariant** (a higher-order graph
reporting a *worse* bound ⇒ provably buggy) as an automatic artifact detector the non-monotone lifted
family structurally cannot give. Keep the CLARABEL pin + Rump recheck on it.

**If only two: A and B.** A hardens the whole stack for free; B is the one most likely to close the
2-gene tail and reach exact JSR.

## Explicitly flagged — over-engineering / unsound (do NOT adopt as certifiers)

- **Probabilistic scenario / conformal contraction — REJECT as a certifier.** Probabilistic ≠ sound;
  a too-small N silently inflates the true violation probability = exactly the fabricated-finding class
  llcore just learned to fear. OK only as a **non-certifying triage prescreen** (rank which residual
  genes deserve an expensive proof); a p-value must NEVER upgrade a gene past "plausibly contracting".
- **VSDP/INTLAB + Minlog proof-assistant route — SKIP** (MATLAB-only / overkill; Rump gives ~95% of the
  rigor with zero deps).
- **MOSEK/SDPA "wide solver jury" — NOT AVAILABLE** (license/build on Windows). Honest jury =
  SCS(ADMM)+CLARABEL(IPM), two families, thin — don't claim a 4-solver consensus you can't run.
- **dReal — right family (δ-complete with tanh), wrong engine here** (Windows-hostile, 2021-stale,
  WSL/IBEX). Defer; the modern nonlinear leg is α,β-CROWN (pip+torch) = a deliberate later pilot.
- **Z3 for the nonlinear map — SKIP** (no native tanh). Keep z3 only as the closed-form tie-checker it
  already is (Track C, 0/3270 disagreement = decorative-but-confirming).
- **Caution (not rejection):** IQC/LipSDP-in-cvxpy and P(t)-LMI are useful (fewer LMIs / richer class)
  but run on the **same conic solver** — they harden the *formulation*, not the *solver*, so they do
  NOT cure the SCS class and must keep the pin + Rump recheck + JSR oracle.

## Fundability

This converts llcore's verification story from *"an SDP solver said feasible"* (whose failure mode the
audit just lived — exactly a reviewer's criticism) into a **layered, solver-independent, partly
machine-checked certificate chain**: a common-P LMI workhorse whose every verdict is upgraded from
float-checked to **Rump-guaranteed PD**; an **OR-of-solvers gate** baking the "solver-swap is the
decisive detector" lesson into a standing invariant; and an **exact-JSR top rung** (invariant polytope)
that closes the residual with no SDP in the path. The honest self-correction + this hardening chain is
a credibility asset stronger than any single dramatic result.

Source: workflow `verification-methods-survey` (7 agents, ~740k tok). 6 family assessments in the run
transcript. Next session: implement Pilot A (zero deps, highest ROI), then Pilot B (close the 2-tail).
