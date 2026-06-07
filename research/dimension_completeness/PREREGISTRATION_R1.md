# PRE-REGISTRATION — R1: does the quadratic-SDP contraction-completeness result GENERALISE with dimension? (n=2,3,4)

> **Registered BEFORE measuring.** Fixed seed, falsifiable gates, honest-disclosure project (post-SCS-artifact).
> Author: llcore verified-evolution arc. Date: 2026-06-04.

## 0. Background and the exact claim under test

The 2026-06-03 CLARABEL audit (`research/AUDIT_SCS_CLARABEL_2026-06-03.md`) established, **at n=2**, on a
fixed-seed pool of 300 empirically-contracting `CoupledGene`:

| level | cumulative certified | fraction of 300 |
|---|---|---|
| L0 = ∞-norm | 88 | 29.3 % |
| L1 = ∞ ∪ 2-norm | 137 | 45.7 % |
| **L2 = + quadratic SDP** | **286** | **95.3 %** |
| residual (contracting, no quadratic cert) | 14 | 4.7 % |

(The degree-4/6 SOS rungs above SDP add only +4 of those 14; the rest are a near-boundary / switched-expansive
tail. The "rich degree ladder" sub-narrative was an SCS artifact and is retracted.) Track D (1291/1363 = 95 %)
and Track B exp1 (4720/5021 = 94 %) reproduce the ~95 % SDP-completeness at n=2. **two-beats-sdp = 0** in every
audit: a common quadratic Lyapunov SDP is a true *superset* of the 2-norm (P=I) certificate.

**This pre-registration tests whether that ~95 % quadratic-SDP completeness over the empirically-contracting
pool GENERALISES as the coupled dimension grows to n=3 and n=4**, using the *already-CLARABEL-pinned* n-dim
certifiers in `coupled_nd.py` (`cert_inf`, `cert_two`, `cert_sdp`). It mirrors the n=2 coverage-frontier method
of `exp_deg6_ladder.py` but on the n-dim substrate and *without* the degree-4/6 rungs (which exist only for n=2).

## 1. Substrate and certifiers (reused, NOT reinvented)

- Genes: `coupled_nd.CoupledNDGene.make(decay∈[0,1]^n, W∈[-2,2]^{n×n}, V=I)`.
- Dynamics / Jacobian / achievable-t box: `step`, `jacobian`, `t_min_per_coord`, `_box_vertices`, `_jac_at_t`.
- Sound contraction certifiers over the 2^n-vertex t-box (all imply ρ(J)<1 over the box ⇒ contraction):
  - `cert_inf`  — closed-form sup ‖J‖_∞ < 1.
  - `cert_two`  — max over 2^n box vertices of σ_max(J) < 1 (solver-independent vertex SVD).
  - `cert_sdp`  — common quadratic Lyapunov vertex-LMI feasibility, **CLARABEL-pinned** (`_SOLVER = cp.CLARABEL`,
    fail-closed: refuses if CLARABEL absent rather than silently using SCS). Fast-path `cert_two ⇒ True`;
    independent float-eigen re-check of the solver's P.
- Pool membership oracle: `empirical_rho(g, n_samples=...)` (from-below sup of ρ(J) over the (s,x) box).

## 2. Method (deterministic; coverage counts do NOT depend on timing/concurrency)

For each n ∈ {2, 3, 4}, with a single fixed master seed:
1. Draw random `CoupledNDGene` (`decay ~ U[0,1]^n`, `W ~ U[-2,2]^{n×n}`).
2. Keep only **empirically-contracting** genes: `empirical_rho(g, n_samples=4000) < 1.0` (same screen as the
   n=2 method). Continue until the pool reaches the per-n target (≈150–300) or a scan budget / time cap is hit.
3. Compute the cumulative coverage frontier on that pool:
   - `inf`      = # certified by `cert_inf`
   - `two_cum`  = # certified by `cert_inf ∪ cert_two`
   - `sdp_cum`  = # certified by `cert_inf ∪ cert_two ∪ cert_sdp`  (= quadratic-class coverage)
   - `residual` = pool − sdp_cum  (empirically contracting but no quadratic certificate)
   - `two_only`, `sdp_only` book-kept; **`two_beats_sdp`** = # certified by 2-norm but NOT by SDP (must be 0:
     SDP is a theoretical superset, since `cert_sdp` returns True whenever `cert_two` does).
4. **Soundness (independent of the solver)** for every SDP-certified gene:
   - (a) high-sample empirical spectral radius `empirical_rho(g, n_samples=SOUND_SAMPLES)` must be `< 1`;
   - (b) an n-dim JSR product lower-bound oracle over the 2^n vertex Jacobians, `jsr_lb < 1`
     (generalises `verify_certifier_jsr_soundness.jsr_lb` to n-dim using `_box_vertices`/`_jac_at_t`).
     Product length capped per n so the product count stays bounded (n=2: len≤6 over 4 verts; n=3: len≤4 over
     8 verts; n=4: len≤3 over 16 verts). A capped JSR_lb is still a valid *one-sided lower bound*: `jsr_lb≥1`
     proves an unsound admit; `jsr_lb<1` is necessary-but-not-sufficient (same caveat as the n=2 audit).
   - An **unsound admit** = any SDP-certified gene with `rho_sound ≥ 1` OR `jsr_lb ≥ 1`. Target = **0**.
5. Confirm CLARABEL is the solver actually used (`coupled_nd._SOLVER is cp.CLARABEL`, `_CLARABEL_OK True`).
6. Record per-n: pool size, scanned, inf, two_cum, sdp_cum, sdp_pct, two_only, sdp_only, two_beats_sdp,
   residual, n_unsound, max rho_sound over certified, max jsr_lb over certified, elapsed.

## 3. Falsifiable gates (registered BEFORE the run)

- **G1 — non-trivial completeness at every n.** At each n∈{2,3,4}, `sdp_cum / pool ≥ 0.80`.
  (Below 0.80 at any n ⇒ G1 FAIL ⇒ the quadratic-SDP certifier does **not** stay broadly complete as dim grows.)
- **G2 — SDP remains a (near-)superset of 2-norm.** `two_beats_sdp == 0` at every n (small slack: ≤1 tolerated
  only as a numerical-margin artifact, must be explained). FAIL if any n shows a genuine 2-norm-only certificate
  SDP misses.
- **G3 — soundness: 0 unsound admits at every n.** `n_unsound == 0` (no SDP-certified gene with rho_sound≥1 or
  jsr_lb≥1) for n=2,3,4. ANY unsound admit ⇒ G3 FAIL ⇒ a real certifier bug at that dimension.
- **G4 — the real question: does SDP-completeness HOLD (~95 %) or DECAY with n?**
  Let `p_n = sdp_cum/pool`. Define:
  - **"SDP GENERALISES"** ⇔ `p_n ≥ 0.90` at every n∈{2,3,4} **AND** `p_2 − p_4 ≤ 0.05` (completeness does not
    drop more than 5 points from n=2 to n=4) **AND** the residual fraction stays `≤ ~10 %` at n=4.
  - **"SDP DEGRADES with dim"** ⇔ `p_n` falls below 0.90 at n=3 or n=4, **OR** `p_2 − p_4 > 0.10` (a clear
    monotone decay of ≥10 points), **OR** the residual fraction at n=4 grows past ~15 %.
  - **"MIXED / WEAKLY GENERALISES"** ⇔ anything between (e.g. p stays ≥0.85 but drifts down 5–10 points, or the
    residual roughly doubles but stays modest). Reported plainly as partial generalisation, not inflated.

## 4. What each outcome would MEAN (committed in advance — no post-hoc reinterpretation)

- **Generalises:** the central arc claim ("the right contraction verifier is a quadratic SDP/Lyapunov, and on
  CPU it is ~95 % complete") is dimension-robust — a stronger, transferable result.
- **Degrades:** a **valid and valuable negative finding** — the quadratic Lyapunov class is increasingly
  incomplete as coordinates couple (more room for non-normal / switched contractions a single quadratic P cannot
  cover), motivating higher-degree / non-monotone / path-complete Lyapunov certificates at higher n. Reported as
  the finding, NOT hidden, NOT spun.

## 5. Honest-disclosure constraints (binding)

- CLARABEL pinned for every SDP solve; reuse `coupled_nd`'s already-pinned certifiers; confirm `_SOLVER`.
- Soundness checked **independently of the solver** (empirical ρ at high sample + JSR product oracle). Report any
  false admit; target 0.
- If completeness degrades, report it plainly with exact numbers. A negative/degradation result is a valid
  outcome. Do NOT inflate, do NOT cherry-pick n.
- Fixed seed; report exact pool sizes, per-rung counts, percentages, elapsed. `research/` only; do NOT touch
  `src/`.

## 6. Known limits (stated in advance)

- Random `(decay, W)` sampling acceptance-rejects to an empirically-contracting pool; the pool is a *sample* of
  contracting dynamics, not the full contracting set. Pool size (≈150–300/n) bounds the resolution of `p_n`
  (a ±few-percent CI). Reported, not over-claimed.
- The JSR oracle is a finite-length, one-sided lower bound (length capped per n for tractability); `jsr_lb<1` is
  necessary-but-not-sufficient for soundness. The guarantee is the certifier theorem (vertex-LMI ⇒ hull
  stability by matrix convexity for the quadratic class) + independent float-eigen re-check of P; the oracles are
  falsification checks.
- n ∈ {2,3,4} only. No claim beyond n=4.
