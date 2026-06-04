# R_REACH DESIGN — a tighter SOUND reachable-set over-approximation to replace the conservative t-box

> Design-only. No recovery measurement is run here (per workflow). Soundness is argued at the
> THEOREM level because every candidate gene already contracts (ρ<1), so an empirical ρ / JSR
> oracle is VACUOUS as a soundness check for a NEW relaxation — it passes regardless of whether
> the certificate is a valid proof. The deliverable is a *proof* that the proposed tighter set is a
> sound OVER-approximation, plus a falsifiable pre-registration for a FUTURE measurement.
>
> Sources read: `verified_evolution_sdp_gate/coupled_nd.py`, `coupled_z3_contraction/coupled_map.py`,
> `spectral_lyapunov_contraction/lyapunov_sdp_certifier.py`, `dimension_completeness/R2A_VERDICT.md`,
> `highdeg_residual_attribution.json`, `R1_VERDICT.md`. src/ untouched; this is research/ design.

---

## 0. Notation (locked, matches the code exactly)

The coupled-RWKV map (`coupled_nd.step`, `coupled_map.step`), with `V = I`:

```
s' = decay ⊙ s + (1 − decay) ⊙ tanh(W s + V x),   s ∈ [−1,1]^n, x ∈ [−x_max, x_max]^n,
                                                   decay ∈ [0,1]^n, W ∈ [−2,2]^{n×n}, V = I_n.
```

Exact state Jacobian (`coupled_nd.jacobian`):

```
J(t) = D + diag((1−decay) ⊙ t) · W,   D := diag(decay),
t_i = sech²(pre_i) = 1 − tanh²(pre_i) ∈ (0, 1],   pre_i = (W s + V x)_i = (W s)_i + x_i.
```

`J` is **affine in t** for fixed gene. Define the affine vertex map (`coupled_nd._jac_at_t`):
`J(t) = D + diag((1−decay)⊙t) W`. The certifier proves contraction over a set 𝒯 ⊆ (0,1]^n of
admissible t-tuples by testing the *extreme points* of `conv{ J(t) : t ∈ 𝒯 }` and (for SDP) finding a
common P with `P − J^T P J ≻ 0` at those extremes; matrix convexity of `J↦J^T P J` then extends the
decrease to the whole hull (`lyapunov_sdp_certifier` docstring; the "polytopic-LMI argument").

The **true reachable Jacobian set** is

```
𝒥_true = { J(t(s,x)) : s ∈ [−1,1]^n, x ∈ [−x_max,x_max]^n },
   where t_i(s,x) = sech²((W s)_i + x_i).                                    (R0)
```

Soundness requirement for ANY relaxation 𝒯: contraction certified over `{J(t): t∈𝒯}` is a valid proof
of contraction of the map **iff** `𝒥_true ⊆ { J(t) : t ∈ 𝒯 }`, equivalently (since `J(·)` is an
injective affine map of t for any gene whose `(1−decay_i)` rows are non-degenerate, and a sound
relaxation only needs to cover the *image*) **iff** the projection of `t(s,x)` onto the relevant
coordinates is contained in 𝒯. We work in t-space throughout.

---

## 1. The current t-box over-approximation and WHY it is loose

### 1.1 Current set (exactly as implemented)

`coupled_nd.t_min_per_coord` / `coupled_map.t_min_per_coord`:

```
M_i = Σ_j |W_ij| + x_max · Σ_j |V_ij| = Σ_j |W_ij| + x_max   (V = I)
t_min_i = sech²(M_i) = 1 − tanh²(M_i).
𝒯_box = ∏_{i=1}^n [t_min_i, 1].                                              (R1)
```

This is sound per coordinate: `|pre_i| = |(Ws)_i + x_i| ≤ Σ_j|W_ij|·|s_j| + |x_i| ≤ Σ_j|W_ij| + x_max
= M_i`, and `sech²` is even and strictly decreasing in `|·|`, so `t_i = sech²(pre_i) ∈ [sech²(M_i), 1] =
[t_min_i, 1]`. ✔ Each coordinate's marginal range is exactly captured (the bound `t_min_i` is even
*attained* when `pre_i = ±M_i`, i.e. at a sign-aligned corner of `(s,x)`).

The certifier then proves contraction at all `2^n` vertices of `𝒯_box` (`_box_vertices`).

### 1.2 Why it is loose — the t-coupling

`𝒯_box` is the **Cartesian product** of the n marginal ranges. It therefore contains every tuple
`(t_1,…,t_n)` with each `t_i` independently anywhere in its own range — in particular the
**all-minimum corner** `t = (t_min_1,…,t_min_n)` and every mixed min/max corner. But the `t_i` are
**not** independent: they are all driven by the *same* `(s,x)` through the shared linear pre-activations
`pre_i = (Ws)_i + x_i`. Reaching `t_i = t_min_i` requires `|pre_i| = M_i`, i.e. `(Ws)_i + x_i = ±M_i`,
which by the triangle-inequality equality condition **forces** specific signs and near-saturation
(`|s_j| ≈ 1, |x_i| ≈ x_max`, signs aligned to `sign(W_ij)` and `sign(x_i)=sign((Ws)_i)`). The same `s`
then determines all other `pre_k = (Ws)_k + x_k` — it is generally NOT free to also saturate the k-th
coordinate's worst sign-pattern. So the achievable `(t_1,…,t_n)` lives on a **coupled,
lower-effective-dimensional set** 𝒯_true ⊊ 𝒯_box, and the box's expansive corners (where several `t_i`
simultaneously hit their minima with the conflicting sign patterns each minimum demands) are
**generically unreachable**.

This is exactly the R2a diagnosis (`R2A_VERDICT.md`, `highdeg_residual_attribution.json`): ≈63 % of the
n=3,4 residual is **switched-expansive over 𝒯_box** — a product of the box-VERTEX Jacobians has spectral
radius ≥ 1 (`jsr_lb ≥ 1`, min observed 1.0009) — yet the genes are empirically contracting because the
nonlinear map *never visits the box's expansive corner*. No common Lyapunov of any degree can fix a
genuinely-expansive switched set; the only lever is to **shrink the t-set toward 𝒯_true**. The box's
slack is purely the independence assumption on the `t_i`.

### 1.3 Quantifying the slack (the lever)

Two structural facts make the box maximally loose precisely where it hurts:

- **Marginal tightness, joint looseness.** Each interval `[t_min_i,1]` is individually tight (attained),
  so 1-D refinements buy nothing. ALL the recoverable slack is in the *joint* (off-diagonal-of-the-box)
  region. This is why R2a's higher SOS degree (a richer certificate over the SAME box) recovers little —
  the obstruction is the *set*, not the *certificate class*.
- **Sign conflict.** `t_min_i` needs `s` aligned to row `i` of `W` (`s_j = ±1` with sign `sign(W_ij)·σ`
  for a global sign σ) AND `x_i = ±x_max`. For two rows i,k to *simultaneously* hit their minima the
  same `s ∈ {±1}^n` must align to BOTH `W_i·` and `W_k·`. Unless rows i,k are sign-identical, no single
  `s` does this — so the box corner `(t_min_i, t_min_k, …)` over-states how small the pair can jointly be.

---

## 2. Candidate tighter over-approximations

For each: the t-set representation, and how the certifier's vertex/LMI set changes.

### Candidate A — shared-magnitude scalar coupling (1-parameter envelope per coordinate)  ★ recommended core

Introduce a single shared "excitation magnitude" scalar and bound each `pre_i` by a **coupled** (not
independent) budget. The cleanest sound version uses the shared state-norm:

Let `r := ‖s‖_∞ ∈ [0,1]` (the SAME r drives every coordinate, because there is one state). Then for every
i simultaneously:
```
|pre_i| = |(Ws)_i + x_i| ≤ (Σ_j |W_ij|) r + x_max =: a_i r + x_max,   a_i := Σ_j |W_ij|,
⇒ t_i ≥ sech²(a_i r + x_max) =: τ_i(r),     a single monotone curve in the shared r ∈ [0,1].   (A1)
```
The reachable t-set is over-approximated by the **swept lower-envelope region**
```
𝒯_A = { t ∈ (0,1]^n : ∃ r ∈ [0,1], t_i ≥ τ_i(r) ∀i }
     = ⋃_{r∈[0,1]} ∏_i [τ_i(r), 1].                                                            (A2)
```
Key point: it is the SAME r in every coordinate. The box `𝒯_box` corresponds to letting each coordinate
pick its *own* `r_i = 1` independently; `𝒯_A` forbids that — a coordinate that is near its minimum
(`r ≈ 1`) forces *all* coordinates toward their `r=1` floors *together*, but a coordinate cannot be near
its `r=0` value (`τ_i(0)=sech²(x_max)`, close to 1) while another is at its `r=1` floor. `𝒯_A ⊆ 𝒯_box`
strictly whenever the `a_i` differ or `x_max>0`, because the all-min corner now requires one global `r=1`,
which `𝒯_A` does include — so A alone does NOT remove the all-min corner. **A's value is removing the
MIXED corners** (one `t_i` at min while another is near 1), which is where the sign-conflict slack lives.

> Honest caveat on A: using `r=‖s‖_∞` as the single coupler is sound (see §3) but conservative — it
> keeps the worst `x` per coordinate. A by itself shrinks the *mixed* corners but retains the all-min
> corner. To attack the all-min corner (the switched-expansive driver) we need the sign-coupling of
> Candidate C. A is the scaffolding; C is the cut that bites.

**Certifier change for A:** discretize `r` on a grid `0=r_0<…<r_K=1` and, on each subinterval
`[r_k, r_{k+1}]`, the t-region is the box `∏_i [τ_i(r_{k+1}), 1]` (use the right endpoint = smallest τ =
largest box, sound). Certify a COMMON P over the union of `K` boxes (all `K·2^n` vertices in one LMI), or
a path-complete / per-segment P (multiple Lyapunov, monotone in r). Strictly fewer infeasibilities than
one big box because each segment's box is contained in `𝒯_box`. As `K→∞` it tends to the exact envelope
`𝒯_A`. (Implementation: same `cert_sdp` machinery, vertex list replaced by the union of segment-box
vertices.)

### Candidate B — ellipsoidal / zonotopic bound on the pre-activation vector  (lift to pre-space)

Bound the whole pre-vector jointly instead of coordinate-wise. `pre = W s + x` with `s ∈ [−1,1]^n`,
`x ∈ [−x_max,x_max]^n`. The reachable `pre`-set is the **zonotope** `𝒵 = W·[−1,1]^n ⊕ [−x_max,x_max]^n`
(Minkowski sum of two boxes; an exact polytope with ≤ known generators). Over-approximate `𝒵` by an
ellipsoid `ℰ = {pre : pre^T Q^{-1} pre ≤ 1}` (e.g. the Löwner / minimum-volume enclosing ellipsoid of the
zonotope, or the cheap diagonal `Q = diag(M_i²)`), then push through `t_i = sech²(pre_i)`:
```
𝒯_B = { t : t_i = sech²(pre_i), pre ∈ 𝒵 }  ⊆  hull/interval-image of sech²(ℰ).               (B1)
```
This captures inter-coordinate correlation (the off-diagonal of `W W^T` ties the `pre_i` together).
`𝒯_B ⊆ 𝒯_box` because `𝒵` is contained in the per-coordinate box `∏[−M_i, M_i]` (which is what (R1)
uses), and ellipsoidal containment only tightens it. **Certifier change:** the t-set is no longer a box, so
the vertex test is replaced by either (i) an interval over-approximation of `𝒯_B` (back to a tighter box
— wasteful) or (ii) sampling extreme `pre` directions of `𝒵` (its `2·#generators` facet-normals) → the
corresponding `t` points → their convex hull → LMI at hull vertices. More faithful than A but more
machinery (zonotope generators, hull). B is the "right" geometric object; A is its cheap 1-parameter
slice and C its sign-aware refinement.

### Candidate C — sign-coupling cut (the corner-killer)  ★ this is what bites the switched-expansive residual

The switched-expansive obstruction is the **all-min / mixed-min CORNERS** of `𝒯_box`. C adds a sound
constraint that *excludes the unreachable corners directly*, leaving the rest of the box. Reaching the
joint minimum of coordinates in a set `S ⊆ {1,…,n}` (all `t_i = t_min_i, i∈S`) requires, by the triangle
equality condition, a single `s ∈ [−1,1]^n` with `(Ws)_i = (Σ_j|W_ij|)·σ_i` for each `i∈S` and aligned
`x_i`. Equality `|(Ws)_i| = Σ_j|W_ij|` forces `s_j = σ_i · sign(W_ij)` for every j with `W_ij ≠ 0`. If two
indices `i,k ∈ S` demand conflicting signs on some shared `s_j` (i.e. `σ_i sign(W_ij) ≠ σ_k sign(W_kj)`
for some j with `W_ij W_kj ≠ 0`), then **the joint minimum is unreachable** and that corner can be
EXCISED soundly. More generally, define for any candidate joint sign pattern the *best achievable*
simultaneous floor by ONE optimization (below) and use it instead of the independent floors. Representation:
```
𝒯_C = { t : ∃ s∈[−1,1]^n, x∈[−x_max,x_max]^n with t_i ≥ sech²((Ws)_i + x_i) ∀i }
     = exact reachable t-LOWER-set (this IS 𝒯_true's down-closure).                            (C1)
```
**Certifier change:** replace the static box vertices by the vertices of the *reachable* lower-set. The
practical, sound, finite version is Candidate D.

### Candidate D — sound sampled-corner hull + rigorous interval margin (the implementable one)  ★ best_candidate

Combine C's exactness target with a *provably sound* finite construction:

1. Enumerate the `2^n` sign patterns `σ ∈ {±1}^n` of the state corners `s = σ` (the abs-sum extremes of
   `pre` live at `s ∈ {±1}^n` and `x ∈ {±x_max}^n`; `pre_i` is affine in `(s,x)`, so its extremes over the
   box are at these corners — exact, no sampling error).
2. For each `(s,x)` corner, compute `pre_i`, hence the **exactly reachable** `t`-corner
   `t^{(σ)}_i = sech²((Wσ)_i + x_i^{best})`. These `4^? ` corners (state-corner × input-corner) are points
   that the map **provably attains** (so they are *in* 𝒥_true), giving an inner description; their convex
   hull `H = conv{ J(t^{(c)}) : corners c }` is an INNER approximation of `conv 𝒥_true`.
3. **Soundness margin (the crux):** an inner hull is NOT a sound over-approximation by itself. Inflate it
   to a sound OUTER set by a *rigorous* per-coordinate interval margin derived from the smoothness of
   `t_i = sech²(pre_i)`: between adjacent reachable corners the true `t_i` cannot stray outside
   `[min, max]` of `sech²` over the *interval of `pre_i` actually swept*, and we bound that interval
   EXACTLY (`pre_i` is affine ⇒ its range over the box is `[−M_i, M_i]`, but restricted to the reachable
   joint pattern it is the segment between corner values). The sound outer set is
   `𝒯_D = ⋂_i { t : t_i ∈ [ min over reachable patterns of sech²(pre_i), 1] }` **intersected with** the
   pairwise sign-feasibility constraints from C (exclude corners with conflicting required signs). Because
   every excised corner is *proved* unreachable (sign conflict ⇒ `pre_i, pre_k` cannot both saturate), and
   every retained region *contains* the true reachable t (corners are attained, interior by continuity),
   `𝒥_true ⊆ {J(t):t∈𝒯_D} ⊆ {J(t):t∈𝒯_box}`. **Certifier change:** the vertex list fed to `cert_two` /
   `cert_sdp` becomes the (smaller) set of vertices of `𝒯_D` instead of `_box_vertices`. Drop-in.

> Why D over C: C is the exact reachable lower-set but its description is an optimization (NP-hard in
> general). D is C's *finite, sound, conservative* realization — it can only OVER-approximate 𝒯_true
> (never miss a reachable point), so it is a valid proof basis, while still excising the proved-unreachable
> sign-conflict corners that drive the switched-expansive residual.

---

## 3. ★ THEOREM-LEVEL SOUNDNESS ARGUMENT (for the recommended candidate)

We prove soundness for the **sign-coupling cut** mechanism (Candidate C/D), since that is the mechanism
that actually removes the switched-expansive corners. Candidate A's envelope soundness is a corollary
(it is a coarser sound relaxation that *contains* 𝒯_D, hence also contains 𝒥_true).

### 3.1 Assumptions (every one explicit)

- (H1) State bound: `s ∈ [−1,1]^n`. This is forced by the map itself: `s'_i = decay_i s_i +
  (1−decay_i) tanh(·) ∈ [−1,1]` is a convex combination of `s_i ∈ [−1,1]` and `tanh(·) ∈ (−1,1)`, so
  `[−1,1]^n` is forward-invariant (`coupled_map.iterate_state_growth` docstring confirms `|s|` never
  exceeds 1). The certifier already uses `|s_j| ≤ 1`. **Soundness depends on this bound holding for the
  trajectory; it does because of invariance, INDEPENDENT of any contraction claim.** ✔ (not circular).
- (H2) Input bound: `x ∈ [−x_max, x_max]^n`, `x_max = max_input_abs` (default 1.0), `V = I`. This is the
  certifier's standing assumption (admissible inputs). Soundness is *conditional* on it; if a deployment
  feeds `|x| > x_max` the proof does not cover it (same caveat as the current box — no regression).
- (H3) Exact Jacobian form: `J(t) = D + diag((1−decay)⊙t) W`, affine in t, `t_i = sech²(pre_i)`,
  `pre_i = (Ws)_i + x_i`. Verified from the code (`coupled_nd.jacobian`, `coupled_map.jacobian`).
- (H4) `decay_i ∈ [0,1]` so `(1−decay_i) ≥ 0` (used only to keep `J` affine with the stated structure;
  not needed for the set-containment argument itself).

### 3.2 The key lemmas

**Lemma 1 (exact marginal range — already used).** For each i, `pre_i` is affine in `(s,x)` over the box
`[−1,1]^n × [−x_max,x_max]^n`, so its range is `[−M_i, M_i]` with `M_i = Σ_j|W_ij| + x_max` (the abs-sum
attained at a sign-aligned corner). Hence `t_i = sech²(pre_i) ∈ [sech²(M_i), 1] = [t_min_i, 1]`. ∎
(This is (R1); it gives the marginal soundness the box already has.)

**Lemma 2 (sign-conflict exclusion).** Fix indices `i ≠ k` and a target where BOTH saturate:
`pre_i = ε_i M_i` and `pre_k = ε_k M_k` for chosen signs `ε_i, ε_k ∈ {±1}` (the only way `t_i=t_min_i` AND
`t_k=t_min_k`). Saturation `|pre_i| = M_i = Σ_j|W_ij| + x_max` with `|s_j|≤1, |x_i|≤x_max` is, by the
triangle inequality, attained **iff** `x_i = ε_i x_max` and `s_j = ε_i·sign(W_ij)` for every j with
`W_ij ≠ 0` (the equality case of `|Σ_j W_ij s_j + x_i| ≤ Σ_j|W_ij| + x_max`). Likewise saturating k
requires `s_j = ε_k·sign(W_kj)` for every j with `W_kj ≠ 0`. If there exists j with `W_ij W_kj ≠ 0` and
`ε_i sign(W_ij) ≠ ε_k sign(W_kj)`, the two requirements on `s_j` are contradictory ⇒ **no `s` saturates
both** ⇒ the joint corner `(t_i,t_k)=(t_min_i,t_min_k)` (with signs ε_i,ε_k) is **unreachable**. ∎

**Lemma 3 (a sound replacement floor for an excluded corner).** When Lemma 2 excludes a sign pattern, the
*best simultaneously achievable* floor for the involved coordinates is obtained by ONE bounded linear
program per pattern:
```
for each sign assignment σ ∈ {±1}^n on s and δ ∈ {±1}^n on x:
    pre_i(σ,δ) = Σ_j W_ij σ_j + x_max δ_i      (s=σ ∈ {±1}^n, x = x_max·δ are the box corners)
    t̃_i = sech²(pre_i(σ,δ))                    (reachable — this (s,x) is admissible)
```
The set of corners `{ t̃(σ,δ) }` are POINTS THE MAP ATTAINS (so `J(t̃) ∈ 𝒥_true`). For coordinate i, the
sound joint lower floor *consistent with a single global `s`* is `t_i ≥ min_{σ,δ} sech²(pre_i(σ,δ))`
EVALUATED ONLY over the patterns where the *other* targeted coordinates are also at/near their floors —
but because we keep the FULL corner set (all `2^n × 2^n` of them, or the relevant subset) and take their
hull, we never need to compute a closed-form floor: the hull of attained corners is automatically a
subset of `conv 𝒥_true`. ∎

**Theorem (soundness of 𝒯_D).** Construct `𝒯_D` as in Candidate D:
1. Compute the corner set `C = { t̃(σ,δ) : σ∈{±1}^n, δ∈{±1}^n }` of attained reachable t-points (Lemma 3).
2. Let `𝒯_D` be any set satisfying `conv(C) ⊆ {J(t) : t∈𝒯_D}` AND `𝒯_D ⊇ 𝒯_true`.
   The *constructive* choice: `𝒯_D := { t : t_i ∈ [ ℓ_i, 1 ] ∀i } ∩ Σ`, where
   `ℓ_i := min_{σ,δ} sech²(pre_i(σ,δ)) = t_min_i` (the marginal floor is still attained, Lemma 1) and
   `Σ` is the polytope obtained by removing the half-space corners proved unreachable by Lemma 2.

Then **𝒥_true ⊆ {J(t) : t ∈ 𝒯_D} ⊆ {J(t) : t ∈ 𝒯_box}.**

*Proof.* (Upper containment) Every t excised from `𝒯_box` to form `𝒯_D` is a corner region proved
unreachable by Lemma 2, so removing it cannot drop any reachable point. Hence `𝒯_D ⊆ 𝒯_box` and the
right inclusion holds, with the certifier's hull only shrinking. (Lower containment / the soundness that
matters) Take any reachable Jacobian `J* = J(t*) ∈ 𝒥_true`, i.e. `t*_i = sech²((Ws*)_i + x*_i)` for some
admissible `(s*,x*)`. For each i, `pre*_i = (Ws*)_i + x*_i ∈ [−M_i, M_i]` (Lemma 1) ⇒ `t*_i ∈ [ℓ_i,1]`.
Moreover `(s*,x*)` is a *single* admissible point, so `t*` cannot lie in any sign-conflict corner excluded
by Lemma 2 (such corners require an `s` that does not exist). Therefore `t* ∈ 𝒯_D`. Since `t*∈𝒥_true` was
arbitrary, `𝒥_true ⊆ {J(t):t∈𝒯_D}`. ∎

**Corollary (valid proof of contraction).** If the certifier finds a common `P ≻ 0` with
`P − J(v)^T P J(v) ≻ 0` at every vertex `v` of `𝒯_D` (or proves `cert_two`/`cert_inf` < 1 over `𝒯_D`),
then by matrix convexity of `J ↦ J^T P J` and convexity of `𝒯_D`, the decrease holds for all
`J ∈ conv{J(t):t∈𝒯_D} ⊇ 𝒥_true`. Hence `V(z)=z^T P z` strictly decreases along EVERY reachable Jacobian
action: for all admissible `(s,x)`, `J(s,x)^T P J(s,x) ≺ P`, giving a uniform contraction rate `γ<1` in
the P-norm and `ρ(J(s,x))<1` everywhere on the reachable set. This is a VALID proof of contraction of the
nonlinear map (the mean-value / contraction-mapping argument applies because the bound holds at *every*
point of the reachable Jacobian set, not at sampled points). ∎

### 3.3 What must be TRUE for soundness to hold (and steps I flag as uncertain)

- **MUST hold (and does):** (i) state invariance `s∈[−1,1]^n` (H1, proved by convex-combination
  structure — not by contraction, so non-circular); (ii) input admissibility `|x|≤x_max` (H2, the
  certifier's domain); (iii) the affine-in-t structure (H3); (iv) the corners used in Lemma 3 are
  *actually attained* — they are, because `s=σ∈{±1}^n` and `x=x_max δ` are admissible points.
- **The corner-extreme claim** (`pre_i`'s box-extremes are at `(s,x) ∈ {±1}^n × {±x_max}^n`): TRUE because
  `pre_i` is affine and a linear function over a box attains its extremes at vertices. ✔
- **⚠ FLAG 1 (the one step to scrutinize): the convex-hull-of-corners covers the reachable t-set.**
  Lemma 1 guarantees the *marginal* (per-coordinate) range is covered, so the constructive `𝒯_D` (the box
  minus proved-unreachable corners) is a sound OUTER set — that part is airtight. BUT if a future
  implementation uses the *inner* `conv(C)` directly as the certification set (instead of the box-minus-Σ
  outer set), it must ADD the smoothness margin (Candidate D step 3): `t_i = sech²(pre_i)` is nonlinear,
  so the image of a `pre`-segment between two corners can BULGE outside the chord. `sech²` is not convex on
  all of ℝ (it has inflection points at `pre = ±arctanh(1/√3) ≈ ±0.658`), so the chord between two corner
  t-values does NOT bound the interior t-values in general. **Mitigation (keeps soundness):** do not use
  the raw inner hull; use the *outer* set `𝒯_D = box ∩ Σ` (per-coordinate intervals `[t_min_i,1]` are
  exact bounds by Lemma 1, valid regardless of `sech²` curvature), and only the *corner-exclusion* Σ is
  the tightening. Σ removes regions of t-SPACE that no `s` reaches; it never relies on `sech²` curvature.
  So the recommended construction is immune to FLAG 1. I flag it because a careless "hull of sampled
  corners" implementation would be UNSOUND.
- **⚠ FLAG 2 (tightening strength vs the residual):** Σ excises corners only when a *strict sign conflict*
  exists (Lemma 2). For genes whose `W` has near-sign-aligned rows, few corners are excluded and the
  tightening is weak — `𝒯_D ≈ 𝒯_box` for those. So D is *guaranteed sound* but its *recovery power* on
  the specific switched-expansive residual is an empirical question (that is what the pre-registration in
  §4 tests). The soundness theorem does NOT depend on FLAG 2; only the *usefulness* does.
- **⚠ FLAG 3 (Candidate A's single-coupler conservatism):** A uses `r=‖s‖_∞` shared across coordinates but
  keeps the worst `x` per coordinate, so A removes mixed corners but NOT the all-min corner; soundness of A
  is clear (it is a superset of 𝒯_D), but A alone may not recover the switched-expansive genes. Use A only
  as the cheap scaffold or for the per-segment multi-Lyapunov variant; rely on D for the cut.

---

## 4. Falsifiable PRE-REGISTRATION for a FUTURE measurement (NOT run here)

> To be executed only AFTER the soundness theorem (§3) is accepted. Fixed seed, research/ only,
> src/ untouched, CLARABEL pinned (abort otherwise — inherit the project's hard guardrail).

### 4.1 Test set
The EXACT R1 switched-expansive residual genes already on disk: from
`dim_completeness_residual_genes.json`, the n=3 (41) and n=4 (104) residual, partitioned by
`highdeg_residual_attribution.json` into **switched-expansive** (`jsr_lb ≥ 1`: 20 at n=3, 47 at n=4) and
finite-gap. The switched-expansive subset is the PRIMARY target (it is the part R2a proved is beyond any
SOS degree over `𝒯_box`).

### 4.2 Procedure
For each target gene at n∈{3,4}:
1. Build the tighter set `𝒯_D` (Candidate D): enumerate `(s,x)` corners, compute reachable t-corners,
   apply Lemma 2 sign-conflict exclusion to get the reduced vertex list.
2. Run the UNCHANGED certifier machinery (`cert_two`, then `cert_sdp` with CLARABEL + independent
   eigen re-check) over `𝒯_D`'s vertices instead of `_box_vertices`.
3. Record `recovered_D` = genes now certified that `𝒯_box` could not certify, per n, per
   switched-expansive vs finite-gap.

### 4.3 HARD soundness gate (the crux — soundness rests on the PROOF, not the vacuous oracle)
- **SG-proof (primary, blocking):** the construction of `𝒯_D` MUST instantiate the §3 theorem exactly:
  (a) every excised corner is justified by a Lemma-2 sign conflict (assert per excised corner, machine-
  checked: there exists `j` with `W_ij W_kj ≠ 0` and the required signs conflict); (b) the per-coordinate
  intervals are the exact `[t_min_i,1]` (Lemma 1); (c) NO inner-hull / `sech²`-chord shortcut is used
  (FLAG 1). If any check fails the run is INVALID — it does not matter what recovery number comes out.
- **SG-containment (machine-checkable surrogate for the proof, still NOT the vacuous ρ oracle):** sample
  `(s,x)` densely AND at all `4^? ` corners; for every sampled reachable `t(s,x)`, assert
  `t(s,x) ∈ 𝒯_D` (point-in-polytope). This can FALSIFY an implementation bug (a reachable point outside
  `𝒯_D` ⇒ unsound construction). Crucially this is a falsification of the *set construction*, not a
  contraction check — it can fail and thereby catch unsoundness, whereas the ρ/JSR oracle cannot
  (every gene contracts, so ρ<1 is vacuous as the spec warns). **The recovery claim is admitted ONLY if
  SG-proof passes; SG-containment is a bug-catch, not the soundness basis.**
- **SG-no-vacuous:** explicitly record that ρ<1 / `jsr_lb` over `𝒯_box` is NOT used as the soundness
  criterion (it is the very thing being relaxed). The JSR over the NEW set `𝒯_D` MAY be reported as a
  *consistency* number, but soundness = SG-proof.

### 4.4 Falsifiable verdict thresholds (fixed now)
- **RR-recover:** fraction of the switched-expansive subset newly certified over `𝒯_D`, per n.
  - **"reachability-is-the-binding-constraint" (confirms R2a diagnosis):** `𝒯_D` recovers a MAJORITY
    (>50 %) of the *switched-expansive* subset at BOTH n=3 and n=4, with SG-proof PASS and 0 SG-containment
    violations. This would prove the binding constraint was the t-box reachable-set over-approximation, not
    the Lyapunov class — the explicit R2a hypothesis.
  - **"partial":** 25–50 % at n=4.
  - **"reachability-not-enough":** `<25 %` at n=4 even with the tighter sound set ⇒ the residual is
    expansive even over the *true* reachable set (a deeper obstruction than the box looseness) — report
    plainly; do NOT inflate.
- **RR-sound (blocking):** SG-proof PASS for 100 % of recovered genes; 0 SG-containment violations across
  ≥10^5 samples + all corners per gene. Any violation ⇒ the run is unsound and discarded (honest-
  disclosure: report the violating gene, do not silently drop it).
- **RR-honesty:** if recovery is small, the registered conclusion is "the switched-expansive residual is
  expansive even on the true reachable set" — a valid, valuable negative result, NOT a failure to hide.

### 4.5 Tractability note
Corner enumeration is `2^n` (s) × `2^n` (x) = `4^n` corners (n=4 ⇒ 256) — trivial. Lemma-2 exclusion is
`O(n² · n)` sign checks per gene. The reduced vertex set is ≤ `2^n`, so the LMI is no larger than the
current one. The whole sweep is comparable to R1's 507 s budget.

---

## 5. One-line summary
Replace the independent per-coordinate t-box `∏_i[t_min_i,1]` by the **sign-coupling-cut set** `𝒯_D` =
that box with the **proved-unreachable joint-saturation corners excised** (a single shared `s∈[−1,1]^n`
cannot saturate two coordinates whose `W`-rows demand conflicting signs on a shared `s_j`). The theorem in
§3 proves `𝒥_true ⊆ {J(t):t∈𝒯_D} ⊆ {J(t):t∈𝒯_box}`, so certifying contraction over `𝒯_D` is a VALID
proof; the §4 pre-registration tests, behind a HARD proof-based gate (NOT the vacuous ρ oracle), whether
`𝒯_D` recovers the switched-expansive residual at n=3,4.
