# PRE-REGISTRATION — M3 (③本丸): is QD/behavioral niching load-bearing on a REAL LLM loss landscape?

> Written 2026-06-06 BEFORE any GPU run. The last open path of the ③ arc: Step4 showed MAP-Elites is
> genuinely load-bearing on synthetic deceptive corridors; Step C / ladder-1 / E-A / Step D / BG9
> showed every real CPU substrate fails to host ③ (landscapes smooth/unimodal, or the hard part is
> low-dimensional so random-restart teleports). BG9's structural verdict (`../kernel_diversification/
> BG9_VERDICT.md` §2, §4): **③ wins only when the difficulty lives in a high-dimensional behavior
> space whose good region is measure-zero under direct sampling.** The pre-registered GPU hypothesis
> (BG9 §4, quoted): "③が full-LLM で load-bearing なら、その難所は高次元 behavior 空間にあり直接
> サンプル/backprop で到達困難なはず".

## 1. Question and honest scoping

**Q (③):** On a real (small) LLM loss landscape — held-out CE of the Stage-B hybrid Transformer,
restricted to its verified-core subspace `(decay, W)`, trunk frozen after gradient warm-up — does
MAP-Elites (behavioral-archive niching) reach better fitness than the strong direct-sampling
baselines (random search, random-restart hillclimb, panmictic GA) at equal evaluation budget?

**Scoping (declared up front):**
- "Real LLM loss landscape" here = the **core subspace** (n=64 ⇒ 4,160 continuous dims) of the
  Stage-B char-LM, NOT the full parameter space. This is the same substrate the whole arc certifies
  and evolves; it is high-dimensional in BG9's sense (unlike kernel_id ∈ {0..3}), and its fitness is
  a genuine LM held-out CE. Full-weight-space evolution at this budget would be vacuous (HD-1 showed
  even (1+1)-EVO underperforms there); the core subspace is where evolution is a live contender.
- **Gradient is reported as a separate strong baseline, not a ③ judge.** Per BG10/HD-1/Stage-B,
  Adam on the same core is expected to dominate all selection methods. ③'s verdict is about the
  *selection family* (does niching beat direct sampling); the GRAD row tells us whether the whole
  family matters when gradients exist. Both facts are reported; neither is hidden in the other.
- Gates (cert_inf) are NOT applied during M3 evolution (orthogonal question, settled by HD-1/
  Stage-B); the final core's `cert_inf` / empirical ρ are recorded as metrics only.

## 2. Methods (equal eval budget E, CRN-paired seeds, identical eval batches)

Fitness = −CE on fixed held-out eval batches (np rng `seed+7`, identical across methods/candidates).
Genome = core `(decay ∈ [0,1]^n, W ∈ [-2,2]^{n×n})`, mutation = Gaussian σ=0.12 + clip (HD-1's).
Warm start: all methods start from the same gradient-warmed frozen-trunk model and the same initial
core (CRN). One fitness evaluation = one budget unit; E is identical across M1–M4.

- **M1 random search** — E independent genomes sampled like init; best kept.
- **M2 RR-hillclimb** — (1+1) greedy with restart every R=E/10 evals (restart = fresh random genome;
  the BG9 teleport baseline).
- **M3 panmictic GA** — pop 16, tournament-2, mutation-only (no crossover; matches arc precedent).
- **M4 MAP-Elites (③)** — archive 8×8 over a pre-registered 2-D behavior descriptor; uniform-random
  elite selection → mutate → place-if-better.
- **M5 GRAD** — Adam (lr 3e-3) on the core only; budget = E/3 optimizer steps (a forward+backward ≈
  3 forward-evals, FLOP-matched approximation; wall-clock also reported). Reported alongside, per §1.

**Behavior descriptors (both pre-registered; M4 runs once per descriptor):**
- **B1 (dynamics):** `(mean(decay), mean(|W|) row-sum / 2)` — each clipped to a pre-registered box
  ([0,1] × [0,1]) and binned 8×8. CLT-concentrated under random sampling (the Step4 mechanism), so
  extreme cells are stepping-stone-only territory.
- **B2 (functional):** fixed random 2-D projection (seed-pinned Gaussian matrix) of the 8-dim
  per-shard CE vector (held-out split into 8 contiguous shards), tanh-squashed to [-1,1]² and binned
  8×8. "What the model is good at," compressed.

## 3. Controls (validity gates — judged BEFORE the real verdict)

- **P+ (positive control, Step4 transplant into core space — AMENDED pre-run after red-team):**
  synthetic deceptive-corridor fitness on the SAME genome:
  `f = max(0.5 − |b₁ − 0.5|, 5·(1 − |b₁ − 0.9|) − 4)` with **`b₁ = mean(W_slice)/2 + 0.5` over a
  FIXED 24-entry probe slice** `W.flat[:24]` (red-team blocker fix: the original full-4096-mean
  coordinate is CLT-frozen — per-mutation drift σ/4096^0.5 ≈ 0.0019 makes the corridor untraversable
  by EVERY method incl. MAP-Elites, guaranteeing N/A; the 24-entry slice restores Step4's
  dimensionality, drift σ/24^0.5 ≈ 0.0245/step, while the peak at slice-mean 0.8 stays ~50σ of the
  *sampling* distribution away = still teleport-proof). Local optimum at b₁ ≈ 0.5 (sampling mass),
  deceptive valley, global peak band at b₁ = 0.9. **P+ uses descriptor B1p = (b₁, mean(decay)) —
  corridor-aligned by design, giving MAP-E its best shot (named explicitly here; the original text
  said "B1" — text/code drift fixed)**. **Harness validity requires MAP-Elites(B1p) to beat M1, M2,
  AND M3 on P+** (≥3/4 seeds). **Required pre-GPU: a CPU simulation of P+ at n=64/E=6000 must show
  MAP-E crossing while M1/M2/M3 stall — recorded below before any GPU push.** If P+ fails ⇒ verdict
  N/A, real rows exploratory only.
- **N0 (negative control):** smooth strictly-concave fitness `f = −‖gene − gene₀‖²` (unimodal, no
  deception). All methods must converge similarly; a MAP-E "win" here ⇒ harness false-positive ⇒ N/A.

## 4. Pre-registered verdict rules (3-value, arc convention)

Primary comparison on REAL fitness (held-out CE), 4 seeds, paired:
- **The PRIMARY verdict descriptor is B1** (it owns the P+ control; red-team fix for the 1-of-2
  multiplicity that would have ~doubled the false-PASS rate). **B2 is secondary/exploratory** — a
  B2-only win is reported as "exploratory positive, unvalidated descriptor", never as ③ PASS.
- **③ PASS** — MAP-E(B1) beats M1, M2, and M3 (mean and ≥3/4 seeds), with P+ valid and N0 null,
  and the winning margin above the measurement-jitter floor (1e-4 in fitness; GPU nondeterminism).
  (Then: does it approach M5 GRAD? reported, not gated.)
- **③ NEGATIVE (decisive)** — P+ valid, N0 null, but MAP-E fails to beat the direct-sampling
  baselines on the real landscape ⇒ "③ is not load-bearing on the real small-LLM core landscape" —
  closing the arc's last path with the same 3-stage evidence shape as BG9 (informative negative).
- **N/A** — P+ invalid or N0 false-positive: scale/budget-limited harness, no real-landscape claim.

Prediction registered (from the arc, honest prior): **③ NEGATIVE is the most likely outcome** —
HD-1 found the real core landscape lets gradient profit and plain EVO drift without payoff,
suggesting smooth-enough navigability for direct sampling; the BG9-style "RR teleports" risk
re-materializes as "Adam/RR navigate the real landscape directly". ③ PASS would be the surprise
worth its own follow-up; either way the long-gated GPU question gets its pre-registered answer.

## 5. Budget & sizes

Feasibility: n=64, E=1,500, 2 seeds, B1 only, P+/N0 included (~preview only, no verdicts).
Full: n=64, E=6,000, 4 seeds, descriptors B1+B2, all five methods + P+/N0. Single Kaggle T4
session each (eval = 4 fwd batches ≈ 0.1–0.2 s ⇒ ~25–50 min/seed-set, checkpointed/resumable).
n=256 extension only if the full run leaves session budget (reported as exploratory if so).

## 6. Honest limits (declared up front)

- Core-subspace scoping (§1) — claims are about THIS landscape; "full-weight-space ③" stays open
  (and is compute-infeasible here, disclosed not dodged).
- 4 seeds: sign-consistency + magnitudes, no p-theater (arc convention).
- Budget-sensitivity (HD-1 lesson): conclusions only from the full run; feasibility = preview.
- Behavior-descriptor dependence: ③ NEGATIVE under B1+B2 does not exclude some untested descriptor
  unlocking ③ — the claim is scoped to the two pre-registered descriptors (chosen for the BG9
  mechanism and functional-diversity rationale, not post-hoc).
- The P+ corridor is in B1p's behavior space; B2 has no synthetic positive control (disclosed: B2's
  harness validity is inherited from the shared MAP-E machinery, weaker — hence B2 = secondary only).

## 7. Pre-run amendments (3-lens adversarial review, 2026-06-06 — ALL before any GPU run)

Red-team (design / fairness / torch lenses; 2 blockers + 3 majors + minors) forced these changes,
each verified by CPU simulation where applicable:
1. **P+ corridor re-based onto a fixed 24-entry probe slice** (blocker: full-mean coordinate
   CLT-frozen at n=64 ⇒ N/A guaranteed; reviewer simulated all methods stuck at 0.5 at E=6000).
2. **B1 axis-2 re-based onto the probe slice** (blocker: `mean(|W| rowsum)/4` saturates in bin 7 for
   100% of reachable genomes — archive collapsed to ~2-6 of 64 cells, structurally crippling M4).
   New B1 = `(mean(decay), mean(|W_slice|)/2)`, both axes mutation-mobile (~0.015–0.025/step).
3. **M2 = warm-restart RR** (major: cold random restarts are a liability — not the BG9 teleport —
   on a warm-started real landscape; restart now re-perturbs the warm core at 3σ).
4. **B1 primary / B2 secondary** (major: 1-of-2 descriptor PASS was an undisclosed 2× multiplicity).
5. **B1p naming for P+** made explicit (major: silent text/code descriptor drift on the gating control).
6. Minor fixes: GA partial-final-generation for exact budget parity; B2 empty-octile imputation
   (CE=0 contamination) + std-adaptive scaling; M5 restores the shared model core afterward + grad
   assert; smoke E=64; eval_batches=3 (text said 4); jitter floor 1e-4 for win adjudication.
7. **L0 landscape suite added** (user request: "GPU が使えるなら landscape の拡張も"): per seed,
   1,000 genomes (600 random + 100×4 warm-perturbed σ ∈ {0.03, 0.12, 0.5, 1.0}) → real CE + B1
   coords. Descriptive (no gate): measures direct-sampling difficulty (fraction of random genomes
   beating warm) and local smoothness (perturbation CE vs σ) — the BG9-style mechanism evidence for
   whichever verdict the M-suite returns.
8. **P+ corridor: second amendment after validation iterations (ALL still pre-GPU).** The 24-slice
   V-shaped corridor (amendment 1) ALSO failed CPU validation (0/4): a sloped deceptive valley pulls
   within-bin elites inward and breaks the archive ratchet. Final design = **faithful transplant of
   Step4 exp2 `deceptive_eval`** (the arc's PROVEN positive control): behavior = means of two
   10-entry halves of a fixed 20-entry W probe under the unit window `u = clip(w+0.5, 0, 1)`;
   fitness = max(0.6·local Gaussian at (0.3,0.3), σ=0.18; 1.0·global at (1,1), σ=0.07) **+
   observation noise N(0, 0.01)** (Step4's `_NOISE` — load-bearing: it powers neutral archive churn
   across the flat stretch); recorded bests are honest-rescored noiselessly; archive grid 12×12 and
   MAP-E init batch = max(20, E/10) (both Step4-faithful; 8×8/16 were too coarse/thin).
   **Synthetic suites run at E_synth = 20,000** (numpy-cheap; the REAL suite stays at E = 6,000 —
   GPU-budget-bound). P+ win margin for validity = 0.05 (decisively above noise).
9. **P+ CPU validation (n=64, E_synth=20,000, 4 seeds, margin 0.05): PASS 4/4** —
   M4 reaches the global corner (0.9913 / 1.0000 / 0.9998 / 0.9159) while M1 ≈ 0.33-0.36,
   M2 ≈ 0.594-0.599, M3 ≈ 0.593-0.600 stall at/below the local optimum. The harness CAN detect ③
   at this genome dimension when the landscape hosts it. (At the real budget E=6,000 the crossing is
   seed-dependent, 2/4 — disclosed: the validity gate certifies the machinery at E_synth, and the
   real-landscape verdict is correspondingly scoped to "③ at E=6,000 on this landscape".)
10. **B1 occupancy re-validated**: 61/144 bins reachable under an E-scale random walk (was 2-6/64
   before the fix); init sampling occupies ~5 bins (stepping-stone structure intact).
