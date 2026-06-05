# Navigability thread — PRE-REGISTRATION (verifier strictness ⇄ evolvability trade-off)

> **Origin.** The R-LLM-0 L3 result (`../verified_lm_evolution/VERDICT.md §3c`) found that the conservative
> inf-norm gate's harm under evolution is **navigability, not ceiling**: the inf region *contains* genes
> 0.118 nats below unigram, yet inf-gated evolution collapses to unigram, because random mutations almost
> never land in the tiny inf-feasible set. This thread promotes that finding to a first-class, falsifiable
> research question — and gives the prior verified-evolution arc's conclusion ("the right contraction
> verifier is SDP/Lyapunov, not inf-norm") a **new reason**: not just *coverage* (SDP certifies more
> contracting genes) but *navigability* (the SDP feasible set is traversable by evolution; the inf one is a
> trap). TRIZ framing: the constraint (soundness) and the strength (evolvability) appear to conflict — the
> resolution is a verifier that is **sound ∧ navigable**.
>
> **Discipline (llcore standard):** design-first → smallest falsifiable PoC → honest measurement
> (CLARABEL-pinned, 0-unsound independently confirmed) → adversarial red-team + Codex pair-review →
> research/ isolated, src/ untouched, push 未. Negatives are valid results.

## 0. One-line thesis (falsifiable)

**"The reachable fitness of evolution under a sound contraction gate is governed by the *navigability* of
the gate's feasible set (how easily mutation can traverse it), not merely by the set's *ceiling* (the best
gene it contains). An over-conservative-but-sound gate (inf-norm) traps evolution far below a ceiling it
provably contains; a less-conservative sound gate (SDP) is both sound and navigable."**

Refuted if: the reachable-fitness ordering across gates is explained by ceiling differences alone (not by
navigability / acceptance-rate), OR if no gate exhibits a meaningful navigability gap (reached ≈ ceiling
for all), OR if the effect is a pure artifact of one optimizer that vanishes under a fair search.

## 1. Substrate (reuse, unchanged)

Reuse `../verified_lm_evolution/lm_substrate.py` (n=8 byte-LM, deterministic) + the arc certifiers
`../verified_evolution_sdp_gate/coupled_nd.py` (cert_inf/two/sdp, CLARABEL-pinned). Gates = none / inf /
two / sdp. Corpus = 8192 B (the gated-run corpus, so landscape ceilings and gated reach are directly
comparable — the L3 §3c lesson). research/ isolated; src/ untouched.

## 2. Pre-registered FALSIFIABLE gates

| gate | claim | PASS condition | how measured | status |
|---|---|---|---|---|
| **N0 navigability gap** | reachable < ceiling, and the **gap grows with strictness** | gap(inf) > gap(two) > gap(sdp), monotone, robust over ≥10 seeds (gap = region-ceiling_from_landscape − gated-reached CE) | 8192B landscape best-CE per region vs gated reached CE | **already supported**: inf 0.117 > two 0.102 > sdp 0.073 (1 landscape × 10 gated seeds) — to confirm over seeds + bootstrap |
| **N1 acceptance rate = mechanism** | stricter gate → lower child-admit rate → fewer viable moves → bigger gap | admit_rate(inf) ≪ admit_rate(sdp); gap correlates with (1−admit_rate), ρ<−0.7 | instrument `evolve()` to log per-gate fraction of proposed children the gate admits (and resample_cap hits) | TODO |
| **N2 the region holds the good genes (trap ≠ ceiling)** | inf-gated evolution leaves *reachable, sound* fitness on the table | a non-trivial fraction of inf-region landscape genes (≥10%) beat the inf-gated reached fitness | count landscape inf-region genes with CE < gated-inf-reached CE | **already supported** (8192B inf region best 3.44 ≪ reached 3.557); formalize the fraction |
| **N3 connectivity / can stronger search escape?** | does inf-gated reach below unigram with *more exploration*? | run inf-gate with (a) larger mutation σ, (b) random restarts, (c) seed-from-best-inf-gene (Codex #10). YES it improves ⇒ navigability/EA-tuning; NO even with aggressive search ⇒ feasible set genuinely disconnected | 3 inf-gate variants vs baseline inf-gate, paired seeds | TODO (either outcome informative) |
| **N4 the sound sweet spot (TRIZ resolution)** | sdp is the option that is sound **and** navigable; none's extra navigability is bought with *unsound* genes | sdp ≫ inf on reachable fitness while 0% expansive; none reaches lower CE but its winners are `non_certified` (expansive) | reachable-fitness vs strictness curve + winner-region + %expansive per gate | **partially supported** (gated winner-regions: none→non_certified expansive; sdp→sdp_only 0% expansive) |
| **N5 optimizer-independence (CPU preview of GPU stage A)** | the gap-ordering is a property of the *feasible-set geometry*, not the EA | repeat N0 with RR-hill-climb (and, if cheap, a CMA-ES-style search) under each gate; gap-ordering holds across optimizers ⇒ geometric; only the EA is trapped ⇒ optimizer-specific | 2nd optimizer × 4 gates × seeds | TODO (feeds the GPU gradient-vs-evolution test, `../verifier_navigability_gpu/`) |

## 3. Honest framing (what this thread does and does NOT claim)

- It does **NOT** revive the L3 "verifier helps the LM learn language" claim (the null doesn't tie, §3c).
- It **DOES** explain *why* a better verifier helps *evolution* reach lower loss: navigability. This is a
  statement about optimizer × feasible-set geometry, largely **structure-independent** (consistent with the
  null not tying — the gap is about the search landscape's shape, not the task's signal).
- The contribution: **verifier–optimizer co-design.** "Choose the verifier that is sound *and* whose
  feasible set your optimizer can traverse." Connects to and strengthens
  `[[../verified_evolution_sdp_gate/VERDICT.md]]` (SDP is the right verifier — now also for navigability) and
  to the third-axis arc's structural insight (`[[project_llcore_init_2026_05_29]]`: ③ load-bearing needs
  high-dim behavior — here the dual: an over-tight constraint collapses the navigable dimension).

## 4. Execution params (CPU tractability)

n=8, vocab=256, 8192 B corpus. Gated evolve pop12/gens10 (reuse). N0/N2 reuse existing JSON +
landscape. N1 needs an `evolve()` admit-rate probe (additive, fitness-invariant). N3/N5 new small runs.
CLARABEL pinned; 0-unsound re-confirmed via the JSR oracle. ≥10 seeds where a paired test is claimed;
lower with honest disclosure if CPU-bound. Each gate gets Codex pair-review before its verdict.

## 5. Adversarial red-team (post-measurement)

- **Lens A — gap = ceiling-noise?** Is the navigability gap just landscape best-CE sampling noise? Use
  bootstrap CIs on both ceiling and reached; require non-overlap.
- **Lens B — admit-rate confound.** Does low admit-rate merely slow convergence (would more gens close the
  gap)? Run inf-gate with 3× gens; if the gap persists, it's a trap, not slowness.
- **Lens C — circularity.** Region label (cert) and reachability share only the gene; confirm non-leaky.
- **Lens D — optimizer cherry-pick.** N5 must use a *fairly tuned* second optimizer (not a strawman).
- **Lens E — corpus robustness.** Re-check N0 on a 2nd corpus slice; gap-ordering should survive.
