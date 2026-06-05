# R-LLM-0 L3 VERDICT — Verified evolution INSIDE a CPU-real tiny byte-LM

> **Status (2026-06-05): landscape mechanism COMPLETE + adversarially red-teamed; gated-evolution
> reachability CONFIRMED (10 paired seeds, pre-reg gate L3 PASS, frac=10/10); honest-null control +
> same-corpus (8192B) ceiling-vs-navigability check IN PROGRESS (`run_l3b.ps1`, NT=1, checkpointed).**
> Pre-registration: `PREREGISTRATION.md`. Substrate: `lm_substrate.py`. Certifiers (reused, unchanged):
> `../verified_evolution_sdp_gate/coupled_nd.py`. research/ isolated, src/ untouched, push 未.

---

## 0. One-line result (honest, narrowed by red-team)

**On a real n=8 byte-level language model, relaxing the contraction certifier from inf-norm to a
less-conservative BUT STILL SOUND certificate (two_norm / sdp) expands the admissible
provably-contracting search space to include genes with strictly lower held-out perplexity — confirmed
in both random search (landscape) and 10-seed gated evolution (p=0.000977). The arc's signature payoff
holds on real LM loss, not just the synthetic rotation task. The load-bearing effect is the
*inf-vs-sound-relaxed* gap (best-found ~0.04–0.085 nats; distributional MW p<0.01), all admitted genes
0% empirically-expansive. NOT claimed: the finer two_norm-vs-sdp ordering (sampling noise), that the
certifier itself "improves modeling" (it is search-space expansion), or that the inf region cannot beat
unigram on the gated corpus (its EA search merely collapsed to unigram — open, see §3b).**

This *reverses* the underpowered 40/60-gene smoke (which had `inf` winning by luck of a thin shell);
the pre-registered 900-gene landscape, with every region well-sampled, shows the opposite and correct
ordering. (Lesson logged: do not read a frontier off an underpowered pool — [[feedback_benchmark_honest_disclosure]].)

**Two complementary pieces of evidence agree:** (i) the **landscape** (random sampling) shows the sound
relaxed regions *contain* lower-CE genes than inf (region-ceiling / admissibility); (ii) **gated
evolution** (10 paired seeds, §3b) shows evolution *under* a sound relaxed gate actually *reaches* lower
CE while the inf gate is pinned at unigram (reachability, pre-reg L3 PASS, 10/10). Still open: the
ceiling-vs-navigability mechanism for inf's pinning (being resolved by an 8192 B landscape), and the
honest-null control (running).

---

## 1. L0 — the tiny reservoir LM actually functions (PASS)

Landscape corpus 12288 B, `unigram_CE = 5.2399`. Every certifier region's best held-out CE beats the
Laplace-smoothed byte-unigram baseline by **0.40–0.57 nats**:

| region | n | best contracting CE | beats unigram by | % empirically expansive |
|---|---|---|---|---|
| inf | 346 | 4.8377 | 0.4022 | 0.0 |
| two_norm_only | 100 | 4.7954 | 0.4445 | 0.0 |
| sdp_only | 189 | 4.7525 | 0.4874 | 0.0 |
| non_certified | 265 | 4.7052 (raw 4.6684) | 0.5715 | **78.9** |

Baseline is not a strawman: the logistic readout's zero-feature limit *exactly* recovers the unigram
(zero-init + bias column), so any gain is genuine sequential signal. No temporal leakage (held-out is
strictly later than train). **L0 holds.**

## 2. L1 / L2 — soundness on the real substrate (PASS)

- **Soundness is a theorem, the 0% is a consistency check (Codex #14).** The certificate's soundness
  comes from the certifier *proof* (Lemmas 1–3 + cert_inf/two/sdp theorems, `PREREGISTRATION.md §2`).
  The empirical figures below are a *from-below consistency check* that the certifier admitted nothing
  *observably* expansive — they corroborate, they do not constitute, soundness.
- **L1 (consistency holds):** all three *certified* regions (inf / two_norm / sdp) are **0.0%**
  empirically-expansive on the real byte-LM — no admitted gene was observed expansive.
- **L2 (oracle non-vacuous / gate load-bearing):** `non_certified` is **78.9%** expansive — the gate
  excludes a large genuinely-expansive population, so certification does real work (the oracle can and
  does falsify).
- **Non-circular but NOT statistically independent (Codex #1).** `classify_region` is **outcome-blind /
  non-leaky**: a pure function of the Jacobian box `(decay,W) → t_min_per_coord → infnorm/SVD/SDP-LMI`,
  never reading reservoir states, readout, corpus, or CE; and `held_out_ce` never calls any certifier.
  So the label is not computed *from* the outcome (no leakage/circularity). It is **not** statistically
  independent of CE — both are downstream of the same sampled `(decay,W)` — which is exactly why the
  region carries fitness signal.
- **Readout controlled:** byte-for-byte identical readout (zero-init, fixed `readout_steps`/`lr`/`l2`,
  fixed shared embedding + split) for every gene — the CE ordering reflects core dynamics, not readout
  capacity. A linear edge-of-chaos (ρ) confound is **weakly** disfavoured: `corr(CE, emp_rho) = +0.145`
  (wrong sign for "higher ρ → lower CE"); this is weak negative evidence against a *linear, pooled*
  confound only, not a full refutation (Codex #9).

## 3. L3 — the verifier-perplexity frontier (mechanism PASS; reachability IN PROGRESS)

### 3a. Mechanism / region-ceiling (landscape, COMPLETE — survives red-team)

The conservative **inf region is distributionally excluded from the low-perplexity tail**: **0 of its
346 contracting genes** reach CE < 4.82, versus 8–25% for the less-conservative sound regions. inf is
the *largest* certified shell (346 > sdp 189) yet has the *worst* ceiling — the opposite of a
more-tickets sampling artifact. Subsampling inf down to sdp's N never reaches sdp's best (**0 of 20,000
resamples, P < 5e-5** — not literally zero, Codex #7); Mann-Whitney puts the whole inf shell
**distributionally** worse than two_norm (p=0.008) and sdp (p=0.007), so the inf-vs-relaxed effect is
not merely an order statistic. **A sound relaxation (two_norm / sdp, both 0% empirically-expansive)
*admits access to* genes ~0.04–0.085 nats better than inf's best.**

**Read this as search-space expansion, not certifier "improvement" (Codex #8/#16).** The relaxed regions
are *different parameter subsets* `(decay,W)` that happen to contain better-for-this-task dynamics and
*require a stronger certificate to prove sound*. The honest mechanism is: **relaxing inf→sound-relaxed
certification expands the admissible (provably-contracting) search space to include lower-CE genes** —
not that the certifier itself improves modeling. The *magnitude* gap (4.8377→4.7525) is a best-found /
existential frontier figure (median gaps are small, ~0.005–0.008); the load-bearing distributional
claim is the inf-vs-{two,sdp} shift (MW p<0.01).

**Honest narrowing (red-team, 2 dissents forced this):** the strict 4-rung monotone ladder
`inf > two > sdp > non` does **not** survive. The two_norm-vs-sdp inner rung is sampling noise:
medians *reverse* it (two 4.8766 < sdp 4.8793), MW p≈0.5, bootstrap best-CE CIs overlap, and a
best-of-N simulation predicts sdp's lower min purely from its larger N (189 vs 100). So the claim is
**inf ≪ {two_norm, sdp}** (both sound), with two_norm and sdp mutually indistinguishable, and
non_certified excluded as unsound. **Not** a strictly-monotone multi-rung law.

### 3b. Reachability under evolution (gated, 10 paired seeds — PRE-REG GATE L3 PASSES)

Does evolution *under* a looser sound gate actually *reach* lower CE (vs the region merely *containing*
better genes)? Gated evolution (`exp_gated.py`, gates none/inf/two/sdp, CRN-paired, pop12/gens10,
8192 B corpus, `unigram_CE=3.5571`). **10 paired seeds** (`exp_gated_real10_results.json`):

| gate | mean fitness | mean CE | winner region (all 10 seeds) | vs inf (paired) |
|---|---|---|---|---|
| inf_norm | 0.028529 | 3.5568 | inf | — (= unigram exactly) |
| two_norm | 0.029277 | 3.5310 | two_norm_only | **+0.00075, 10/10** |
| sdp | 0.029780 | 3.5140 | sdp_only | **+0.00125, 10/10** |
| none (ungated) | 0.030422 | 3.4926 | non_certified | +0.00189, 10/10 |

**The pre-registered L3 gate PASSES decisively.** Both sound relaxations beat the conservative inf gate
in **10/10 seeds** (`frac_a_gt_b = 1.0`) → exact one-sided sign **and** Wilcoxon signed-rank p = 1/2¹⁰ =
0.000977 (Codex #2), past Bonferroni for the 2 pre-registered sound comparisons (0.05/2 = 0.025) — and
even at 3 comparisons including ungated `none` (0.05/3 = 0.0167). Mechanism is clean: each gate's best
gene lands **exactly in its own certifier region** (inf→inf, two→two_norm_only, sdp→sdp_only,
none→non_certified). **inf-gated *search* collapsed to the unigram solution** — identical fitness
0.02852885423306474 across all 10 different seeds (= `exp(−unigram_CE)` to 8 sig figs), i.e. zero
improvement over no-context — while the sound sdp gate recovers ~66% of the *ungated* improvement (sdp
CE 3.514 vs none 3.493 vs unigram 3.557), staying inside the sound `sdp_only` region. So **a stronger
sound verifier lets evolution REACH lower real-LM perplexity; the conservative gate's search forfeits it
entirely** — admissibility upgraded to reachability.

*Disclosure (Codex #11/#12):* "search collapsed to unigram" is a statement about the **EA outcome**, not
a proof that the inf region's ceiling *is* unigram. The exact-identical inf fitness across 10 seeds means
inf-gated EA is effectively deterministic — either the inf feasible set near the init is tiny, or the
zero-readout (unigram-equivalent) is a degenerate basin no admitted mutation escapes. The 8192 B
landscape (below) probes whether inf genes better than unigram even exist on this corpus.

**Honest open question (ceiling vs navigability):** inf is pinned at *exactly* unigram. Two readings —
(A) the inf region's ceiling ≈ unigram on this 8192 B corpus, or (B) the inf gate is so tight that
evolution cannot navigate it (an evolvability handicap). The 12288 B landscape (§3a) shows inf *does*
contain better-than-unigram genes there, but it is a **different corpus**, so it cannot settle (A) vs
(B) for the gated 8192 B run. An 8192 B landscape (matching the gated corpus) is running to resolve
this. Either way the payoff holds (sound relaxation reaches lower CE under evolution); only the
*mechanism* (ceiling vs navigability) is open. `<<8192B landscape result + ceiling-vs-navigability verdict>>`

- **HONEST-NULL CONTROL (pre-reg gate L3-null, RUNNING `bz2a7xod7`):** `--null` shuffles the corpus →
  sequential structure destroyed → all gates must tie (p>0.1). **Load-bearing falsifier.** `<<null tie result>>`

---

## 4. Red-team summary (8 adversarial lenses; 6 pass, 2 force narrowing)

| lens | survives | note |
|---|---|---|
| L0-sanity | ✓ high | LM beats unigram 0.40–0.57 nats; baseline not a strawman; no leakage |
| sampling-lottery | ✓ high | inf bigger (346) yet worse ceiling; equal-N subsample ladder invariant; 0/346 below sdp best |
| inf-gated-broken-vs-ceiling | ✓ med | landscape (random-sampled) confirms inf's higher ceiling is a region-fact, not an evolvability artifact |
| circularity | ✓ high | region label and CE share only the raw gene; ρ-confound wrong sign |
| readout-confound | ✓ high | identical readout for all genes/regions |
| soundness-L1L2 | ✓ high | certified 0% expansive, non_certified 78.9% |
| **monotone-by-chance** | ✗ high | two-vs-sdp rung is noise (median reversal, MW≈0.5, CI overlap) → drop strict ladder |
| **regime-scope-overclaim** | ✗ high | scope to the single sound inf-vs-relaxed gap on this n=8 byte-LM; drop "gated concurs" until run completes |

**Bottom line (red-team):** a real, sound, mechanistically-explained payoff exists — the sound
relaxation beats the conservative gate on real LM loss with zero expansive admissions — but it is a
**single ~0.04–0.085-nat inf-vs-sound-relaxed gap on one tiny n=8 byte-LM**, not a strictly-monotone
multi-rung law, and (until 3b completes) a **region-ceiling/admissibility** result rather than a proven
**reachability ("unlocks")** result.

---

## 5. Honest limits & next

- reservoir/ESN LM (fixed embedding + per-gene readout), **not** a gradient-trained Transformer →
  GPU stage B (`PREREGISTRATION.md §1`).
- n=8 (the certifiers enumerate 2^n t-box vertices; n=32 is infeasible). Scaling to n=32+ needs the
  **vertex-free sound certifier R-LLM-1** (algorithmic, soundness-first). See
  `CPU_MEMORY_EFFICIENCY_PLAN.md §3`.
- **Next:** (1) finish gated paired run + null control (3b) → promote or retract the reachability clause;
  (2) Codex pair-review; (3) commit (research/ isolated, src/ untouched, push 未).
