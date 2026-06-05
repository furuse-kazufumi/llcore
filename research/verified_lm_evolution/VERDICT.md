# R-LLM-0 L3 VERDICT — Verified evolution INSIDE a CPU-real tiny byte-LM

> **Status (2026-06-05): landscape mechanism COMPLETE + adversarially red-teamed; gated-evolution
> reachability + honest-null control IN PROGRESS (`run_gated.ps1`, NT=1, incremental checkpoint).**
> Pre-registration: `PREREGISTRATION.md`. Substrate: `lm_substrate.py`. Certifiers (reused, unchanged):
> `../verified_evolution_sdp_gate/coupled_nd.py`. research/ isolated, src/ untouched, push 未.

---

## 0. One-line result (honest, narrowed by red-team)

**On a real n=8 byte-level language model, a less-conservative BUT STILL SOUND contraction certifier
admits genes with strictly lower held-out perplexity than the conservative inf-norm gate — the arc's
signature payoff holds on real LM loss, not just on the synthetic rotation task. The load-bearing
effect is the *inf-vs-sound-relaxed* gap (~0.04–0.085 nats, 0% empirically-expansive); the finer
two_norm-vs-sdp ordering is within sampling noise and is NOT claimed.**

This *reverses* the underpowered 40/60-gene smoke (which had `inf` winning by luck of a thin shell);
the pre-registered 900-gene landscape, with every region well-sampled, shows the opposite and correct
ordering. (Lesson logged: do not read a frontier off an underpowered pool — [[feedback_benchmark_honest_disclosure]].)

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

- **L1 (soundness holds):** all three *certified* regions (inf / two_norm / sdp) are **0.0%**
  empirically-expansive on the real byte-LM — the certifier admits no expansive gene.
- **L2 (oracle non-vacuous / gate load-bearing):** `non_certified` is **78.9%** expansive — the gate
  excludes a large genuinely-expansive population, so certification does real work.
- **Non-circular (verified by code):** `classify_region` is a pure function of the Jacobian box
  `(decay, W) → t_min_per_coord → infnorm/SVD/SDP-LMI`; it never reads the reservoir states, readout,
  corpus, or CE. `held_out_ce` never calls any certifier. Region label and fitness share only the raw
  gene.
- **Readout controlled:** byte-for-byte identical readout (zero-init, fixed `readout_steps`/`lr`/`l2`,
  fixed shared embedding + split) for every gene — the CE ordering reflects core dynamics, not readout
  capacity. The ρ-confound (edge-of-chaos) is ruled out: `corr(CE, emp_rho) = +0.145` (wrong sign).

## 3. L3 — the verifier-perplexity frontier (mechanism PASS; reachability IN PROGRESS)

### 3a. Mechanism / region-ceiling (landscape, COMPLETE — survives red-team)

The conservative **inf region is distributionally excluded from the low-perplexity tail**: **0 of its
346 contracting genes** reach CE < 4.82, versus 8–25% for the less-conservative sound regions. inf is
the *largest* certified shell (346 > sdp 189) yet has the *worst* ceiling — the opposite of a
more-tickets sampling artifact. Subsampling inf down to sdp's N never reaches sdp's best (P=0.0000);
Mann-Whitney puts the whole inf shell stochastically worse than two_norm (p=0.008) and sdp (p=0.007).
**A sound relaxation (two_norm / sdp, both 0% expansive) admits genes ~0.04–0.085 nats better than inf.**

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
in **10/10 seeds** (`frac_a_gt_b = 1.0`) → one-sided sign/Wilcoxon p ≈ 2⁻¹⁰ ≈ 0.001, past Bonferroni
(0.05/2). Mechanism is clean: each gate's best gene lands **exactly in its own certifier region**
(inf→inf, two→two_norm_only, sdp→sdp_only, none→non_certified). inf-gated evolution is **pinned at
unigram fitness (0.0285288, identical to ~8 sig figs across all 10 seeds)** — the over-conservative gate
yields *zero* improvement over no-context — while the sound sdp gate recovers ~66% of the *ungated*
improvement (sdp CE 3.514 vs none 3.493 vs unigram 3.557), staying inside the sound `sdp_only` region.
So **a stronger sound verifier lets evolution REACH lower real-LM perplexity; the conservative gate
forfeits it entirely** — admissibility upgraded to reachability.

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
