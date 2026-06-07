# Codex pair-review — R-LLM-0 L3 PAYOFF claim (gpt-5.4, read-only, 2026-06-05)

Numbers passed inline (no file reads, to dodge the recurring file-read timeout). 16 findings; **none
overturn the core result** — all are wording/disclosure precision and claim-scoping. Verified each
against the actual numbers before accepting ([[feedback_external_ai_verify]]). Dispositions:

| # | type | finding | disposition |
|---|---|---|---|
| 1 | **BLOCKER** | "region label and CE are *independent*" is wrong — non-circular ≠ statistically independent (both are downstream of the same `(decay,W)`) | **FIX**: reword to "outcome-blind / non-leaky" (label assignment never reads CE/readout/corpus); drop "independent" |
| 2 | OK | frac=1.0 over 10 paired seeds → exact one-sided sign **and** Wilcoxon p = 1/2¹⁰ = 0.000977 (two-sided 0.001953) | confirmed; keep "p≈0.001 one-sided, all 10 diffs>0 no ties" |
| 3 | disclose | Bonferroni denominator = 3 if `none` is counted (0.05/3=0.0167), not 2 (0.025) | **FIX**: state 2 pre-reg sound comparisons (two,sdp vs inf)→0.025; passes even at 3→0.0167 |
| 4 | narrowing | drop strict 4-rung ladder (two-vs-sdp median reversed, MW≈0.5) | already done |
| 5 | narrowing | landscape `inf<<{two,sdp}` is a **best-of-region (existential)** gap; median gaps small (~0.005-0.008) | **FIX**: keep inf-vs-relaxed distributional (MW p<0.01, worst median) but mark the *magnitude* gap as best-found/existential |
| 6 | OK | inf bigger (346) yet worse best-CE → not a sample-count artifact | keep |
| 7 | disclose | "P=0.0000" is not a valid p-value | **FIX**: "0 of N resamples, P<1/N" |
| 8 | narrowing | live **search-space/expressivity** confound: relaxed regions are different parameter subsets that may simply contain better-for-task dynamics → claim is "sound relaxation **admits access to** lower-CE genes", NOT "the certifier improves modeling" | **FIX**: add explicit search-space-expansion framing |
| 9 | disclose | `corr(CE,ρ)=+0.145` is weak (linear, pooled) — not a full refutation of an edge-of-chaos confound | **FIX**: soften "ruled out" → "weak negative evidence against a *linear* confound" |
| 10 | OK | 8192B inf-region landscape is the right (A)-vs-(B) discriminator; stronger: also seed inf-gated EA from the best known inf gene | running; note the seed-from-best follow-up |
| 11 | narrowing | "inf pinned at unigram" too strong → "inf-gated **search** collapsed to the unigram solution" (not "the inf region cannot beat unigram") | **FIX** wording |
| 12 | disclose | identical inf fitness (0.02852885…) across 10 seeds → tiny feasible region OR degenerate default readout? disclose + investigate | **FIX**: disclose; 8192B landscape + seed-from-best will probe |
| 13 | narrowing | null control not strictly required for the narrow gate-comparison, but load-bearing to rule out pipeline/eval artifact | running (`run_l3b.ps1`) |
| 14 | disclose | "0% empirically-expansive" is a **consistency check**, not soundness proof (soundness = certifier theorem) | **FIX**: separate the two explicitly |
| 15 | disclose | landscape (12288B) and gated (8192B) are different corpora — note when triangulating | already disclosed; keep |
| 16 | OK | strongest defensible claim: "relaxing inf→sound-relaxed certificates **expands the admissible search space** and, in both random search and gated evolution, yields better **best-found** held-out CE than inf; two-vs-SDP unresolved; inf-region-on-8192B not yet shown to lack better-than-unigram genes" | adopt as the canonical wording |

**Bottom line:** Codex finds **no blocker to the result**, one terminology blocker ("independent"), and a
set of scoping/disclosure refinements. The defensible claim (finding 16) is: a sound relaxation of the
contraction certifier **expands the admissible search space to include lower-CE genes**, confirmed in
both random search and 10-seed gated evolution (p≈0.001), on a real n=8 byte-LM — pending the null
control and the 8192B ceiling-vs-navigability check.
