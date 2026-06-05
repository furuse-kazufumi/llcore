# PAPER_DRAFT.md — adversarial critique record (2026-06-06)

First draft produced by a 9-agent Workflow (7 evidence-grounded section drafters → synthesis →
honest-disclosure/over-claim critic). The critic re-checked every quantitative claim against the
primary result files.

**Verdict: `overturns_core = false`.** No finding overturns any core result. Three MINOR findings:

1. **(applied)** Abstract paired "~87.2% (inf∪B2)" with a "~22% tail" — but 87.2% leaves a ~13% tail;
   the ~22% tail belongs to **B2 alone (77.6%)**. The body (§7) was already correct. Abstract rewritten
   to pair each coverage with its own tail.
2. **(applied)** Abstract called `inf ∪ B2` a "single-SVD bound" — single-SVD is **B2**; `inf ∪ B2` is
   `O(n²) + 1 SVD`. Abstract now separates B2 (1 SVD) from the union gate.
3. **(no action — numbers verified correct)** §2–§3's quantitative claims (rotation 0.411→0.767/0.859,
   region ceilings 0.38/0.89/0.90, ungated 19.7/16.7/1.9% drift, SDP 95%=286/300, Track-D 1291/1363,
   degree ladder, JSR, Rump 286==286, ND ceilings + payoff, R2a nulls) come from DEG6_VERDICT.md /
   ND_VERDICT.md / PAIRREVIEW_audit_rump_2026-06-04.md / RUMP_HARDENING_VERDICT.md (not in the critic's
   4-file list). The critic verified **all** of them against those files: complete agreement, no
   fabrication. Recorded only as a source-tracking note.

**Also applied:** wired the three CPU/SVG figures (`../paper_assets/fig_cost_speedup.svg`,
`fig_admit_coverage.svg`, `fig_l3_gate_gap.svg`) into their matching captions.

**Status:** first draft. §6 spot-check **done** (2026-06-06): the BG10 numbers (unigram 3.2512, GRAD
~2.485, null 3.309) are correctly attributed to NAVIGABILITY_GPU_VERDICT.md and NOT conflated with the
L3 reservoir numbers (unigram 3.5571); added an explicit honest caveat that the EVO trap is masked in CE
by the gradient-warm wrapper and shows only in admit-rate (connecting §6 to the pure-EVO §5). Remaining
(future passes): render the remaining conceptual figure placeholders; add a related-work pass; the
EN/ZH/KO multilingual versions for any public derivative. **Push deferred (no llcore remote).**
