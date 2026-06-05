# Codex pair-review (2026-06-06) — BG10 + L3 §3c + cost-reduction PoC

External reviewer = Codex (gpt-5.x), read-only (`codex exec -s read-only`, prompt via **stdin pipe** —
the earlier arg-form hung on "Reading additional input from stdin..."; the stdin pipe is the fix).
Two-pillar discipline ([[reference_codex_two_pillar]]): every finding re-checked against the raw JSON
before adoption. Tokens used by Codex: ~66.8k.

## Overall verdict (Codex)
> "Nothing overturns the core results; artifact A's audited §3c numerics check out against the JSON, and
> the substantive issues are scoping/wording over-claims in BG10 plus the theorem-strength phrasing in PoC-1."

`research/verified_lm_evolution/VERDICT.md` §3c numerics: **verified correct against the result JSONs** by
Codex independently (matching the in-house 3-lens verification Workflow). No change needed there.

## 3 MAJOR findings (all scoping/wording; all verified against data; all adopted)

1. **BG10 headline over-claims the trap.** "for a gradient-trained LM the contraction gate is essentially
   FREE" — the data show a large **admit-rate** trap for EVO (inf ~1% vs none 100%) but NOT an end-loss
   trap: EVO mean CE is nearly tied / non-monotone (`inf 2.6138 ≈ two 2.6167 ≈ sdp 2.6179 ≈ none 2.6198`;
   inf fractionally best). Verified vs `result_confirm.json`. **Adopted:** NAVIGABILITY_GPU_VERDICT.md
   one-line scoped to "free *in final CE*" + a Scope/honesty box; this was already partly disclosed in the
   verdict's Honest-limits and the paper §6 G2 caveat (EVO trap masked in CE by the gradient-warm wrapper).

2. **BG10 generalizes beyond the run + GRAD reject is gate-dependent.** "pick the verifier on
   soundness/coverage, not navigability" is one n=8/1-layer/6-seed CPU run; and GRAD's per-step reject
   rate is gate-dependent (`none 0.000 / sdp 0.0125 / two 0.0667 / inf 0.1514`), so "free" means free in
   final CE, not zero interaction. Verified vs `result_confirm.json`. **Adopted:** scoped the BG10 verdict
   one-line and the paper §6 contribution bullet to "this n=8 setup / final CE, not a proven law"; the
   paper §6 G3 already noted the ~0.15 inf reject.

3. **PoC-1 "Conservative-by-construction confirmed" conflates proof with the run.** The JSON shows only
   0 *observed* violations on the 3000-gene sampled pool, not a machine-checked proof. (The M+R bound IS
   provably an upper bound — that is a paper-level argument; the run is a from-below consistency check of
   it.) **Adopted:** SKETCH.md PoC-1 soundness row reworded to "provably an upper bound (conservative by
   construction); the run's 0 observed violations is a from-below consistency check, not a proof" —
   aligning with the project's standing "soundness is a theorem, the 0% is a consistency check" discipline.

## Files touched
- `research/verifier_navigability_gpu/NAVIGABILITY_GPU_VERDICT.md` (one-line + Scope/honesty box)
- `research/verifier_cost_reduction/SKETCH.md` (PoC-1 soundness wording)
- `research/paper/PAPER_DRAFT.md` (§6 contribution bullet scoped)

No core result changed; all edits tighten scope/wording per honest disclosure. push deferred (no remote).
