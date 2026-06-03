export const meta = {
  name: 'deg6-adversarial-review',
  description: 'Adversarial multi-lens review of the llcore degree-6 verifier-ladder findings: each skeptic tries to REFUTE one headline claim against the actual code + result JSONs; then a completeness critic.',
  phases: [
    { title: 'Refute', detail: 'one skeptic per headline claim — read code+JSON, try to break it' },
    { title: 'Confirm', detail: 'independently confirm each surfaced confound is real' },
    { title: 'Synthesize', detail: 'completeness critic + consolidated verdict' },
  ],
}

const DIR = 'D:/projects/llcore/research/verified_evolution_sdp_gate'

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['claim', 'refuted', 'severity', 'analysis', 'recommendation'],
  properties: {
    claim: { type: 'string' },
    refuted: { type: 'boolean', description: 'true if the claim is wrong/overstated/confounded' },
    severity: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'blocker'] },
    analysis: { type: 'string', description: 'the specific confound/bug found, or why the claim survives, citing concrete numbers/lines' },
    recommendation: { type: 'string', description: 'what to change in code or verdict wording' },
  },
}

const claims = [
  {
    key: 'certifier-soundness',
    prompt: `You are an adversarial reviewer. Try to REFUTE: "the lifted degree-6 Lyapunov certifier (verifier_deg6.py certify_degN, degree=3) is a SOUND certificate of contraction of the nonlinear coupled-RWKV map." Read ${DIR}/verifier_deg6.py and ${DIR}/verifier_deg4.py. The lifted LMI imposes decrease only at the t-box VERTICES; the nonlinear map's mean Jacobian is a CONVEX-HULL point, and for degree>=2 the symmetric power A^[k] is non-convex in A, so vertex-feasibility does NOT imply hull-decrease of THIS V. Check whether soundness genuinely follows via the JSR convex-hull theorem (JSR{vertices}=JSR{conv}). Then read ${DIR}/verify_certifier_jsr_soundness_results.json: do ALL certified genes have JSR_lb<1? If any certified gene has JSR_lb>=1, the certifier is UNSOUND. Also check the independent P-eigenvalue re-check and the rho<1 vertex pre-screen. Run python if useful (py -3.11). Return your structured verdict.`,
  },
  {
    key: 'coverage-complementarity',
    prompt: `Adversarial reviewer. Try to REFUTE: "the verifier coverage ladder strictly advances (inf 88 -> ... -> deg6 247 of 300 contracting) AND degree-4/6 lifted certificates are COMPLEMENTARY (non-nested): deg4\\deg6=23 and deg6\\deg4=13 both > 0." Read ${DIR}/exp_deg6_ladder.py and ${DIR}/exp_deg6_ladder_results.json. Key suspicion: is the complementarity a SOLVER-INSTABILITY artifact? The cvxpy solve emits "Solution may be inaccurate"; borderline residual genes near the feasibility boundary could flip deg4<->deg6 with solver tolerance/margin, faking non-nesting. Check ${DIR}/redteam_deg6_results.json lens4 (margin 1e-6..1e-8 robustness) if present. Consider re-running a few complementarity genes at different margins (py -3.11). Decide if non-nesting is real or a tolerance artifact. Return structured verdict.`,
  },
  {
    key: 'jsr-attribution',
    prompt: `Adversarial reviewer. Try to REFUTE: "of the deg6-residual (genes rho<1 but uncertified even by deg6), ~89% are genuine finite-degree gaps (JSR_lb<1) and only ~11% are correct rejections (JSR_lb>=1)." Read ${DIR}/jsr_bracket.py, ${DIR}/jsr_bracket_results.json, and the refreshed ${DIR}/verify_certifier_jsr_soundness_results.json (uncertified_refresh at longer max_len). Key suspicion: JSR_lb is a LOWER bound truncated at finite product length; longer products could push more genes to JSR>=1, shrinking the "finite-gap" fraction (the 89% is an UPPER bound on finite-gap). Check how the fraction moves from max_len=5 to max_len=6. Also: is JSR_lb<1 evidence of a finite-degree gap, or just inconclusive (true JSR could be in (JSR_lb,1])? Assess honesty of the framing. Return structured verdict.`,
  },
  {
    key: 'capability-payoff',
    prompt: `Adversarial reviewer. Try to REFUTE the capability-frontier conclusion. Read ${DIR}/verify_deg4_payoff_results.json (region-constrained optimization + winner attribution + 50k soundness), ${DIR}/verify_region_ceiling_results.json (random-sample ceilings), and ${DIR}/exp_deg6_capability.py. Context: a 2-seed smoke showed L3=sdp+deg4 reaching rotation 0.9765 > L2=sdp 0.8881 (suspiciously good); the region-ceiling (random sample) showed deg4 ceiling 0.605 < sdp 0.72 (possibly density-biased / under-sampled). Determine the TRUE answer: is there a SOUND deg4 capability payoff over sdp on rotation at n=2, or is it NULL? Check the L3 winner's region attribution and 50k soundness in verify_deg4_payoff_results.json. Flag if EITHER the "too good" smoke OR the "too low" random ceiling biased the conclusion. Return structured verdict on whether the verdict's capability claim is correct.`,
  },
  {
    key: 'dimension-threshold',
    prompt: `Adversarial reviewer. Try to REFUTE: "the capability gap (residual transient amplification minus quad-certified) jumps from ~+0.4 at n=2 to ~+2.0-2.3 at n>=3 (a dimension threshold), so higher-degree verifiers become load-bearing as the substrate scales." Read ${DIR}/exp_deg6_dimension.py and ${DIR}/exp_deg6_dimension_results.json. Key suspicions: (1) n=4 found only ~6 quad-certified contracting genes (severe under-power) — is the gap estimate meaningless there? (2) the behavioral-contraction proxy (median final ratio < 0.6) and transient_amp (max over 10 random directions, ||s0||=0.25) — are these sound, or could they count expansive/saturating genes as "residual"? (3) is comparing T_residual (huge region) vs T_quad (tiny region) confounded by sample size like the region-ceiling was? Consider re-running n=4 with a larger budget (py -3.11 exp_deg6_dimension.py is parameterized in run()). Return structured verdict on whether the dimension-threshold claim is supported or over-stated.`,
  },
  {
    key: 'methodology-honesty',
    prompt: `Adversarial reviewer / honesty auditor. Read ${DIR}/DEG6_PREREGISTRATION.md and the DRAFT ${DIR}/DEG6_VERDICT.md (if present). Check: (1) does the verdict adhere to the pre-registered gates and honest-null commitments, or did any gate get redefined post-hoc to pass? (2) are any numbers reported as "ceilings"/"payoffs" actually confounded by region size or GA noise (the project's repeated trap)? (3) is the headline framing ("two frontiers of certificate strength, decoupling above SDP") supported by the data or an over-narrative? (4) are all honest caveats (n=4 under-power, JSR truncation, random-sample bias, solver inaccuracy warnings) disclosed? Return a structured verdict on the verdict's overall honesty and any overclaim to fix.`,
  },
]

phase('Refute')
const reviews = await parallel(claims.map(c => () =>
  agent(c.prompt, { label: `refute:${c.key}`, phase: 'Refute', schema: VERDICT_SCHEMA })
    .then(v => ({ ...v, key: c.key }))
))
const valid = reviews.filter(Boolean)

phase('Confirm')
const flagged = valid.filter(r => r.refuted || ['medium', 'high', 'blocker'].includes(r.severity))
const confirmations = await parallel(flagged.map(r => () =>
  agent(`A peer reviewer flagged a possible problem in the llcore deg6 study.\nClaim: "${r.claim}"\nReported confound (severity ${r.severity}): ${r.analysis}\nIndependently CONFIRM OR DISMISS this. Read the relevant files under ${DIR} and re-run a minimal check (py -3.11) if it decides the matter. Is the confound real and does it change a verdict conclusion?`,
    { label: `confirm:${r.key}`, phase: 'Confirm', schema: VERDICT_SCHEMA })
    .then(v => ({ ...v, key: r.key }))
))

phase('Synthesize')
const critic = await agent(
  `You are the completeness critic for the llcore degree-6 verifier-ladder study. Here are the adversarial reviews:\n${JSON.stringify(valid, null, 2)}\nand the confirmations:\n${JSON.stringify(confirmations.filter(Boolean), null, 2)}\nProduce a consolidated assessment: (1) which headline claims SURVIVE adversarial review and which need correction, (2) what is still UNVERIFIED (a modality not run, a claim unchecked), (3) the single most important fix to the verdict before it can be trusted. Be concrete and cite the findings.`,
  { label: 'completeness-critic', phase: 'Synthesize' })

return { reviews: valid, confirmations: confirmations.filter(Boolean), critic }
