export const meta = {
  name: 'audit-lock-review',
  description: 'Final adversarial verification before locking the SCS→CLARABEL audit: reproduce the CLARABEL numbers, check the solver swap introduced NO new false positives (soundness), confirm the audit doc matches the result JSONs, and a completeness critic.',
  phases: [
    { title: 'Verify', detail: 'reproduce numbers, check CLARABEL soundness, cross-check audit doc' },
    { title: 'Lock', detail: 'completeness critic — anything uncorrected or a new artifact?' },
  ],
}

const RES = 'D:/projects/llcore/research'
const DIR = RES + '/verified_evolution_sdp_gate'

const V = {
  type: 'object',
  required: ['claim', 'holds', 'severity', 'analysis', 'recommendation'],
  properties: {
    claim: { type: 'string' },
    holds: { type: 'boolean', description: 'true if the audited claim is correct/reproducible' },
    severity: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'blocker'] },
    analysis: { type: 'string', description: 'concrete numbers/commands; what reproduced or failed' },
    recommendation: { type: 'string' },
  },
}

const lenses = [
  {
    key: 'clarabel-soundness',
    prompt: `Adversarial reviewer. The audit SWAPPED cvxpy's solver from SCS to CLARABEL to fix false NEGATIVES. The risk of swapping a solver is the opposite error: NEW FALSE POSITIVES (certifying a gene that is actually expansive). Verify CLARABEL introduced NO false positives. Read ${DIR}/verifier_deg6.py, ${DIR}/coupled_components.py (_sdp_certifies), ${RES}/spectral_lyapunov_contraction/exp_d_runner.py and its ${RES}/spectral_lyapunov_contraction/exp_d_results.json (D1_soundness: n_false_admits must be 0, worst pnorm-gain <=1). Run python (py -3.11): take genes CLARABEL now certifies that SCS rejected, and check each has empirical_rho<1 AND jsr_lb<1 (the JSR product oracle in ${DIR}/verifier_jsr.py / verify_certifier_jsr_soundness.py). If ANY CLARABEL-certified gene is switched-expansive (jsr_lb>=1) the audit is unsound. Return verdict.`,
  },
  {
    key: 'numbers-match',
    prompt: `Adversarial auditor. Verify the consolidated audit doc ${RES}/AUDIT_SCS_CLARABEL_2026-06-03.md matches the actual result JSONs. Check every CLARABEL number in its table against: ${DIR}/exp_deg6_ladder_results.json (sdp 286/300, deg-union +4, complementarity 0/1, residual 10), ${DIR}/exp_deg4_results.json (D4 residual 31, recovered 4 = 12.9%), ${RES}/spectral_lyapunov_contraction/exp_d_results.json (sdp_tmin1_admit 1291, D3 sdp-beats-two 692, two_beats_sdp 0, D4 residual 72), and ${DIR}/exp1_results.json (Track B exp1, just produced — report its sdp admit count and region ceilings and whether the audit's 'expected sdp↑/residual↓' holds). Flag any number in the audit doc that does NOT match its JSON. Return verdict.`,
  },
  {
    key: 'corrections-complete',
    prompt: `Adversarial auditor. The audit claims it corrected the prior verdicts. Verify the correction banners are present and accurate: ${DIR}/DEG6_VERDICT.md (complementarity retracted, residual 53→10, dimension retracted), ${DIR}/DEG4_VERDICT.md (banner: 172/57/33% → 31/4/13%), ${DIR}/VERDICT.md (banner: SDP understated, +254→+692, residual ~3-5%). Also: are there OTHER SDP-using files still on the SCS default that were missed? grep the tree (py -3.11 or rg) for cvxpy '.solve()' calls WITHOUT a solver argument under ${RES} and ${DIR}/../.. /src. List any unpinned SDP solve. Return verdict on completeness.`,
  },
]

phase('Verify')
const reviews = (await parallel(lenses.map(l => () =>
  agent(l.prompt, { label: `verify:${l.key}`, phase: 'Verify', schema: V }).then(v => ({ ...v, key: l.key }))
))).filter(Boolean)

phase('Lock')
const critic = await agent(
  `Completeness critic for the SCS→CLARABEL audit lock. The verification reviews:\n${JSON.stringify(reviews, null, 2)}\nDecide: (1) is the audit SAFE TO LOCK (CLARABEL sound = no new false positives, audit numbers match JSONs, corrections complete)? (2) any blocker that must be fixed before locking? (3) the one most important residual risk. Be concrete; cite the reviews.`,
  { label: 'lock-critic', phase: 'Lock' })

return { reviews, critic, safe_to_lock: reviews.every(r => r.holds || r.severity === 'none' || r.severity === 'low') }
