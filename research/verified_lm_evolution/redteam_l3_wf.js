export const meta = {
  name: 'rllm-l3-redteam',
  description: 'Adversarial red-team of the R-LLM-0 L3 PAYOFF result (a stronger sound contraction verifier unlocks lower real byte-LM perplexity): is the positive headline real or an artifact?',
  phases: [
    { title: 'Evidence', detail: 'one agent reads the landscape result JSON + gated log + code, returns a structured digest' },
    { title: 'Skeptics', detail: 'parallel anti-thesis lenses, each trying to FALSIFY the payoff' },
    { title: 'Synthesis', detail: 'aggregate verdicts into a survives/falsified/narrowed call' },
  ],
}

const DIR = './research/verified_lm_evolution'
const SDP = './research/verified_evolution_sdp_gate'

// The POSITIVE headline under attack (the result REVERSED from the underpowered 40/60-gene smoke).
const HEADLINE = (args && args.headline) ||
  'L3 PAYOFF: on a real tiny byte-LM, a less-conservative BUT STILL SOUND contraction verifier unlocks ' +
  'strictly lower held-out perplexity than the conservative inf-norm gate. In the 900-gene landscape the ' +
  'best CONTRACTING held-out CE falls monotonically as the certifier relaxes: inf 4.8377 > two_norm_only ' +
  '4.7954 > sdp_only 4.7525 (all 0% empirically-expansive = sound), while non_certified reaches 4.7052 but ' +
  'is 78.9% expansive (correctly rejected). Crucially the inf region has MORE genes (346) than sdp_only (189) ' +
  'yet a WORSE ceiling, so the ordering is a genuine region-ceiling difference, not a sampling/lottery artifact. ' +
  'Gated evolution (in progress) concurs: inf-gated runs sit at ~unigram fitness (0.0285) while two/sdp/none ' +
  'reach higher fitness. Therefore the arc signature claim holds on REAL LM loss, not just synthetic rotation.'

// Result files (landscape JSON exists; gated JSON not yet — read the live log for per-seed fitness).
const READ_FILES = [
  `${DIR}/exp_landscape_results.json`,
  `${DIR}/l3_landscape.log`,
  `${DIR}/l3_gated_real.log`,
  `${DIR}/l3_gated_null.log`,
  `${DIR}/exp_landscape.py`,
  `${DIR}/exp_gated.py`,
  `${DIR}/lm_substrate.py`,
  `${DIR}/PREREGISTRATION.md`,
  `${SDP}/coupled_nd.py`,
]

const DIGEST_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['l0_beats_unigram', 'landscape_ladder', 'region_sizes', 'gated_seeds', 'gated_null_status', 'soundness_L1L2', 'notes'],
  properties: {
    l0_beats_unigram: { type: 'boolean' },
    landscape_ladder: { type: 'object', additionalProperties: true, description: 'per-region best_contracting_CE + the monotone deltas + unigram_ce' },
    region_sizes: { type: 'object', additionalProperties: true, description: 'n per region (to assess sampling/lottery)' },
    gated_seeds: { type: 'array', items: { type: 'object', additionalProperties: true }, description: 'per-seed {none,inf,two,sdp} fitness parsed from l3_gated_real.log (may be partial / in progress)' },
    gated_null_status: { type: 'string', description: 'has the null control run produced data yet? what does it show?' },
    soundness_L1L2: { type: 'object', additionalProperties: true, description: 'pct_expansive per region (admitted regions should be 0, non_certified > 0)' },
    notes: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'headline_survives', 'confidence', 'reasoning', 'forces_narrowing', 'what_would_overturn'],
  properties: {
    lens: { type: 'string' },
    headline_survives: { type: 'boolean' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    reasoning: { type: 'string', description: 'concrete, cite the numbers/code' },
    forces_narrowing: { type: 'string', description: 'if the payoff survives only in a narrower form, state the exact narrower claim; else ""' },
    what_would_overturn: { type: 'string' },
  },
}

phase('Evidence')
const digest = await agent(
  `You are a meticulous research auditor. Read these files (use the Read tool):\n` +
  READ_FILES.map((f) => `- ${f}`).join('\n') +
  `\n\nContext: an experiment wires the "CoupledNDGene" verified-evolution recurrent core into a real tiny ` +
  `byte-level LM and asks whether a STRONGER (but still sound) contraction verifier unlocks lower held-out ` +
  `perplexity. The 900-gene landscape is complete (exp_landscape_results.json + l3_landscape.log). The gated ` +
  `evolution run is IN PROGRESS — parse per-seed fitness from l3_gated_real.log (lines like ` +
  `"seed 2: none=0.0304 inf_norm=0.0285 two_norm=0.0293 sdp=0.0300"); l3_gated_null.log may be empty (null ` +
  `control not started yet). Extract the actual numbers into the schema faithfully. Do NOT interpret yet. ` +
  `Note explicitly in notes how many gated seeds exist and whether the null control has data.`,
  { phase: 'Evidence', label: 'evidence-digest', schema: DIGEST_SCHEMA }
)

const DIGEST_JSON = JSON.stringify(digest, null, 2)

const LENSES = [
  {
    key: 'L0-sanity',
    prompt: `L0 substrate sanity: the tiny reservoir LM must beat the unigram baseline (else the frontier ` +
      `question is moot). Confirm best CE < unigram_CE. headline_survives=false ONLY if L0 fails.`,
  },
  {
    key: 'sampling-lottery',
    prompt: `Sampling/lottery confound — the CENTRAL threat to a "region ceiling" claim. The payoff says ` +
      `sdp_only (best CE 4.7525) beats inf (4.8377). But best-CE is an EXTREME-VALUE statistic: a region with ` +
      `more genes gets more lottery tickets at a low CE. CHECK region sizes: the claim is robust ONLY if the ` +
      `WINNING (lower-CE) regions are NOT simply the bigger ones. Here inf has 346 genes vs sdp_only 189 — inf ` +
      `is BIGGER yet WORSE, which argues the ordering is a true ceiling, not sampling. Verify this from the ` +
      `digest. Also check medians (does the ORDERING hold on median, or only on best-of-region?). If the ` +
      `best-CE ladder is driven by sample size, headline_survives=false. If bigger-region-worse-ceiling holds, ` +
      `survives=true. State whether the claim must be narrowed to "best-of-region ceiling" vs "typical gene".`,
  },
  {
    key: 'inf-gated-broken-vs-ceiling',
    prompt: `Mechanism ambiguity: gated inf runs sit at ~unigram (0.0285). Two readings: (A) inf region has a ` +
      `genuinely HIGHER (worse) ceiling [supports payoff], or (B) the inf GATE is so restrictive that evolution ` +
      `cannot navigate/optimize within it, so it's an evolvability artifact, not a ceiling fact. The landscape ` +
      `is the discriminator: it RANDOM-SAMPLES the inf region (not gated evolution), so if inf's landscape best ` +
      `(4.8377) is independently worse than sdp_only's (4.7525), the higher ceiling is real and NOT a gated- ` +
      `navigation artifact. Confirm the landscape (random-sampled) ordering matches the gated ordering. If they ` +
      `agree, survives=true (ceiling is real). If the landscape did NOT show the ladder, survives=false.`,
  },
  {
    key: 'monotone-by-chance',
    prompt: `Is the clean monotone ladder (inf>two>sdp>non_cert) statistically real or a lucky ordering? The ` +
      `deltas are small (~0.04-0.09 nats). With 4 regions there are 24 orderings; one monotone ladder could be ` +
      `chance. Look for corroboration: (a) does the gated run reproduce the SAME ordering across seeds ` +
      `(inf lowest fitness every seed)? (b) is the ladder monotone on the gated paired deltas too? Cite the ` +
      `gated seed consistency. If the ordering is reproduced independently by gated paired seeds, survives=true; ` +
      `if it rests only on the single landscape best-of-region, narrow the claim and lower confidence.`,
  },
  {
    key: 'circularity',
    prompt: `Circularity: classify_region uses the SAME cert_inf/two/sdp that define the gates. Is "best CE per ` +
      `region" secretly a re-encoding of the certifier rather than independent fitness? Held-out CE comes from ` +
      `the reservoir readout, computed independently of the certifier (which reads only decay,W). Argue whether ` +
      `the frontier read is non-circular. survives=true if non-circular.`,
  },
  {
    key: 'readout-confound',
    prompt: `Readout confound: each gene gets its OWN ridge/logistic readout fit. Could the regional CE ` +
      `differences reflect readout CAPACITY/optimization differences rather than the recurrent core's reachable ` +
      `dynamics? The readout (fit_logistic_readout) is the SAME procedure (same n_steps, l2, lr) for every gene/ ` +
      `region, and the embedding is fixed+shared. Argue whether the payoff can be explained by readout rather ` +
      `than the verified core. survives=true if the readout is controlled (same for all regions).`,
  },
  {
    key: 'soundness-L1L2',
    prompt: `Soundness consistency: admitted regions (inf/two/sdp) must be 0% empirically-expansive (L1), and ` +
      `non_certified must contain expansive genes (L2, non-vacuous oracle). Check pct_expansive per region. The ` +
      `payoff story REQUIRES that sdp_only (the sound win) is 0% expansive while non_certified (lower CE but ` +
      `unsound) is high. If an admitted region shows expansive genes, that is a soundness break — surface it. ` +
      `survives=true if L1/L2 hold and the sound win (sdp_only) is genuinely contracting.`,
  },
  {
    key: 'regime-scope-overclaim',
    prompt: `Over-generalization: the correct honest claim is NARROW — "on this byte-LM corpus, n=8 reservoir ` +
      `substrate, fixed embedding, the SOUND sdp/two regions reach ~0.04-0.09 nats lower held-out CE than inf". ` +
      `Does the headline over-claim (e.g. "verifier payoff is large/general for all LMs", or conflate the ` +
      `unsound non_certified win with the sound one)? The honest payoff is the SOUND sdp-vs-inf gap (~0.085 ` +
      `nats), modest but real and on real LM loss. survives=true only if scoped to the narrow, sound claim; ` +
      `state the exact wording it must be narrowed to.`,
  },
]

phase('Skeptics')
const verdicts = await parallel(
  LENSES.map((L) => () =>
    agent(
      `You are an adversarial skeptic trying to FALSIFY this POSITIVE headline:\n\n"${HEADLINE}"\n\n` +
      `Faithful evidence digest:\n\`\`\`json\n${DIGEST_JSON}\n\`\`\`\n\nYou may Read any of these for detail:\n` +
      READ_FILES.map((f) => `- ${f}`).join('\n') +
      `\n\n${L.prompt}\n\nThis is a POSITIVE result, so apply MAXIMUM scrutiny (honest-disclosure: a surprising ` +
      `win must be doubted before celebrated). Be concrete, cite numbers. If the evidence is ambiguous, lean ` +
      `headline_survives=false and say what is missing. Fill the schema for lens="${L.key}".`,
      { phase: 'Skeptics', label: `lens:${L.key}`, schema: VERDICT_SCHEMA }
    ).then((v) => ({ ...v, lens: L.key }))
  )
)

const clean = verdicts.filter(Boolean)
const survived = clean.filter((v) => v.headline_survives).length
const failed = clean.filter((v) => !v.headline_survives)

phase('Synthesis')
const synthesis = await agent(
  `You are the lead reviewer. POSITIVE headline under test:\n\n"${HEADLINE}"\n\n` +
  `Evidence digest:\n\`\`\`json\n${DIGEST_JSON}\n\`\`\`\n\n${clean.length} adversarial lenses ran. ${survived} ` +
  `say it survives, ${failed.length} dissent.\nPer-lens verdicts:\n\`\`\`json\n${JSON.stringify(clean, null, 2)}\n\`\`\`\n\n` +
  `Write a tight markdown synthesis for a VERDICT.md "Red-team" section: (1) does the PAYOFF survive adversarial ` +
  `review? (2) the exact narrowed/corrected wording the headline must take (the honest claim is the SOUND ` +
  `sdp-vs-inf gap ~0.085 nats on this n=8 byte-LM, NOT a general/large claim). (3) any soundness (L1/L2) issue. ` +
  `(4) the single most important remaining measurement (e.g. completing the gated paired run + null control) ` +
  `that would strengthen or overturn it. Be honest-disclosure first; do not let a positive result inflate the ` +
  `claim beyond what the numbers support.`,
  { phase: 'Synthesis', label: 'synthesis' }
)

return { headline: HEADLINE, survived, total: clean.length, dissenting_lenses: failed.map((v) => v.lens), digest, verdicts: clean, synthesis }
