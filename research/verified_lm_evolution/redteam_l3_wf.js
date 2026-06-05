export const meta = {
  name: 'rllm-l3-redteam',
  description: 'Adversarial red-team of the R-LLM-0 L3 result (verifier payoff on a real byte-LM): is the headline real or an artifact?',
  phases: [
    { title: 'Evidence', detail: 'one agent reads result JSONs + code, returns a structured digest' },
    { title: 'Skeptics', detail: 'parallel anti-thesis lenses (PREREG §5 A-F + anti-NULL)' },
    { title: 'Synthesis', detail: 'aggregate verdicts into a survives/falsified call' },
  ],
}

// args = { dir, headline } where headline is the human-stated claim to attack
// (e.g. "L3 NULL: a better/less-conservative sound verifier does NOT lower held-out
//  byte-LM perplexity; the conservative inf-norm gate already contains the optimum").
const DIR = (args && args.dir) || 'D:/projects/llcore/research/verified_lm_evolution'
const HEADLINE = (args && args.headline) ||
  'L3 NULL: on a real tiny byte-LM, a less-conservative sound contraction verifier (two_norm/sdp) does NOT unlock lower held-out perplexity than the conservative inf-norm gate; the optimal recurrence is near-diagonal so the verifier-fitness frontier is synthetic-rotation-specific.'

const READ_FILES = [
  `${DIR}/exp_landscape_results.json`,
  `${DIR}/exp_gated_results.json`,
  `${DIR}/exp_gated_null_results.json`,
  `${DIR}/exp_landscape.py`,
  `${DIR}/exp_gated.py`,
  `${DIR}/lm_substrate.py`,
  `${DIR}/PREREGISTRATION.md`,
  `${DIR}/../verified_evolution_sdp_gate/coupled_nd.py`,
]

const DIGEST_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['l0_beats_unigram', 'landscape', 'gated_real', 'gated_null', 'notes'],
  properties: {
    l0_beats_unigram: { type: 'boolean', description: 'Does the best gene beat the unigram CE (LM actually functions)?' },
    landscape: {
      type: 'object', additionalProperties: true,
      description: 'per-region {n, best_ce, best_ce_contracting, median_ce, pct_expansive} and unigram_ce',
    },
    gated_real: {
      type: 'object', additionalProperties: true,
      description: 'per-gate mean_fit + the two_norm_vs_inf / sdp_vs_inf paired {mean_delta, frac_a_gt_b, deltas}, winner_regions',
    },
    gated_null: {
      type: 'object', additionalProperties: true,
      description: 'same shape as gated_real but on shuffled corpus (null control)',
    },
    notes: { type: 'string', description: 'anything anomalous in the raw numbers or code worth flagging' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'headline_survives', 'confidence', 'reasoning', 'what_would_overturn'],
  properties: {
    lens: { type: 'string' },
    headline_survives: { type: 'boolean', description: 'true if the stated headline still stands under this lens' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    reasoning: { type: 'string', description: 'concrete, cite the numbers/code; no hand-waving' },
    what_would_overturn: { type: 'string', description: 'the specific measurement/condition that would flip this lens' },
  },
}

phase('Evidence')
const digest = await agent(
  `You are a meticulous research auditor. Read these files (use the Read tool):\n` +
  READ_FILES.map((f) => `- ${f}`).join('\n') +
  `\n\nThey are the pre-registration, code, and result JSONs of an experiment that wires the ` +
  `"CoupledNDGene" verified-evolution recurrent core into a real tiny byte-level language model and ` +
  `asks whether a STRONGER (but still sound) contraction verifier unlocks lower held-out perplexity ` +
  `(the "arc" claim), or whether it ties (honest NULL). Extract the actual numbers into the schema. ` +
  `Do NOT interpret yet — just report faithfully what the JSONs and code say. If a result file is ` +
  `missing or malformed, say so explicitly in notes.`,
  { phase: 'Evidence', label: 'evidence-digest', schema: DIGEST_SCHEMA }
)

const DIGEST_JSON = JSON.stringify(digest, null, 2)

const LENSES = [
  {
    key: 'L0-sanity',
    prompt: `Lens: L0 substrate sanity. If the tiny reservoir LM does NOT beat the unigram baseline, ` +
      `the whole L3 question is moot (you cannot study a verifier-fitness frontier on a broken LM). ` +
      `Check best_ce vs unigram_ce in the landscape digest. The headline is only meaningful if L0 holds. ` +
      `Report headline_survives=false ONLY if L0 fails (LM is broken). Otherwise survives=true.`,
  },
  {
    key: 'circularity',
    prompt: `Lens: circularity. classify_region() uses the SAME cert_inf/cert_two/cert_sdp that define the ` +
      `evolution gates. Is the frontier read circular — i.e., is "best CE per region" secretly a re-encoding ` +
      `of the gate decision rather than an independent fitness signal? Held-out CE is computed from the ` +
      `reservoir readout, independent of the certifier. Argue whether the region-attribution is non-circular. ` +
      `headline_survives=true if the NULL/payoff read is non-circular.`,
  },
  {
    key: 'admission-size-pro-null',
    prompt: `Lens: admission-size confound (pro-NULL direction). The NULL reading says "inf region already ` +
      `contains the best gene". But maybe inf region just has FAR more genes (more lottery tickets), so its ` +
      `best_ce is lowest by sample size, not by a genuinely higher ceiling. Compare per-region n and the ` +
      `best_ce_contracting AND median_ce. If inf's MEDIAN is also best (not just its best), the NULL is robust ` +
      `to sample-size. If inf wins only on best_ce while having many more genes, flag the NULL as possibly a ` +
      `sampling artifact. headline_survives=true only if the NULL survives this confound.`,
  },
  {
    key: 'underpowered',
    prompt: `Lens: underpowered NULL. A NULL ("gates tie") is only honest if the effect is genuinely ~0, not if ` +
      `we merely lacked power. Inspect the gated_real paired deltas (two_norm_vs_inf, sdp_vs_inf): are the ` +
      `mean_deltas near zero / sign-mixed (real NULL), or positive-but-nonsignificant (underpowered)? Also check ` +
      `n_seeds. If deltas are tiny and sign-mixed => NULL is real. If consistently positive but small => caution. ` +
      `headline_survives=true if the NULL is a true ~0 effect, not an underpowered miss.`,
  },
  {
    key: 'null-control-validity',
    prompt: `Lens: null-control validity. exp_gated --null shuffles the corpus (destroys sequential structure) so ` +
      `memory is useless and ALL gates should tie at ~unigram. Verify the null run actually behaves as a null ` +
      `(gates tie, fitness ~ unigram_fitness). If the null control itself shows a gate difference, the harness ` +
      `has a confound and BOTH the real and null reads are suspect. headline_survives=true if the null control ` +
      `is clean (validates the harness).`,
  },
  {
    key: 'regime-scope',
    prompt: `Lens: regime scope / over-generalization. The arc payoff was demonstrated on a SYNTHETIC rotation ` +
      `objective where the optimum is non-normal. The honest NULL here is "real LM corpus induces a near-diagonal ` +
      `optimum, so contraction is free". Is the conclusion correctly SCOPED to "this byte-LM corpus / reservoir ` +
      `substrate / n=8", or does the write-up over-claim "verifier payoff never matters for LMs"? The correct ` +
      `honest claim is the narrow one. headline_survives=true if the headline is the narrow, correctly-scoped NULL.`,
  },
  {
    key: 'soundness-L1L2',
    prompt: `Lens: soundness consistency (L1/L2). Independent of the frontier, the safety claim must hold: every ` +
      `gate-admitted gene must be empirically contracting (rho<1, no NaN) on the real corpus (L1), and the ` +
      `ungated pool must genuinely contain expansive genes so the oracle is non-vacuous (L2, pct_expansive>0 in ` +
      `non_certified). Check the landscape rows/summary. If an admitted region shows expansive genes, that is a ` +
      `soundness break and must be surfaced. headline_survives=true if L1/L2 hold (admitted=contracting, ungated=has-expansive).`,
  },
]

phase('Skeptics')
const verdicts = await parallel(
  LENSES.map((L) => () =>
    agent(
      `You are an adversarial skeptic trying to FALSIFY this headline:\n\n"${HEADLINE}"\n\n` +
      `Here is the faithful evidence digest (extracted from the result JSONs):\n\`\`\`json\n${DIGEST_JSON}\n\`\`\`\n\n` +
      `You may also Read any of these files for detail if needed:\n${READ_FILES.map((f) => `- ${f}`).join('\n')}\n\n` +
      `${L.prompt}\n\nBe concrete and cite numbers. Default to skepticism: if the evidence is ambiguous, ` +
      `lean headline_survives=false and explain what is missing. Fill the schema for lens="${L.key}".`,
      { phase: 'Skeptics', label: `lens:${L.key}`, schema: VERDICT_SCHEMA }
    ).then((v) => ({ ...v, lens: L.key }))
  )
)

const clean = verdicts.filter(Boolean)
const survived = clean.filter((v) => v.headline_survives).length
const failed = clean.filter((v) => !v.headline_survives)

phase('Synthesis')
const synthesis = await agent(
  `You are the lead reviewer. The headline under test:\n\n"${HEADLINE}"\n\n` +
  `Evidence digest:\n\`\`\`json\n${DIGEST_JSON}\n\`\`\`\n\n` +
  `${clean.length} adversarial lenses ran. ${survived} say the headline survives, ${failed.length} dissent.\n` +
  `Per-lens verdicts:\n\`\`\`json\n${JSON.stringify(clean, null, 2)}\n\`\`\`\n\n` +
  `Write a tight synthesis (markdown) for a VERDICT.md "Red-team" section: (1) does the headline survive ` +
  `adversarial review? (2) which lenses, if any, force a narrowing or correction of the headline, and exactly ` +
  `how should it be reworded? (3) any soundness (L1/L2) issue that must be disclosed? (4) the single most ` +
  `important remaining measurement that would strengthen or overturn the conclusion. Be honest-disclosure first: ` +
  `if this is a NULL, say plainly whether it is a real ~0 effect or merely underpowered, and scope it narrowly.`,
  { phase: 'Synthesis', label: 'synthesis' }
)

return { headline: HEADLINE, survived, total: clean.length, dissenting_lenses: failed.map((v) => v.lens), digest, verdicts: clean, synthesis }
