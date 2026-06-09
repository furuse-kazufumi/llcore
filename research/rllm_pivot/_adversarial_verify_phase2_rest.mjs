export const meta = {
  name: 'phase2-rest-adversarial-verify',
  description: 'Adversarially verify the 3 Phase-2-remaining experiments (real-CE capability / F8 framework-ness / Mamba Lyapunov) against their code+data; try to refute each verdict.',
  phases: [
    { title: 'Refute', detail: 'one skeptic per experiment re-derives claims from .py+results.json and tries to refute' },
    { title: 'Synthesize', detail: 'consolidate MAJOR/MINOR findings across the three' },
  ],
};

const DIR = 'D:/projects/llcore/research/rllm_pivot';

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['experiment', 'verdict_holds', 'major_findings', 'minor_findings', 'recomputed_checks', 'summary'],
  properties: {
    experiment: { type: 'string' },
    verdict_holds: { type: 'boolean', description: 'true if the experiment\'s stated verdict survives adversarial scrutiny (no MAJOR refutation)' },
    major_findings: { type: 'array', items: { type: 'string' }, description: 'MAJOR issues that would overturn the verdict (numerical mismatch, unsound certifier, fabricated data, leakage that flips the conclusion). Empty if none.' },
    minor_findings: { type: 'array', items: { type: 'string' }, description: 'MINOR issues (framing, missing caveat, rounding) that refine but do not overturn.' },
    recomputed_checks: { type: 'array', items: { type: 'string' }, description: 'concrete numbers you independently recomputed and whether they matched the results.json' },
    summary: { type: 'string' },
  },
};

const targets = [
  {
    key: 'A_realce',
    label: 'real-SmolLM2-CE capability',
    py: 'phase2_capability_realce.py',
    json: 'phase2_capability_realce_results.json',
    log: 'phase2_capability_realce.log',
    claim: 'capability on a REAL SmolLM2-hidden-derived next-cluster CE terrain. KEY honest-disclosure result: MAP-Elites beats the FINITE-DIFFERENCE gradient ~20/20 (a tempting false-positive), BUT a STRONG ANALYTIC gradient (torch Adam, exact gradients via autograd, same budget = forward-CE-eval count) BEATS MAP-Elites => the ME-over-finite-diff win is an ARTIFACT of the weak finite-diff baseline (cold-start, dim+1 evals/step, ~95 steps). With a strong gradient, gradient >= evolution => capability NEGATIVE on the real terrain too (consistent with M3 + synthetic). Verdict label = ARTIFACT+NEGATIVE. honest_eval = diff>0 ∧ one-sided Wilcoxon p<0.05 ∧ n_seeds>=15 ∧ |paired_sign_delta|>=0.147.',
    refute_focus: 'This is the make-or-break experiment; scrutinize hard. (1) Is the terrain genuinely REAL-LLM-derived (real SmolLM2 hidden states, not synthetic)? Read get_real_hidden_sentences. (2) train/held-out LEAKAGE: clusters fit on train sentences only? held-out sentences unseen during optimization? PCA/projection fit on train only? (3) ★Is the torch analytic-gradient baseline FAIR and genuinely STRONG: does it optimize the SAME CE the numpy terrain.train evaluates (re-derive: does opt_gradient_torch use the same centroids/beta/Xs/targets as terrain._ce_sentence)? Is the budget counting honest (1 Adam step = 1 forward CE eval; backward free = realistic backprop)? Is the param box (sigmoid/2tanh) equivalent to numpy clip box? (4) ★INDEPENDENTLY RE-RUN: load the module, build terrain for 2-3 seeds, run opt_mapelites vs opt_gradient_torch vs opt_gradient, and CONFIRM that analytic torch gradient beats ME on held-out (the ARTIFACT conclusion). If torch does NOT beat ME, the verdict is wrong (MAJOR). (5) Recompute ME_vs_gradient and ME_vs_gradient_torch mean_diff/sign_delta/Wilcoxon p from heldout_raw; confirm they match results.json and the verdict logic. (6) Is the centroid-readout beta (between-cluster scale) result-independent (not p-hacked)? (7) Is the honest framing correct — does the verdict AVOID claiming evolution has capability, and correctly attribute the finite-diff ME-win to weak-baseline artifact? (8) Caveat completeness: is it disclosed that the synthetic phase2 NULL_TIE used the SAME weak finite-diff gradient (so the negative may be understated there too)?',
  },
  {
    key: 'B_f8',
    label: 'F8 framework-ness',
    py: 'phase2_framework_f8.py',
    json: 'phase2_framework_f8_results.json',
    log: null,
    test: 'test_framework_f8.py',
    claim: 'Part (b) 3-plug-point swap = PASS (GeneCodec / Objective / VerifierBackend each swap as a SINGLE object into the UNCHANGED src llcore.evolution.minimal_ga.evolve). Part (a) generalization load-bearing = NULL (structural-diversity MAP-Elites archive does NOT beat param-shift baseline on held-out task family). pytest 17 passed.',
    refute_focus: 'Try to refute: (1) Is src minimal_ga.evolve REALLY unchanged (git diff src/)? Is the SrcCodecAdapter honestly boilerplate or does it smuggle the real work? (2) Is the verifier-gate a fitness-wrapper PROXY (disclosed) — does that proxy invalidate the swap claim? Are the final admit counts computed with the REAL verifier.certifies? Check admit ladder none>=sdp>={inf,two} and sdp⊇two (cert_sdp fast-path). (3) Part(a): is the budget equal (POP*GENS=720) for diverse vs paramshift? Is best selected by TRAIN then evaluated on HELD-OUT (no leakage)? Recompute diverse_vs_paramshift_heldout mean_diff/p and confirm NULL. (4) Is the generalization terrain discriminating? (5) Run the pytest yourself: `py -3.11 -m pytest D:/projects/llcore/research/rllm_pivot/test_framework_f8.py -q` and confirm green.',
  },
  {
    key: 'C_mamba',
    label: 'Mamba SSM Lyapunov positive-control',
    py: 'phase2_mamba_lyapunov.py',
    json: 'phase2_mamba_lyapunov_results.json',
    log: null,
    claim: 'Mamba-130M is stable-by-construction: every layer SSM has continuous diagonal A=-exp(A_log)<0 (100% of 589824 channel,state), so discrete A-bar=exp(dt*A) has |diag|<=1 and lambda_max=max(dt*A)<=0 for all dt>0 => trivial PASS. SmolLM2 (llama) has NO SSM A matrix => stability must be imposed by adapter+gate. base-level discrimination = PASS. HONEST: marginal channels at base dt arise from dt~0 not A~0; non-strict (<=0) at base dt but strict for any dt>0.',
    refute_focus: 'Try to refute: (1) Is A=-exp(A_log) the CORRECT Mamba parameterization (matches transformers MambaMixer: A=-exp(A_log.float()))? Verify by reading the actual state_dict key A_log and recomputing A on one layer. (2) Is the claim over-stated — is it STRICT contraction or only NON-POSITIVE (<=0)? Confirm the honest marginal-note (dt~0 channels) is accurate and not hiding A~0. (3) Re-load Mamba-130M and independently recompute global_lambda_max_base = max over all layers of max(dt_base*A) and confirm it matches (-1.8e-9). (4) Is SmolLM2 genuinely free of SSM keys (re-check state_dict)? (5) Is the diagonal-A assumption valid for this hf checkpoint (A_log shape (d_inner,state_size))? (6) Scope honesty: is it clearly stated this is SSM-state-recurrence stability only, NOT full-Mamba Lipschitz?',
  },
];

phase('Refute');
const verdicts = await parallel(targets.map((t) => () => {
  const logLine = t.log ? `- log: ${DIR}/${t.log}` : '';
  const testLine = t.test ? `- pytest: ${DIR}/${t.test} (run it: py -3.11 -m pytest ${DIR}/${t.test} -q)` : '';
  return agent(
    `You are an ADVERSARIAL verifier in the llcore "evolvable LLM" research (Verified-Plasticity Evaluation Framework). Discipline: honest-disclosure. Your job is to TRY TO REFUTE the stated verdict of one experiment by checking it against its actual code and data, and by INDEPENDENTLY RE-RUNNING small checks (re-load data, recompute key numbers). Default to skepticism: prefer "verdict_holds=false" only if you find a MAJOR issue (numerical mismatch vs results.json, unsound certifier, fabricated/leaked data that flips the conclusion). Framing/caveat gaps are MINOR.

Experiment: ${t.label}
Files:
- script: ${DIR}/${t.py}
- results: ${DIR}/${t.json}
${logLine}
${testLine}

Stated claim/verdict to scrutinize:
${t.claim}

Refutation focus (work through each):
${t.refute_focus}

Method:
- Read the .py and results.json. Use \`py -3.11 -c "..."\` to INDEPENDENTLY recompute at least 2 key numbers from the raw arrays in results.json (e.g. mean_diff, sign_delta, Wilcoxon p, or re-load a model layer and recompute a statistic) and state whether they MATCH the json.
- Use \`git -C D:/projects/llcore diff --stat src/\` to confirm src/ is unchanged where the claim says so.
- Do NOT modify any files. Do NOT git commit. Read-only verification + ephemeral \`py -3.11 -c\` recomputation only.
- Python = py -3.11. Models (SmolLM2-135M, Mamba-130M) are in HF cache (offline ok).

Return the structured verdict: list MAJOR findings (overturn) and MINOR findings (refine), the concrete numbers you recomputed and whether they matched, and whether the verdict holds.`,
    { label: `refute:${t.key}`, phase: 'Refute', schema: VERDICT_SCHEMA }
  );
}));

phase('Synthesize');
const clean = verdicts.filter(Boolean);
const majors = clean.flatMap((v) => (v.major_findings || []).map((m) => `[${v.experiment}] ${m}`));
const minors = clean.flatMap((v) => (v.minor_findings || []).map((m) => `[${v.experiment}] ${m}`));
const allHold = clean.every((v) => v.verdict_holds);

return {
  n_verified: clean.length,
  all_verdicts_hold: allHold,
  any_major: majors.length > 0,
  major_findings: majors,
  minor_findings: minors,
  per_experiment: clean.map((v) => ({
    experiment: v.experiment,
    verdict_holds: v.verdict_holds,
    n_major: (v.major_findings || []).length,
    n_minor: (v.minor_findings || []).length,
    recomputed_checks: v.recomputed_checks,
    summary: v.summary,
  })),
};
