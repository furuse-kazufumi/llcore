# SPDX-License-Identifier: Apache-2.0
"""INDEPENDENT red-team soundness check for the contraction + state_norm gates.

Does NOT reuse the implementer's empirical_lipschitz / gate helpers for the core
soundness test. Re-derives everything from the raw RWKV update equation:

    s' = decay*s + (1-decay)*tanh(mix*x + gate_str*s)
    ds'/ds = decay + (1-decay)*gate_str * sech^2(mix*x + gate_str*s)
    sech^2(z) = 1 - tanh(z)^2  in (0,1]

Track A (random clipped genes): for MANY random clipped genes that the certifier
ADMITS (verify_lipschitz_contraction.contraction is True), compute the empirical
state-direction Lipschitz constant by my OWN dense central-difference scan, and
confirm empirical_L <= certified L_upper_bound and (the real soundness gate)
empirical_L < 1. ANY admitted gene with empirical_L >= 1 == FALSE ADMIT == broken.

Also A1 containment: no gene the *free-certifier* (closed-form L_upper_bound<1)
accepts should be rejected by the achievable certifier... but here there is only
ONE certifier; instead I test the inverse containment relevant to soundness:
every gene REJECTED by Z3 must have closed-form L_upper_bound >= 1 (so the reject
set never throws away a closed-form-contractive gene -> certifier == closed form).

state_norm: for random ADMITTED genes, run a long |x|<=1 sequence with MY OWN
integrator (not run_sequence) and confirm |s|<=1 never exceeded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.invariants import (  # noqa: E402
    verify_gene_safe,
    verify_lipschitz_contraction,
    _lipschitz_upper_bound,
)


def my_empirical_L(decay: float, mix: float, gate_str: float,
                   *, n_grid: int = 4001, seed: int = 7) -> float:
    """My OWN empirical state-direction Lipschitz on a DENSE (s,x) grid + jitter.

    Uses analytic-form central differences over a fine grid AND random points,
    independent of the implementer's empirical_lipschitz (different sampling,
    different h, different rng).
    """
    rng = np.random.default_rng(seed)
    # dense deterministic grid over the full achievable box [-1,1]x[-1,1]
    g = np.linspace(-1.0, 1.0, int(np.sqrt(n_grid)) + 1)
    S, X = np.meshgrid(g, g)
    s = np.concatenate([S.ravel(), rng.uniform(-1, 1, n_grid)])
    x = np.concatenate([X.ravel(), rng.uniform(-1, 1, n_grid)])
    h = 1e-7  # different from their 1e-6

    def step(state, inp):
        pre = mix * inp + gate_str * state
        return decay * state + (1.0 - decay) * np.tanh(pre)

    deriv_fd = (step(s + h, x) - step(s - h, x)) / (2.0 * h)
    # also the closed-form analytic derivative for cross-check (no FD error)
    pre = mix * x + gate_str * s
    sech2 = 1.0 - np.tanh(pre) ** 2
    deriv_an = decay + (1.0 - decay) * gate_str * sech2
    return float(max(np.max(np.abs(deriv_fd)), np.max(np.abs(deriv_an))))


def my_state_norm_blowup(decay, mix, gate_str, *, seq_len=2000, seed=99) -> float:
    """My OWN long-sequence |s| max under |x|<=1 (worst-case +/-1 inputs too)."""
    rng = np.random.default_rng(seed)
    # mix random with adversarial all +1 / all -1 / alternating inputs
    seqs = [
        rng.uniform(-1, 1, size=(seq_len, 8)),
        np.ones((seq_len, 8)),
        -np.ones((seq_len, 8)),
        np.tile([1.0, -1.0], (seq_len, 4)),
    ]
    worst = 0.0
    for inputs in seqs:
        state = np.zeros(8)
        for t in range(seq_len):
            pre = mix * inputs[t] + gate_str * state
            state = decay * state + (1.0 - decay) * np.tanh(pre)
            if not np.all(np.isfinite(state)):
                return float("inf")
            worst = max(worst, float(np.max(np.abs(state))))
    return worst


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rng = np.random.default_rng(424242)
    N = 6000

    n_admit_contr = 0
    n_admit_state = 0
    false_admit_contr = 0
    false_admit_state = 0
    contr_reject_with_closedform_lt1 = 0  # would indicate certifier != closed form
    max_emp_over_cert = -1.0  # max ratio empirical_L / cert_bound among admitted
    worst_admit_emp_L = -1.0
    examples_false = []

    # boundary stress: also force genes near L=1 boundary explicitly
    boundary_genes = []
    for d in np.linspace(0.0, 1.0, 21):
        for gs in np.linspace(-2.0, 2.0, 41):
            boundary_genes.append((float(d), 0.0, float(gs)))

    rand_genes = [
        (float(rng.uniform(0, 1)), float(rng.uniform(-1, 1)), float(rng.uniform(-2, 2)))
        for _ in range(N)
    ]
    all_genes = rand_genes + boundary_genes

    for (d, m, gs) in all_genes:
        gene = StateUpdateGene(decay=d, mix=m, gate_str=gs)
        cert = verify_lipschitz_contraction(gene)
        cf = _lipschitz_upper_bound(*gene.clipped().as_array()[[0, 2]])
        # cross check: certifier verdict must equal (closed-form bound < 1)
        if cert.contraction is True and not (cf < 1.0):
            contr_reject_with_closedform_lt1 += 1  # contradiction
        if cert.contraction is False and cf < 1.0:
            contr_reject_with_closedform_lt1 += 1  # rejected a closed-form-contractive gene

        if cert.contraction is True:
            n_admit_contr += 1
            empL = my_empirical_L(d, m, gs)
            worst_admit_emp_L = max(worst_admit_emp_L, empL)
            if cert.L_upper_bound and cert.L_upper_bound > 0:
                max_emp_over_cert = max(max_emp_over_cert, empL / cert.L_upper_bound)
            # SOUNDNESS: an admitted (certified contraction) gene must have empL < 1
            if empL >= 1.0:
                false_admit_contr += 1
                if len(examples_false) < 8:
                    examples_false.append(
                        {"gene": [d, m, gs], "cert_L": cert.L_upper_bound, "empL": empL}
                    )
            # also empirical must be <= certified bound (over-approx soundness)
            if cert.L_upper_bound is not None and empL > cert.L_upper_bound + 1e-6:
                false_admit_contr += 1
                if len(examples_false) < 8:
                    examples_false.append(
                        {"viol": "emp>cert", "gene": [d, m, gs],
                         "cert_L": cert.L_upper_bound, "empL": empL}
                    )

        # state_norm gate
        sres = verify_gene_safe(gene)
        if sres.ok:
            n_admit_state += 1
            # sample only a subset for the expensive long-seq check
            if rng.random() < 0.10:
                mx = my_state_norm_blowup(d, m, gs)
                if not np.isfinite(mx) or mx > 1.0 + 1e-6:
                    false_admit_state += 1
                    if len(examples_false) < 8:
                        examples_false.append(
                            {"viol": "state_norm", "gene": [d, m, gs], "max_abs_s": mx}
                        )

    print(f"genes tested:            {len(all_genes)}")
    print(f"contraction admitted:    {n_admit_contr}")
    print(f"contraction FALSE ADMITS:{false_admit_contr}")
    print(f"  worst admitted emp_L:  {worst_admit_emp_L:.6f}  (must be < 1.0)")
    print(f"  max emp/cert ratio:    {max_emp_over_cert:.6f}  (must be <= 1.0)")
    print(f"certifier != closed-form (reject of cf<1 gene): {contr_reject_with_closedform_lt1}")
    print(f"state_norm admitted:     {n_admit_state}")
    print(f"state_norm FALSE ADMITS: {false_admit_state}")
    if examples_false:
        print("FALSE-ADMIT EXAMPLES:")
        for e in examples_false:
            print("  ", e)
    sound = (false_admit_contr == 0 and false_admit_state == 0
             and worst_admit_emp_L < 1.0 and max_emp_over_cert <= 1.0 + 1e-9
             and contr_reject_with_closedform_lt1 == 0)
    print(f"\nSOUNDNESS HOLDS (independent): {sound}")
    return 0 if sound else 2


if __name__ == "__main__":
    sys.exit(main())
