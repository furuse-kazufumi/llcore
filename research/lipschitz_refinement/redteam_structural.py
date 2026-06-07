# SPDX-License-Identifier: Apache-2.0
"""Red-team the STRUCTURAL A2 claim and the 'exact' over-claim, independently.

Claims under attack:
  (C1) "A2 gain is structurally 0 for ALL decay in [0,1]" — the implementer argues:
       free rejects  <=> |J(1)| >= 1  (because the only other free endpoint is
       |J(0)|=|decay|<=1, never the cause). achievable rejects <=> |J(1)| >= 1 too
       (shares J(1)). So free-reject <=> achievable-reject. I test this by EXHAUSTIVE
       reasoning + a fine independent grid, looking for ANY (d,m,g) where the achievable
       sup < 1 but the free sup >= 1.

  (C2) Is the gain TRULY always 0, or only on this grid? I search adversarially:
       a gain requires L_free >= 1 AND L_achievable < 1, i.e.
         max(|decay|, |J(1)|) >= 1  AND  max(|J(tmin)|, |J(1)|) < 1.
       Since |J(1)| appears in both, gain requires |J(1)| < 1 AND |decay| >= 1.
       But |decay| >= 1 only at decay == 1 exactly (clip upper). At decay==1, J(t)=1 for
       ALL t (since (1-decay)=0), so |J(1)|=1 too -> achievable also rejects. So the only
       way |decay|>=1 forces |J(1)|=1. => gain impossible. I verify this corner explicitly
       and also scan decay extremely close to 1 from below.

  (C3) 'L_achievable is the EXACT box sup'. The 16 'over-rejections' have
       L_achievable = 1.0000000000000002 > 1 while empirical ~0.99999. Is the TRUE box sup
       exactly 1, or did float make L_achievable spuriously cross 1 (which would make the
       refinement REJECT a truly-contractive gene = an over-rejection defect, not just
       'finite-sample miss')? I compute the true sup in exact-ish high precision.
"""
from __future__ import annotations

import json
import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
getcontext().prec = 50

from llcore.state_update import StateUpdateGene  # noqa: E402
from achievable_lipschitz import L_achievable, L_free, t_min_achievable  # noqa: E402


def Jbox_true_sup(d, m, g, n=4001):
    """Independent fine 1-D scan of true box sup of |J| (J monotone in t, t in [sech2(M),1])."""
    M = abs(m) + abs(g)
    tmin = 1.0 / math.cosh(M) ** 2
    ts = np.linspace(tmin, 1.0, n)
    J = d + (1.0 - d) * g * ts
    return float(np.max(np.abs(J)))


def main():
    out = {}

    # ---- C1/C2: exhaustive fine search for any A2 gain ----
    rng = np.random.default_rng(999)
    gain_found = []
    n_scan = 0
    # dense grid + heavy random, biased toward decay near 1 and |J(1)| near 1
    decays = list(np.linspace(0.0, 1.0, 201)) + list(1.0 - np.logspace(-12, -1, 50))
    for d in decays:
        d = min(max(d, 0.0), 1.0)
        for m in np.linspace(-1, 1, 21):
            for g in np.linspace(-2, 2, 41):
                n_scan += 1
                lf = L_free(StateUpdateGene(d, m, g))
                la = L_achievable(StateUpdateGene(d, m, g))
                # gain would be: free rejects (lf>=1) but achievable admits (la<1)
                if lf >= 1.0 and la < 1.0:
                    gain_found.append((d, m, g, lf, la))
    # add 200k random
    for _ in range(200000):
        d = float(rng.uniform(0, 1)); m = float(rng.uniform(-1, 1)); g = float(rng.uniform(-2, 2))
        n_scan += 1
        lf = L_free(StateUpdateGene(d, m, g)); la = L_achievable(StateUpdateGene(d, m, g))
        if lf >= 1.0 and la < 1.0:
            gain_found.append((d, m, g, lf, la))

    out["C2_gain_search"] = {
        "points_scanned": n_scan,
        "value_level_gain_found": len(gain_found),
        "examples": gain_found[:10],
        "note": "value-level gain = lf>=1 & la<1. If 0, A2 structurally cannot gain on this map.",
    }

    # explicit decay==1 corner: J(t)=1 for all t -> both reject, no gain
    d1_la = L_achievable(StateUpdateGene(1.0, 0.5, 1.5))
    d1_lf = L_free(StateUpdateGene(1.0, 0.5, 1.5))
    out["C2_decay_eq_1_corner"] = {"L_free": d1_lf, "L_achievable": d1_la,
                                   "both_reject (>=1)": d1_lf >= 1.0 and d1_la >= 1.0}

    # ---- C3: the 16 over-rejection genes. Is true box sup EXACTLY 1, or did float push la>1 spuriously? ----
    over_genes = [(1/3, mm, -2.0) for mm in np.linspace(-1, 1, 16)]
    c3 = []
    for (d, m, g) in over_genes:
        la = L_achievable(StateUpdateGene(d, m, g))
        # true sup: at t=1, J(1)=d+(1-d)*g. For d=1/3,g=-2: J(1)=1/3+(2/3)(-2)=1/3-4/3=-1 -> |J(1)|=1 EXACTLY.
        # So the true box sup IS exactly 1 (analytically), independent of mix.
        j1 = d + (1.0 - d) * g
        j1_exact = Decimal(1) / 3 + (Decimal(2) / 3) * Decimal(-2)  # for the d=1/3,g=-2 case
        true_sup_fine = Jbox_true_sup(d, m, g, n=20001)
        c3.append({"d": d, "m": m, "g": g, "L_achievable": la, "J1": j1,
                   "abs_J1": abs(j1), "true_sup_fine_scan": true_sup_fine,
                   "J1_exact_decimal": str(j1_exact)})
    out["C3_over_rejection_genes"] = {
        "analytic_true_sup_is_exactly_1": True,
        "explanation": "At d=1/3,g=-2: J(1)=1/3+(2/3)*(-2)=-1, so |J(1)|=1 EXACTLY (any mix). "
                       "True box sup is exactly 1 => these genes are on the contraction boundary, "
                       "NOT strictly contractive. Rejecting them is CORRECT (L<1 strict fails). "
                       "L_achievable=1.0000000000000002 is float rounding of exactly 1; the verdict "
                       "(reject) is the sound one. Free certifier rejects the identical 16.",
        "samples": c3[:6],
    }

    Path(__file__).resolve().parent.joinpath("redteam_structural_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
