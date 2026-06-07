# SPDX-License-Identifier: Apache-2.0
"""DECISIVE verification — is the deg4/deg6 rotation capability payoff REAL and SOUND, or GA noise?

The EXP-B smoke showed L3=sdp∪deg4 reaching rotation R²≈0.98 > L2=sdp ≈0.89 (2 seeds). "Too good"
— verify before trusting (user, [[feedback_benchmark_honest_disclosure]]). This is the GA-INDEPENDENT,
NON-CIRCULAR region-ceiling test (mirrors the main VERDICT exp1): classify a dense pool by tightest
sound certificate and measure the max rotation fitness REACHABLE in each cumulative gate region,
then 50 000-sample-verify the ceiling genes are genuinely contracting.

  ceiling_L2(sdp)  = max rotation R² over {inf, two_only, sdp_only}      (quadratic class)
  ceiling_L3(deg4) = max(ceiling_L2, max over deg4-certified residual)
  ceiling_L4(deg6) = max(ceiling_L3, max over deg6-only-certified residual)

Payoff(deg4) = ceiling_L3 − ceiling_L2. SOUND iff the deg4/deg6 ceiling genes have empirical ρ<1
at 50k. If ceiling_L3 ≈ ceiling_L2 ⇒ the smoke 0.98 was GA luck (the winner was sdp-certifiable or
an artifact), NOT a deg4 region payoff.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_map import CoupledGene  # noqa: E402
from coupled_components import (  # noqa: E402
    RotationObjective, _inf_certifies, _two_certifies, _sdp_certifies, empirical_spectral_radius,
)
from verifier_deg4 import cert_deg4_n2  # noqa: E402
from verifier_deg6 import cert_deg6_n2  # noqa: E402


def run(n_scan: int = 6000, seed: int = 31337, top_cert: int = 250,
        sound_samples: int = 50000, top_sound: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    obj = RotationObjective()
    t0 = time.time()

    quad_max = {"inf": -9.9, "two_only": -9.9, "sdp_only": -9.9}
    quad_argmax = {"inf": None, "two_only": None, "sdp_only": None}
    quad_rejected = []   # (fitness, decay, W) for genes no quadratic cert admits

    for _ in range(n_scan):
        decay = rng.uniform(0, 1, 2)
        W = rng.uniform(-2, 2, (2, 2))
        g = CoupledGene.make(decay=decay, W=W)
        f = obj.fitness(g)
        if _inf_certifies(g):
            if f > quad_max["inf"]:
                quad_max["inf"] = f; quad_argmax["inf"] = (decay.tolist(), W.reshape(-1).tolist())
        elif _two_certifies(g):
            if f > quad_max["two_only"]:
                quad_max["two_only"] = f; quad_argmax["two_only"] = (decay.tolist(), W.reshape(-1).tolist())
        elif _sdp_certifies(g):
            if f > quad_max["sdp_only"]:
                quad_max["sdp_only"] = f; quad_argmax["sdp_only"] = (decay.tolist(), W.reshape(-1).tolist())
        else:
            quad_rejected.append((f, decay.tolist(), W.reshape(-1).tolist()))

    ceiling_L2 = max(quad_max.values())
    print(f"[phase1] scanned {n_scan}; quad ceilings {{'inf':{quad_max['inf']:.4f},"
          f"'two':{quad_max['two_only']:.4f},'sdp':{quad_max['sdp_only']:.4f}}} "
          f"-> ceiling_L2={ceiling_L2:.4f}; quad_rejected={len(quad_rejected)} ({time.time()-t0:.0f}s)",
          flush=True)

    # Phase 2: among the highest-fitness quad-rejected genes, find deg4/deg6 certified ceilings.
    quad_rejected.sort(key=lambda r: -r[0])
    deg4_cert = []   # (fitness, decay, W) deg4 certifies (quad-rejected)
    deg6_cert = []   # (fitness, decay, W) deg6 certifies (quad-rejected)
    deg6_only = []   # deg6 certifies but deg4 does not
    examined = 0
    for f, decay, W in quad_rejected[:top_cert]:
        examined += 1
        g = CoupledGene.make(decay=np.asarray(decay), W=np.asarray(W).reshape(2, 2))
        d4 = cert_deg4_n2(g)
        d6 = cert_deg6_n2(g)
        if d4:
            deg4_cert.append((f, decay, W))
        if d6:
            deg6_cert.append((f, decay, W))
        if d6 and not d4:
            deg6_only.append((f, decay, W))

    def _ceil(lst):
        return lst[0][0] if lst else None
    ceiling_deg4_region = _ceil(deg4_cert)        # highest-fitness deg4-certified residual gene
    ceiling_deg6only_region = _ceil(deg6_only)
    ceiling_L3 = max([ceiling_L2] + ([ceiling_deg4_region] if ceiling_deg4_region is not None else []))
    ceiling_L4 = max([ceiling_L3] + ([ceiling_deg6only_region] if ceiling_deg6only_region is not None else []))
    print(f"[phase2] examined top {examined} quad-rejected; deg4_cert={len(deg4_cert)} "
          f"deg6_cert={len(deg6_cert)} deg6_only={len(deg6_only)}; "
          f"ceiling_deg4={ceiling_deg4_region} ceiling_L3={ceiling_L3:.4f} ceiling_L4={ceiling_L4:.4f}",
          flush=True)

    # Phase 3: SOUNDNESS — the top deg4/deg6 ceiling genes must be empirically contracting @50k.
    def _sound_check(lst, k):
        bad = []
        checked = []
        for f, decay, W in lst[:k]:
            g = CoupledGene.make(decay=np.asarray(decay), W=np.asarray(W).reshape(2, 2))
            rho = empirical_spectral_radius(g, n_samples=sound_samples)
            checked.append({"fitness": round(f, 4), "rho_50k": round(rho, 5),
                            "sound": rho < 1.0})
            if rho >= 1.0:
                bad.append({"fitness": f, "rho_50k": rho, "decay": decay, "W": W})
        return checked, bad
    deg4_checked, deg4_bad = _sound_check(deg4_cert, top_sound)
    deg6only_checked, deg6only_bad = _sound_check(deg6_only, top_sound)
    print(f"[phase3] soundness@50k: deg4 top{top_sound} unsound={len(deg4_bad)} ; "
          f"deg6_only top{top_sound} unsound={len(deg6only_bad)} ({time.time()-t0:.0f}s)", flush=True)

    out = {
        "n_scan": n_scan, "seed": seed,
        "quad_region_ceilings": {k: round(v, 4) for k, v in quad_max.items()},
        "ceiling_L2_sdp": round(ceiling_L2, 4),
        "ceiling_L3_deg4": round(ceiling_L3, 4),
        "ceiling_L4_deg6": round(ceiling_L4, 4),
        "payoff_deg4_over_sdp": round(ceiling_L3 - ceiling_L2, 4),
        "payoff_deg6_over_deg4": round(ceiling_L4 - ceiling_L3, 4),
        "n_deg4_cert_in_top": len(deg4_cert), "n_deg6_only_in_top": len(deg6_only),
        "deg4_top5_fitness": [round(r[0], 4) for r in deg4_cert[:5]],
        "deg6_only_top5_fitness": [round(r[0], 4) for r in deg6_only[:5]],
        "deg4_ceiling_gene": ({"decay": deg4_cert[0][1], "W": deg4_cert[0][2]} if deg4_cert else None),
        "soundness_deg4_top": deg4_checked,
        "soundness_deg6only_top": deg6only_checked,
        "n_unsound_deg4": len(deg4_bad), "n_unsound_deg6only": len(deg6only_bad),
        "SOUND": (len(deg4_bad) == 0 and len(deg6only_bad) == 0),
        "REAL_deg4_payoff": (ceiling_L3 - ceiling_L2) >= 0.02,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "verify_region_ceiling_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("soundness_deg4_top", "soundness_deg6only_top")}, indent=2),
          flush=True)
    return out


if __name__ == "__main__":
    run()
