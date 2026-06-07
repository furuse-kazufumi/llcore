# SPDX-License-Identifier: Apache-2.0
"""G2 再現性検証: 完走済 cell を同 seed で再走し valley_fraction が bit 一致するか確認.

決定論 fitness + 固定 base_seed の multimodality_report は構造的に再現するはずだが、
honest disclosure 規律 (実測で示す) に従い 2 cell を実走して JSON 値と照合する。
安価な control cell (dim3) を選び G1 を圧迫しない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import exp2_deterministic_c1 as E  # noqa: E402

_RES = _HERE / "exp2_results.json"
_OUT = _HERE / "exp2_g2_reproduce.json"


def main():
    res = json.loads(_RES.read_text(encoding="utf-8"))
    cells = res["cells"]
    # 安価で再現確認に適した cell を選ぶ (control_pos dim3, control_neg dim3)
    targets = [
        "ctrl_multipeak_dim3|nr=12|ne=200|seed=20260530",
        "ctrl_quadratic_dim3|nr=24|ne=300|seed=20260601",
    ]
    landscapes = {l[0]: l for l in E._landscapes()}
    checks = []
    for cid in targets:
        if cid not in cells:
            checks.append({"cell": cid, "status": "missing_in_results"})
            continue
        rec = cells[cid]
        label = rec["label"]
        _, ev, dim, bounds, _b, _g = landscapes[label]
        rep = E.multimodality_report(
            ev, dim=dim, bounds=bounds,
            n_restarts=rec["n_restarts"], n_evals=rec["n_evals"],
            sigma=rec["sigma"], base_seed=rec["base_seed"])
        match_vf = rep["valley_fraction"] == rec["valley_fraction"]
        match_mm = rep["is_multimodal"] == rec["is_multimodal"]
        match_no = rep["n_optima"] == rec["n_optima"]
        checks.append({
            "cell": cid,
            "stored_vf": rec["valley_fraction"], "rerun_vf": rep["valley_fraction"],
            "stored_is_mm": rec["is_multimodal"], "rerun_is_mm": rep["is_multimodal"],
            "stored_n_optima": rec["n_optima"], "rerun_n_optima": rep["n_optima"],
            "vf_bit_match": match_vf, "is_mm_match": match_mm, "n_optima_match": match_no,
            "reproduced": bool(match_vf and match_mm and match_no),
        })
        print(f"{cid}: stored_vf={rec['valley_fraction']} rerun_vf={rep['valley_fraction']} "
              f"reproduced={checks[-1]['reproduced']}")
    all_repro = all(c.get("reproduced") for c in checks)
    out = {"g2_all_reproduced": all_repro, "checks": checks}
    _OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nG2_all_reproduced = {all_repro} -> {_OUT}")
    return 0 if all_repro else 1


if __name__ == "__main__":
    raise SystemExit(main())
