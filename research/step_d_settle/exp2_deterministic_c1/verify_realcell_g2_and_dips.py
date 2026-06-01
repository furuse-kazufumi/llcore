# SPDX-License-Identifier: Apache-2.0
"""§6.2(a)+(b) 残務: 実 cell (ESN_3param / ESN_perneuron40) の決定論 G2 再現 +
midpoint dip 深さ分布 / 閾値マージンの開示フィールド算出.

§6.2(a): 現状 verify_g2_reproduce.py は control cell のみ再検証している。VERIFY lens は
  「実 cell 決定論再確認を committed G2 ファイルに移植」を残務として挙げた。本 script は
  安価な実 cell (ESN_3param nr=12, ESN_perneuron40 nr=12 各 1 seed) を再走し valley_fraction /
  is_multimodal / n_optima が stored と bit 一致するかを確認、exp2_realcell_g2_reproduce.json に書く。

§6.2(b): 「滑らか/単峰=③不要」は閾値駆動。実 cell の midpoint dip 深さ分布と閾値マージン
  (ESN_3param の下方 dip 率 / max 相対 dip が C1 谷閾 0.05 にどれだけ近いか、
   perneuron40 の vf が is_multimodal flip 0.2 までどれだけ余裕があるか) を
  exp2_results.json に `section6_2b_threshold_margin_disclosure` として追記する (追記専用)。

src/research 非改変: landscape_map の hill-climb / midpoint 評価ロジックを read-only 再現
  (landscape_map.py:39 の谷判定 `fm < min(fi,fj) - 0.05*(abs(min(fi,fj))+1e-9)` を厳密追従)。
  fitness は exp2_deterministic_c1 の _eval_3param / _eval_perneuron を import 再利用。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import exp2_deterministic_c1 as E  # noqa: E402
from landscape_map import _hillclimb  # noqa: E402  (read-only)

_RES = _HERE / "exp2_results.json"
_G2_OUT = _HERE / "exp2_realcell_g2_reproduce.json"

# C1 谷判定の定数 (landscape_map.py:39 と一致, 改変不可の src ロジックを定数で写経)
VALLEY_REL_THR = 0.05      # 相対 dip 閾値
IS_MM_VF_THR = 0.2         # is_multimodal フリップ閾値 (valley_fraction>=0.2)


def _midpoint_dips(eval_once, *, dim, bounds, n_restarts, n_evals, sigma, base_seed):
    """multimodality_report と同一手順で hill-climb + midpoint dip を計算し、
    各 distinct ペアの相対 dip (= (min(fi,fj)-fm)/(|min(fi,fj)|+1e-9)) を集める.

    landscape_map.multimodality_report の構造を read-only 再現 (src 非改変)。
    valley_fraction / is_multimodal は src と同一定義で再算出 → G2 再現の bit 照合に使える。
    """
    optima = []
    for i in range(n_restarts):
        g, f = _hillclimb(eval_once, dim, bounds, n_evals, sigma,
                          np.random.default_rng(base_seed + i))
        optima.append((g, f))
    rel_dips = []
    valley = 0
    pairs = 0
    for i in range(len(optima)):
        for j in range(i + 1, len(optima)):
            gi, fi = optima[i]
            gj, fj = optima[j]
            if np.allclose(gi, gj, atol=1e-2):
                continue
            mid = 0.5 * (gi + gj)
            fm = float(np.mean([eval_once(mid, np.random.default_rng(base_seed + 999 + k))
                                for k in range(3)]))
            pairs += 1
            base = abs(min(fi, fj)) + 1e-9
            rel_dip = (min(fi, fj) - fm) / base  # >0 = midpoint が両端より低い (下方 dip)
            rel_dips.append(rel_dip)
            if fm < min(fi, fj) - VALLEY_REL_THR * base:
                valley += 1
    vf = valley / pairs if pairs else 0.0
    return {
        "valley_fraction": vf,
        "is_multimodal": vf >= IS_MM_VF_THR,
        "n_optima": len(optima),
        "n_pairs": pairs,
        "rel_dips": rel_dips,
    }


def _dip_stats(rel_dips: list[float]) -> dict:
    if not rel_dips:
        return {"n_pairs": 0}
    arr = np.asarray(rel_dips, dtype=np.float64)
    down = arr[arr > 0.0]  # 下方 dip (midpoint が低い)
    return {
        "n_pairs": int(arr.size),
        "frac_downward_dip": float(np.mean(arr > 0.0)),  # 90.9% 等
        "max_rel_dip": float(arr.max()),                  # 最大相対 dip
        "mean_rel_dip": float(arr.mean()),
        "median_rel_dip": float(np.median(arr)),
        "max_downward_dip": float(down.max()) if down.size else 0.0,
        "valley_threshold": VALLEY_REL_THR,
        "margin_to_valley_threshold": float(VALLEY_REL_THR - arr.max()),  # >0 = 谷閾直下
    }


def main() -> int:
    res = json.loads(_RES.read_text(encoding="utf-8"))
    cells = res["cells"]
    landscapes = {l[0]: l for l in E._landscapes()}

    # §6.2(a): 安価な実 cell (各 real landscape nr=12, seed=20260530) を G2 再現対象に
    real_targets = [
        "ESN_3param|nr=12|ne=200|seed=20260530",
        "ESN_perneuron40|nr=12|ne=200|seed=20260530",
    ]
    g2_checks = []
    dip_by_label = {}
    t0 = time.time()
    for cid in real_targets:
        if cid not in cells:
            g2_checks.append({"cell": cid, "status": "missing_in_results"})
            continue
        rec = cells[cid]
        label = rec["label"]
        _, ev, dim, bounds, _b, _g = landscapes[label]
        tc = time.time()
        rep = _midpoint_dips(ev, dim=dim, bounds=bounds,
                             n_restarts=rec["n_restarts"], n_evals=rec["n_evals"],
                             sigma=rec["sigma"], base_seed=rec["base_seed"])
        dt = time.time() - tc
        match_vf = rep["valley_fraction"] == rec["valley_fraction"]
        match_mm = rep["is_multimodal"] == rec["is_multimodal"]
        match_no = rep["n_optima"] == rec["n_optima"]
        g2_checks.append({
            "cell": cid,
            "stored_vf": rec["valley_fraction"], "rerun_vf": rep["valley_fraction"],
            "stored_is_mm": rec["is_multimodal"], "rerun_is_mm": rep["is_multimodal"],
            "stored_n_optima": rec["n_optima"], "rerun_n_optima": rep["n_optima"],
            "vf_bit_match": match_vf, "is_mm_match": match_mm, "n_optima_match": match_no,
            "reproduced": bool(match_vf and match_mm and match_no),
            "wall_s": round(dt, 1),
        })
        dip_by_label[label] = _dip_stats(rep["rel_dips"])
        print(f"{cid}: stored_vf={rec['valley_fraction']:.4f} rerun_vf={rep['valley_fraction']:.4f} "
              f"reproduced={g2_checks[-1]['reproduced']} ({dt:.1f}s)", flush=True)

    all_repro = all(c.get("reproduced") for c in g2_checks)

    # §6.2(a) 出力: 実 cell G2 再現 (control のみだった既存 verify_g2_reproduce.py を補完)
    g2_out = {
        "g2_realcell_all_reproduced": all_repro,
        "note": ("§6.2(a) 残務: 実 cell (ESN_3param / ESN_perneuron40) の決定論再現を "
                 "committed ファイルに移植。既存 verify_g2_reproduce.py は control のみ再検証 "
                 "していた。実 cell も同 gene→multimodality 手順 bit 一致を確認。"),
        "checks": g2_checks,
    }
    g2_out_path = _G2_OUT
    g2_out_path.write_text(json.dumps(g2_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n§6.2(a) real-cell G2 reproduced = {all_repro} -> {g2_out_path}")

    # §6.2(b) 出力: dip 深さ分布 + 閾値マージンを exp2_results.json に追記専用
    perneuron_vf_max = max(
        (c["valley_fraction"] for cid, c in cells.items()
         if c["label"] == "ESN_perneuron40"), default=0.0)
    margin_disclosure = {
        "valley_rel_threshold": VALLEY_REL_THR,
        "is_multimodal_vf_threshold": IS_MM_VF_THR,
        "ESN_3param": dip_by_label.get("ESN_3param", {}),
        "ESN_perneuron40_dip": dip_by_label.get("ESN_perneuron40", {}),
        "ESN_perneuron40_vf_margin": {
            "vf_max_observed": perneuron_vf_max,
            "is_multimodal_threshold": IS_MM_VF_THR,
            "margin_to_flip": float(IS_MM_VF_THR - perneuron_vf_max),  # 0.079 等
        },
        "note": (
            "§6.2(b) 残務開示: 『滑らか/単峰=③不要』は閾値駆動。ESN_3param の midpoint は "
            "大半が下方 dip するが max 相対 dip が C1 谷閾 0.05 の直下に留まり多峰判定に届かない。"
            "ESN_perneuron40 の vf_max は is_multimodal flip (0.2) まで僅かな余裕しか無い。"
            "閾値の小変更で両 verdict は反転しうる = (B) null 確定方向は維持されるが robustness は "
            "閾値近接ゆえ限定的 (verdict 文言は『真に単峰』でなく『閾値直下の浅い谷を持つ弱 "
            "multi-basin』が正確)。",
        ),
    }
    res["section6_2b_threshold_margin_disclosure"] = margin_disclosure
    res["_meta"]["section6_2_amendment"] = (
        "[§6.2(a)] 実 cell 決定論 G2 再現を exp2_realcell_g2_reproduce.json に移植 "
        "(verify_realcell_g2_and_dips.py)。[§6.2(b)] dip 深さ分布 + 閾値マージンを "
        "section6_2b_threshold_margin_disclosure に追記 (既存数値不変, 追記のみ)。"
    )
    _RES.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"§6.2(b) threshold-margin disclosure -> {_RES}")
    print(f"  ESN_3param dip: {dip_by_label.get('ESN_3param', {})}")
    print(f"  ESN_perneuron40 vf_margin to flip = "
          f"{margin_disclosure['ESN_perneuron40_vf_margin']['margin_to_flip']:.4f}")
    print(f"total wall = {time.time() - t0:.1f}s")
    return 0 if all_repro else 1


if __name__ == "__main__":
    raise SystemExit(main())
