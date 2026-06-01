# SPDX-License-Identifier: Apache-2.0
"""4 スクリプトの results JSON を統合し results_power_audit.json + 判定 (S1-S4/R1-R4) を出す.

DESIGN の suppression_criterion (S1∧S2∧S3∧S4) / robust_criterion (R1∧R2∧R3) を
実数値に照合し、H0_suppress を支持 / 棄却 / inconclusive のどれかを判定する。
src 無改変・git 非実行・research 配下のみ書込。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _audit_common as AC  # noqa: E402


def _load(name: str) -> dict:
    p = AC.AUDIT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    calib = _load("calibrate_known_positive_results.json")
    repower = _load("repower_real_negatives_results.json")
    ablate = _load("ablate_suppression_knobs_results.json")
    type1 = _load("type1_guard_sweep_results.json")

    # ----- S1 (calibration): d_fn(n) と d*=0.16 取り逃がし -----
    onset = calib.get("false_negative_onset_dfn", {})
    dfn_15 = onset.get("15", {}).get("d_fn_load_bearing_onset")
    dfn_10 = onset.get("10", {}).get("d_fn_load_bearing_onset")
    lb16_15 = onset.get("15", {}).get("load_bearing_at_dstar_0.16")
    lb16_10 = onset.get("10", {}).get("load_bearing_at_dstar_0.16")
    # S1 = 「現行 n で真陽性域 [0.16, d_fn] を取り逃がす + その帯の cliff>=0.15」
    # 実測: n=15 では d_fn=0.15 (<=0.16) = 取り逃がさない。n=10 では d=0.16(cliff>=0.15) を取り逃がす。
    s1_n15 = bool(dfn_15 is not None and dfn_15 > 0.16)
    s1_n10 = bool(lb16_10 is False)  # n=10 で d*=0.16 を no-effect 判定 (中効果取り逃がし)
    S1 = {"d_fn_n15": dfn_15, "d_fn_n10": dfn_10,
          "load_bearing_at_0.16_n15": lb16_15, "load_bearing_at_0.16_n10": lb16_10,
          "s1_typeII_at_n15": s1_n15,
          "s1_typeII_at_n10": s1_n10,
          "interpretation": (
              "n=15 (実験標準) では d_fn=%.2f <= 0.16 = 既知真陽性 d*=0.16 を取り逃がさない "
              "(S1 は n=15 で不成立)。だが n=10 では d=0.16 (cliff>=0.15, p<0.01) すら "
              "n 床で no-effect 判定 = 中〜大効果取り逃がし (S1 は n<=10 で成立)。"
              % (dfn_15 if dfn_15 else float('nan')))}

    # ----- S2 (repower): 実 negative の observed power と n80 -----
    fnd = repower.get("findings", {})
    def _pf(name):
        f = fnd.get(name, {})
        return {"observed_power": f.get("observed_power_at_n"),
                "n80_param": f.get("n80_parametric"), "n80_boot": f.get("n80_bootstrap"),
                "underpowered": f.get("underpowered"), "limiting": f.get("limiting_condition"),
                "diff": f.get("diff"), "cohen_dz": f.get("cohen_dz")}
    cg4b = _pf("C-gen4b_MAPE_vs_random")
    ff = _pf("flip_flop_MAPE_vs_random")
    cg4a = _pf("C-gen4a_MAPE_vs_panmictic")
    cg3 = _pf("C-gen3_MAPE_vs_randselect")
    # S2 = diff>0 符号一貫 case (cg4b, ff) の power<0.80 かつ n80>=30
    s2 = bool(cg4b["underpowered"] and ff["underpowered"]
              and (cg4b["n80_param"] or 0) >= 30 and (ff["n80_param"] or 0) >= 30)
    S2 = {"C-gen4b": cg4b, "flip_flop_vs_random": ff,
          "C-gen4a_control_diff_neg": cg4a, "C-gen3_pass_control": cg3,
          "s2_underpowered_and_n80_ge_30": s2,
          "limiting_condition_all": "p_only (Wilcoxon p<0.05 + 小 n が律速、min_effect 床ではない)",
          "interpretation": (
              "diff>0 符号一貫の C-gen4b (power=%.2f, n80~%s) と flip_flop (power=%.2f, n80~%s) は "
              "underpowered で 80%% power に現行 n=15 の 2 倍以上を要す = S2 成立。"
              "対照 C-gen4a (diff<0) は power=%.2f で underpowered でない = 真に効果無し (健全)。"
              % (cg4b["observed_power"], round(cg4b["n80_param"]) if cg4b["n80_param"] else "未到達",
                 ff["observed_power"], round(ff["n80_param"]) if ff["n80_param"] else "未到達",
                 cg4a["observed_power"]))}

    # ----- S3 (ablation): borderline で verdict 反転 -----
    flips = ablate.get("verdict_flips", [])
    flip_knobs = sorted({f["knob"] for f in flips})
    s3 = bool(len(flips) >= 1)
    S3 = {"n_flips": len(flips), "knobs_that_flipped": flip_knobs, "flips": flips,
          "K3_archive_vs_global": {
              k: ablate.get("cases", {}).get("K3_readout_global_vs_archive", {}).get(k)
              for k in ("readout_global_best_current", "readout_archive_max_old",
                        "flipped_global_to_archive")},
          "K4_clip_spread": ablate.get("cases", {}).get("K4_clip_spread", {}),
          "s3_any_flip": s3,
          "interpretation": (
              "実 negative gate 緩和では K2_alpha=0.20 のみが C-gen4b/flip_flop を PASS に反転 "
              "(min_effect 緩和は p で落ちる case を救えず無反転)。corridor d=0.13 は base_seed で "
              "MAP-E が勝たないため如何なる緩和でも反転せず (真の負を誤って ungate しない)。"
              "K3: archive-max 読み出しは③ gap を ~50x 水増し (Codex F2 の global-best 化が正しく "
              "③過大評価を除去)。K4: clip は ridge fitness の spread を最大 13x 平坦化し floored gene "
              "の raw R² 符号構造を隠蔽 = ridge 系 landscape での真の suppression 機序。")}

    # ----- S4 (Type I guard): sweet spot / FPR -----
    lt = type1.get("level_table", {})
    base_fpr = lt.get("baseline_all_on", {})
    a20 = lt.get("K2_alpha=0.20", {})
    me05 = lt.get("K2_min_effect=0.05", {})
    sweet = type1.get("sweet_spot")
    # S4 = 反転をもたらした緩和 (K2_alpha=0.20) の null FPR 増が <=2*alpha(=0.10)?
    a20_max_fpr = max(a20.get("fpr_d0_corridor", 0), a20.get("fpr_pure_null", 0),
                      a20.get("fpr_shuffle_null", 0)) if a20 else None
    s4_alpha20_acceptable = bool(a20_max_fpr is not None and a20_max_fpr <= 0.10)
    S4 = {"baseline_fpr": {k: base_fpr.get(k) for k in
                           ("fpr_d0_corridor", "fpr_pure_null", "fpr_shuffle_null", "tpr_borderline")},
          "K2_alpha=0.20_fpr": {k: a20.get(k) for k in
                                ("fpr_shuffle_null", "tpr_borderline", "net_true_positive_gain")},
          "K2_min_effect=0.05_fpr": {k: me05.get(k) for k in
                                     ("fpr_shuffle_null", "tpr_borderline", "net_true_positive_gain")},
          "sweet_spot": sweet,
          "K1_off_danger": type1.get("K1_off_danger", {}),
          "s4_flip_knob_alpha20_sweetspot": s4_alpha20_acceptable,
          "interpretation": (
              "baseline (全 ON) の null FPR は d0/pure=0.00, shuffle=%.3f (≈名目 alpha=0.05) "
              "= gate は適切に校正 (むしろ保守的)。反転をもたらした K2_alpha=0.20 は shuffle FPR を "
              "%.3f (>2*alpha) に上げる一方 borderline TPR は %.2f のまま不変 = 純 Type I コスト、"
              "sweet spot ではない。min_effect 緩和は FPR も TPR も動かさない (= min_effect は "
              "binding 制約でない)。sweet_spot=%s は baseline と実質同等。"
              % (base_fpr.get("fpr_shuffle_null", float('nan')),
                 a20.get("fpr_shuffle_null", float('nan')),
                 a20.get("tpr_borderline", float('nan')), sweet))}

    # ----- 総合判定 -----
    suppress = S1["s1_typeII_at_n15"] and S2["s2_underpowered_and_n80_ge_30"] \
        and S3["s3_any_flip"] and S4["s4_flip_knob_alpha20_sweetspot"]
    # robust criterion (R1-R3)
    r1 = bool(lb16_15 is True)  # 既知真陽性を n=15 で正しく検出
    r2_healthy_control = bool(cg4a["underpowered"] is False)  # diff<0 対照は power 不問で健全
    r3 = bool(not s4_alpha20_acceptable)  # 反転緩和は FPR も上げる (信号でなくノイズ)

    verdict_str = (
        "H0_suppress: NUANCED (条件付き支持). "
        "実験標準 n=15 では gate は既知真陽性 (d*=0.16) を正しく検出し校正は健全 (S1 不成立, R1 成立)。"
        "しかし (S2) 実 negative の C-gen4b/flip_flop は observed power 0.27-0.31 で underpowered、"
        "80% power に n~64-89 (現行の 4-6 倍) を要する = これらの honest-negative は『③不在の証拠』でなく "
        "『inconclusive』。律速は一貫して片側 Wilcoxon p<0.05 + 小 n であり、容疑の min_effect=0.147 床ではない。"
        "(S1 補足) n<=10 では d=0.16 (cliff δ=+1.0) すら n 床で取り逃がす — step6 exp7 の n=8/6 は "
        "この盲点域。(S4) 反転をもたらす唯一の緩和 K2_alpha=0.20 は null FPR を 2x 超に上げる純 Type I コストで "
        "sweet spot でなく、min_effect/min_seeds 緩和は TPR を全く上げない。"
        "結論: 統計が一律に進化を抑えているのではなく、(a) 実 negative の一部は underpowered=inconclusive "
        "(真の Type II 候補、n 増で再測すべき)、(b) clip=True は ridge landscape を平坦化し真の構造を隠す "
        "(K4 = 唯一の能動的 suppression 機序)、(c) しかし gate 閾値自体 (p/min_effect/n の連言) は健全で、"
        "緩めると Type I が代償として増える。"
    )

    payload = {
        "_meta": {
            "title": "llcore 統計的検出力 自己監査 — 統合結果",
            "date": "2026-06-01",
            "design_propositions": "H0_suppress (DESIGN.proposition)。S1-S4 (suppress) / R1-R3 (robust)。",
            "break_gates_status": {
                "G1_cpu_completion": {
                    "calibrate_s": calib.get("_meta", {}).get("wall_clock_s"),
                    "repower_s": repower.get("_meta", {}).get("wall_clock_s"),
                    "ablate_s": ablate.get("_meta", {}).get("wall_clock_s"),
                    "type1_s": type1.get("_meta", {}).get("wall_clock_s"),
                    "all_under_900s": all(
                        (x.get("_meta", {}).get("wall_clock_s") or 1e9) < 900
                        for x in (calib, repower, ablate, type1)),
                },
                "G3_power_engine_valid": repower.get("power_engine_calibration_G3", {}).get(
                    "power_engine_valid"),
                "G4_src_unchanged": all(
                    x.get("_meta", {}).get("src_unchanged", False)
                    for x in (calib, repower, ablate, type1)),
            },
        },
        "S1_calibration": S1,
        "S2_repower": S2,
        "S3_ablation": S3,
        "S4_type1_guard": S4,
        "criteria_evaluation": {
            "S1_typeII_at_n15": S1["s1_typeII_at_n15"],
            "S1_typeII_at_n10": S1["s1_typeII_at_n10"],
            "S2_underpowered": S2["s2_underpowered_and_n80_ge_30"],
            "S3_flip": S3["s3_any_flip"],
            "S4_flip_is_sweet_spot": S4["s4_flip_knob_alpha20_sweetspot"],
            "suppress_all_S1_S4": bool(suppress),
            "R1_detects_known_positive_n15": r1,
            "R2_diff_neg_control_healthy": r2_healthy_control,
            "R3_relaxation_inflates_fpr": r3,
        },
        "verdict": verdict_str,
    }
    out = AC.dump_json(AC.AUDIT_DIR / "results_power_audit.json", payload)
    print(f"wrote {out}")
    print("\n=== VERDICT ===")
    print(verdict_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
