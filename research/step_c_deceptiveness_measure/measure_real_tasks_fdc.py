# SPDX-License-Identifier: Apache-2.0
"""Phase A CrossMetric: 実 task に FDC-behavior メトリックを適用して欺瞞性を測る.

狙い
----
``measure_real_tasks.py`` (behavior_elite_dip) は 3 実 task すべてが d* 閾値
(0.1234) 未満 = below_threshold = 滑らか・③不要 と結論した。1 メトリックの結論は
脆いため、**別メトリック (FDC-behavior)** で同じ 3 task を測り、結論が再現するか
(クロスメトリック合意) を検証する。結論が割れたらそれ自体が重要な発見。

呼び出し規約は ``measure_real_tasks.py`` と完全に同一:
  各 task factory が ``(eval_once, behavior_fn, gene_bounds, dim)`` を返し、
  メトリックを ``deceptiveness_with_ci(eval, beh, bounds, dim, rng, ...)`` で適用。
唯一の差分はメトリック module (metric_fdc_behavior) と閾値ソース
(calibrate_fdc_behavior_results.json の threshold_at_d_star=0.30)。

実 task 側コードは read-only (一切変更しない)。本 script は
step_c_deceptiveness_measure/ 内の新規橋渡しラッパのみ。

honest disclosure
-----------------
- 各 task の欺瞞性は複数 seed で mean ± std / 95% CI を報告 (単一値で判断しない)。
- 退化 (degenerate: fitness か behavior 距離の分散ゼロ) を必ず検出・記録する。
  退化していたら FDC は定義不能で deceptiveness=1.0 と誤読されうるので
  below_threshold 判定の信頼性に注記する。
- 閾値未満なら「③ 不要」と正直に結論づける。割れたら割れたと報告する (捏造禁止)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# --- パス設定 (src 非変更・research dir を read-only import) ---
_HERE = Path(__file__).resolve().parent
_RESEARCH = _HERE.parent
_REPO = _RESEARCH.parent
for _p in (_REPO, _RESEARCH, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _make_flip_flop():
    """flip_flop task: research/step_c_memory_tasks を再利用."""
    from step_c_memory_tasks.tasks import make_flip_flop_eval  # type: ignore

    return make_flip_flop_eval()


def _make_variable_delay_recall():
    """variable_delay_recall task: research/ea_multitask の勝者分布."""
    from ea_multitask.tasks import make_variable_delay_recall_eval  # type: ignore

    return make_variable_delay_recall_eval()


def _make_step6_text_proxy():
    """step6_text_proxy task: research/step6_real_proxy を再利用."""
    from step6_real_proxy.task import make_step6_text_proxy_eval  # type: ignore

    return make_step6_text_proxy_eval()


TASKS = {
    "flip_flop": _make_flip_flop,
    "variable_delay_recall": _make_variable_delay_recall,
    "step6_text_proxy": _make_step6_text_proxy,
}


def main() -> int:
    rng_seed = 12345  # measure_real_tasks.py と同じ seed (比較可能性のため)
    out = {
        "metric": "fdc_behavior",
        "purpose": "Phase A CrossMetric: re-test behavior_elite_dip below-threshold conclusion with FDC-behavior",
        "threshold_at_d_star": None,
        "d_star": None,
        "threshold_source": "calibrate_fdc_behavior_results.json",
        "rng_seed": rng_seed,
        "measure_params": {"n_samples": 2000, "honest_n_trials": 10, "n_repeats": 5},
        "tasks": {},
    }

    # FDC-behavior の校正結果から閾値をロード (d*=0.16 における欺瞞性閾値=0.30)
    calib_path = _HERE / "calibrate_fdc_behavior_results.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        cal = calib.get("calibration", {})
        out["threshold_at_d_star"] = cal.get("threshold_at_d_star")
        out["d_star"] = cal.get("d_star")
        out["calibration_spearman_vs_d"] = cal.get("spearman_vs_d")

    threshold = out["threshold_at_d_star"]

    from metric_fdc_behavior import deceptiveness_with_ci  # type: ignore

    for name, factory in TASKS.items():
        entry: dict = {"status": None}
        try:
            eval_fn, behavior_fn, bounds, dim = factory()
        except Exception as e:  # noqa: BLE001
            entry["status"] = "unavailable"
            entry["error"] = f"{type(e).__name__}: {e}"
            out["tasks"][name] = entry
            continue

        # adapter sanity: bounds / dim の形状を記録 (landscape を変質させていないか)
        try:
            lo, hi = bounds
            entry["adapter_sanity"] = {
                "dim": int(dim),
                "bounds_lo": np.asarray(lo, dtype=float).tolist(),
                "bounds_hi": np.asarray(hi, dtype=float).tolist(),
            }
        except Exception:  # noqa: BLE001
            pass

        # メトリック適用 (複数 seed, behavior_elite_dip と同一パラメータ)
        try:
            rng = np.random.default_rng(rng_seed)
            res = deceptiveness_with_ci(
                eval_fn, behavior_fn, bounds, dim, rng,
                n_samples=2000, honest_n_trials=10, n_repeats=5,
            )
            entry["status"] = "measured"
            entry["deceptiveness_mean"] = res["deceptiveness_mean"]
            entry["deceptiveness_std"] = res["deceptiveness_std"]
            entry["ci95_lo"] = res.get("ci95_lo")
            entry["ci95_hi"] = res.get("ci95_hi")
            entry["fdc_mean"] = res.get("fdc_mean")
            entry["fdc_std"] = res.get("fdc_std")
            entry["any_degenerate"] = res.get("any_degenerate")
            entry["below_threshold"] = (
                bool(res["deceptiveness_mean"] < threshold)
                if threshold is not None else None
            )
            # honest: 退化していたら判定の信頼性に警告
            if res.get("any_degenerate"):
                entry["warning"] = (
                    "degenerate sample(s): fitness or behavior-distance variance ~0; "
                    "FDC undefined for those repeats (treated as 0). below_threshold "
                    "judgment may be unreliable."
                )
        except Exception as e:  # noqa: BLE001
            entry["status"] = "metric_error"
            entry["error"] = f"{type(e).__name__}: {e}"
        out["tasks"][name] = entry

    # クロスメトリック合意の集計
    measured = {k: v for k, v in out["tasks"].items() if v.get("status") == "measured"}
    below = [k for k, v in measured.items() if v.get("below_threshold") is True]
    out["summary"] = {
        "n_measured": len(measured),
        "n_below_threshold": len(below),
        "all_below_threshold": (len(measured) == len(TASKS) and len(below) == len(TASKS)),
        "tasks_below": below,
        "tasks_above": [
            k for k, v in measured.items() if v.get("below_threshold") is False
        ],
        # behavior_elite_dip は全 below だった → 一致するか
        "agrees_with_behavior_elite_dip": (
            (len(measured) == len(TASKS)) and (len(below) == len(TASKS))
        ),
    }

    out_path = _HERE / "fdc_behavior_crossmetric.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
