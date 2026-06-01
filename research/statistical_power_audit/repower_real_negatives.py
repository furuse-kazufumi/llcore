# SPDX-License-Identifier: Apache-2.0
"""(B) 検出力逆算 — 実 negative の観測効果量から observed power と 80% power 必要 n を算出.

目的: 「現行 n が underpowered か」を定量化。Wilcoxon は閉形 power 式が無いので全て
シミュレーション。対象は既存 results JSON から **実 per-seed paired delta** を復元
(仮定値でなく実分布)。

手順 (DESIGN.repower_plan):
(1) observed power = 実 per-seed paired delta を母集団とみなしブートストラップ再標本
    (B回) → 各再標本で full strict gate (片側 Wilcoxon p<0.05 ∧ |psd|>=0.147 ∧ diff>0) を
    適用 → pass 率 = 現行 n での achieved power。
(2) 80% power 必要 n: 観測 delta の分布から n を sweep し power 曲線、power>=0.80 の最小 n=n80。
    parametric (正規) と bootstrap (実 delta から復元抽出) の両法。
(3) gate 条件分解: 「p<0.05 のみ」「|psd|>=0.147 のみ」「diff>0 のみ」各単独 power → 律速条件特定。
(4) Cohen dz (= mean/sd of delta) も報告。
G3 破綻ゲート: power 計算器を null(power<=0.05) / 大効果(>=0.98) / C-gen4b 模擬で校正。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _audit_common as AC  # noqa: E402

ALPHA = 0.05
MIN_EFFECT = 0.147
N_SWEEP = [10, 15, 20, 30, 45, 60, 90, 120, 200, 300]


def _gate_pass_on_delta(delta: np.ndarray, *, alpha: float = ALPHA,
                        min_effect: float = MIN_EFFECT,
                        cond: str = "full") -> bool:
    """paired delta 配列 (a-b) に gate を適用。a,b の元を復元せず delta 直接で評価.

    片側 Wilcoxon は delta vs 0 と等価 (signed-rank)。psd は (#正-#負)/n。
    cond: 'full' | 'p_only' | 'effect_only' | 'diff_only' で律速分解。
    """
    diff = float(np.mean(delta))
    psd = AC._paired_sign_delta(delta)
    # 片側 Wilcoxon p (delta vs 0)。AC._paired_p は (a,b) を取るので delta と 0 配列で呼ぶ。
    p = AC._paired_p(delta, np.zeros_like(delta))
    c_diff = diff > 0.0
    c_p = p < alpha
    c_eff = abs(psd) >= min_effect
    if cond == "full":
        return bool(c_diff and c_p and c_eff)
    if cond == "p_only":
        return bool(c_p)
    if cond == "effect_only":
        return bool(c_eff)
    if cond == "diff_only":
        return bool(c_diff)
    raise ValueError(cond)


def _bootstrap_power(delta_pop: np.ndarray, n: int, *, B: int, rng: np.random.Generator,
                     cond: str = "full") -> float:
    """delta_pop から復元抽出 n で B 回再標本し gate pass 率 = power."""
    npop = len(delta_pop)
    passes = 0
    for _ in range(B):
        idx = rng.integers(0, npop, size=n)
        sample = delta_pop[idx]
        if _gate_pass_on_delta(sample, cond=cond):
            passes += 1
    return passes / B


def _parametric_power(mean: float, sd: float, n: int, *, B: int,
                      rng: np.random.Generator, cond: str = "full") -> float:
    """正規 N(mean, sd) から n を B 回 draw し gate pass 率 = power."""
    passes = 0
    for _ in range(B):
        sample = rng.normal(mean, sd, size=n)
        if _gate_pass_on_delta(sample, cond=cond):
            passes += 1
    return passes / B


def _n80_from_curve(ns: list[int], powers: list[float]) -> float | None:
    """power 曲線から power>=0.80 を満たす最小 n を線形内挿."""
    for i in range(len(ns)):
        if powers[i] >= 0.80:
            if i == 0:
                return float(ns[0])
            p0, p1 = powers[i - 1], powers[i]
            n0, n1 = ns[i - 1], ns[i]
            if p1 == p0:
                return float(n1)
            frac = (0.80 - p0) / (p1 - p0)
            return float(n0 + frac * (n1 - n0))
    return None  # 到達せず


def _calibrate_power_engine(B: int, rng: np.random.Generator) -> dict:
    """G3: power 計算器の妥当性校正 (null / 大効果 / C-gen4b 模擬)."""
    # (a) true null: diff=0 → power(= 偽陽性率) <= alpha
    null_pop = rng.normal(0.0, 0.10, size=200)
    null_power = _bootstrap_power(null_pop, 15, B=B, rng=rng)
    # parametric null も
    null_param = _parametric_power(0.0, 0.10, 15, B=B, rng=rng)
    # (b) 大効果 (corridor d=0.20 相当, δ≈+1.0): 全 delta > 0, 平均 0.2 sd 0.05
    big_pop = np.abs(rng.normal(0.20, 0.05, size=200)) + 0.05
    big_power = _bootstrap_power(big_pop, 20, B=B, rng=rng)
    # (c) C-gen4b 模擬: diff=0.063, sd=0.12, n=15 → ~0.59 / n=30 → ~0.83 (DESIGN 予備値)
    cg_15 = _parametric_power(0.06255, 0.12, 15, B=B, rng=rng)
    cg_30 = _parametric_power(0.06255, 0.12, 30, B=B, rng=rng)
    valid = (null_power <= 0.06 and big_power >= 0.95
             and abs(cg_15 - 0.59) <= 0.08 and abs(cg_30 - 0.83) <= 0.08)
    return {
        "null_power_bootstrap": null_power, "null_power_parametric": null_param,
        "big_effect_power": big_power,
        "cgen4b_sim_n15": cg_15, "cgen4b_sim_n30": cg_30,
        "checks": {
            "null_le_0.06": null_power <= 0.06,
            "big_ge_0.95": big_power >= 0.95,
            "cgen4b_n15_059pm008": abs(cg_15 - 0.59) <= 0.08,
            "cgen4b_n30_083pm008": abs(cg_30 - 0.83) <= 0.08,
        },
        "power_engine_valid": bool(valid),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    guard = AC.RunGuard.start("repower_real_negatives")
    B = 1000 if args.smoke else 5000
    n_sweep = [15, 30, 60] if args.smoke else N_SWEEP
    rng = np.random.default_rng(20260601)

    print(f"[repower] B={B} n_sweep={n_sweep} alpha={ALPHA} min_effect={MIN_EFFECT}")
    calib = _calibrate_power_engine(B, np.random.default_rng(424242))
    print(f"  G3 power-engine: null={calib['null_power_bootstrap']:.3f} "
          f"big={calib['big_effect_power']:.3f} "
          f"cg4b_n15={calib['cgen4b_sim_n15']:.3f} cg4b_n30={calib['cgen4b_sim_n30']:.3f} "
          f"VALID={calib['power_engine_valid']}")

    negs = AC.load_real_negative_deltas()
    # 対象: diff>0 符号一貫 (underpowered 候補) + diff<0 対照 (健全側)
    targets = [
        "C-gen4b_MAPE_vs_random",   # diff>0, underpowered 候補
        "C-gen4a_MAPE_vs_panmictic",  # diff<0, 真に効果無し対照
        "flip_flop_MAPE_vs_random",   # δ=+0.33, p=0.15 既知 underpowered 自認
        "flip_flop_MAPE_vs_rr_hillclimb",
        "delayed_parity_MAPE_vs_random",  # 床 R²≈0
        "C-gen3_MAPE_vs_randselect",  # PASS 対照 (power 十分のはず)
    ]

    findings: dict[str, dict] = {}
    for name in targets:
        if name not in negs:
            continue
        rec = negs[name]
        delta = np.asarray(rec["delta"], dtype=np.float64)
        n_obs = len(delta)
        mean = float(np.mean(delta))
        sd = float(np.std(delta, ddof=1))
        dz = float(mean / sd) if sd > 0 else (np.inf if mean > 0 else 0.0)

        # observed power (現行 n) — bootstrap
        obs_power = _bootstrap_power(delta, n_obs, B=B, rng=rng)

        # power 曲線 (parametric + bootstrap)
        boot_curve = [_bootstrap_power(delta, n, B=B, rng=rng) for n in n_sweep]
        param_curve = [_parametric_power(mean, sd, n, B=B, rng=rng) for n in n_sweep]
        n80_boot = _n80_from_curve(n_sweep, boot_curve)
        n80_param = _n80_from_curve(n_sweep, param_curve)

        # gate 条件分解 (現行 n での各単独 power)
        cond_power = {
            c: _bootstrap_power(delta, n_obs, B=B, rng=rng, cond=c)
            for c in ("p_only", "effect_only", "diff_only")
        }
        # 律速 = full に最も近い (最小) 単独 power の条件
        limiting = min(cond_power, key=cond_power.get)

        underpowered = bool(mean > 0 and obs_power < 0.80)
        findings[name] = {
            "note": rec.get("note", ""),
            "source": rec["source"],
            "n_observed": n_obs, "diff": mean, "sd": sd, "cohen_dz": dz,
            "paired_sign_delta": AC._paired_sign_delta(delta),
            "cliff_delta_textbook": AC.textbook_cliff_delta(rec["a"], rec["b"]),
            "wilcoxon_p_onesided": AC._paired_p(rec["a"], rec["b"]),
            "observed_power_at_n": obs_power,
            "n80_parametric": n80_param, "n80_bootstrap": n80_boot,
            "power_curve_n": n_sweep,
            "power_curve_bootstrap": boot_curve,
            "power_curve_parametric": param_curve,
            "cond_power_at_n": cond_power, "limiting_condition": limiting,
            "underpowered": underpowered,
            "delta_per_seed": delta.tolist(),
        }
        print(f"  {name}: diff={mean:+.4f} dz={dz:+.2f} obs_power={obs_power:.3f} "
              f"n80(boot)={n80_boot} n80(param)={n80_param} "
              f"limit={limiting} underpowered={underpowered}")

    meta = guard.finish()
    payload = {
        "_meta": {**meta, "design": "(B) repower real negatives", "B": B,
                  "n_sweep": n_sweep, "alpha": ALPHA, "min_effect": MIN_EFFECT,
                  "note": "実 per-seed paired delta を exp_ea3/exp_c2c3 results JSON から復元 (仮定値でない)。"},
        "power_engine_calibration_G3": calib,
        "findings": findings,
    }
    out = AC.dump_json(AC.AUDIT_DIR / "repower_real_negatives_results.json", payload)
    print(f"[repower] wrote {out}  ({meta['wall_clock_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
