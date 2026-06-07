# SPDX-License-Identifier: Apache-2.0
"""Phase 2a 追補 — 用量反応の **第二独立軸** (disturbance magnitude w̄_task) の決着 run.

## 背景: seq_len 軸の機構的欠陥 (本 runner が解決する問題)

VERDICT.md §10 の (c) 決着で、tube gate の fitness 価値が memory horizon (delay) に
**用量反応** (d0 p=.957 → d4 p=.104 → d8 p=.0056) で立ち上がることを n=20 事前登録で示した。
これを「delay 単軸の偶然でなく予測的法則」に昇格させるには、**機構的に独立な第二の不変量負荷軸**で
同じ単調性が再現する必要がある。

第二軸候補として最初に挙がった ``seq_len`` (入力列長) は **無効**である:
``CopyTask`` の target_idx = ``seq_len - 1 - delay`` なので、target 入力から readout (最終 state)
までの保持ステップ数は **常に delay (=8) で固定**。seq_len を増やしても増えるのは target *前*の
ステップ数だけで、しかも contraction (L<1) 下で古い入力は L^k で減衰するため数ステップで飽和し、
**保持 horizon (不変量負荷) を増やさない**。よって seq_len は用量反応の負荷軸として使えない。

## 正しい第二軸: 外乱振幅 w̄_task (tube 公式 r=G·w̄/(1−L) の分子側)

tube 半径 r = G·w̄/(1−L) は **入力外乱 ‖d‖∞ ≤ w̄** 下の定常追従誤差上界 (``disturbance_checker``
の cross-check が裏付け)。delay 軸が分母 1/(1−L) 側 (L^delay の保持) を動かすのに対し、
**w̄ は分子 G·w̄ 側**を動かす — 機構的に独立な負荷軸。

``DisturbedCopyTask``: 各 step の入力に外乱 d ~ U[−w̄_task, w̄_task] を加算した列で recurrence を
回し、**クリーンな** target x[seq_len-1-delay] を復元させる。保持には外乱除去 (= tight contraction
= 小さい tube 半径) が要る。tube gate は r ≤ r_max で **小 tube の gene のみ admit** するため、
w̄_task が大きいほど外乱除去能力が報われ、plain contraction gate (L<1 のみ) に対する優位が増す、
と予測する。

## 交絡の回避 (synthesis が指摘した w̄ 軸の罠への対処)

「gate の w̄ を動かすと admit 率が変わるアーティファクト」を避けるため:
- **gate の w̄_gate = W_BAR = 0.1, r_max = R_MAX = 0.05 は固定** (Phase 2a と同一)。
  → gate 幾何・binding 挙動は全 w̄_task 値で不変 = admit 率の交絡なし。
- 動かすのは **task の外乱 w̄_task のみ** (task 難度の純粋な軸)。
- w̄_task = 0.0 は外乱なし = Phase 2a d8 (seq_len=32, delay=8) の fresh-seed 再現 = アンカー。

## 事前登録 (PRE-REGISTRATION — 結果を見る前に本ファイルを commit すること)

- **主仮説 H1 (confirmatory)**: w̄_task = 0.20 (最大外乱) において、paired delta
  (trajectory_tube − contraction) の test_best_fitness 平均が 0 でない。
  検定 = sign-flip permutation test (両側, n_resamples=100,000, rng seed=7)。α=0.05。
- **DOSE 宣言 (この軸の核)**: w̄_task ∈ {0.0, 0.05, 0.10, 0.20} (delay=8, seq_len=32 固定) で
  mean delta が **単調非減少**。confirmatory は w̄_task=0.20 のみ。0.0/0.05/0.10 は exploratory
  (多重比較補正なしと明記)。予測機構 = w̄_task↑ → 累積外乱で保持信号劣化 → 外乱除去 (小 tube) gene
  の優位拡大 → gate 価値↑。
- **判定**:
  - p < 0.05 かつ delta>0 かつ degeneracy ガード全通過 → H1 supported:
    w̄_task は用量反応の **独立な第二軸** (gate 価値は外乱負荷にも比例) = 法則は多軸予測的。
  - p < 0.05 かつ delta<0 → H1 reversed: この軸で tube は fitness cost = 法則の反例 (honest 報告)。
  - p ≥ 0.05 → 用量反応はこの軸で立たず = 法則の適用範囲を delay 軸に絞る negative result。
    (|mean delta| が小さければ「外乱下でも安全税 ≈ 0」として報告)。
- **seeds**: 3000..3019 (n=20 新規; pilot 1000-1002 / run_c 2000-2019 と非重複)。
- **arm**: contraction / trajectory_tube の 2 本 (用量反応比較)。
  + w̄_task=0.20 のみ "none" を追加した 3-arm で re-skin (F6) を点検。
- **gate/GA/readout/STATE_DIM パラメータは run_3arm_ab.py から import** (Phase 2a と完全同一構成)。
- **delay=8, seq_len=32 固定** (delay は Phase 2a で効果が立った horizon; seq_len は CopyTask 既定)。

## 反証条件 (結果取得前に固定 — garden of forking paths 回避)

- (F1) w̄_task=0.20 で p≥0.05 → 外乱軸では用量反応が立たない (法則は delay 軸固有)。
- (F2) w̄_task=0.20 で p<0.05 だが mean delta<0 → この軸で tube は fitness cost = 法則の反例。
- (F3) {0→0.05→0.10→0.20} で mean delta が単調非減少でない (飽和でなく途中で有意に下降・反転) →
  非単調 (局所現象)。注: 高 w̄_task での飽和 (頭打ち) は task が両 arm とも解けなくなる ceiling の
  予測なので反証でなく上限境界 — 飽和と反転を区別して登録。
- (F5) trajectory_tube と contraction が同一集合 admit (n_rejections==0) → re-skin (vacuous)。
- (F6) w̄_task=0.20 (3-arm) で tube の優位が contraction−none で全説明され tube 固有 delta≈0 →
  「単一 L<1 gate の re-skin」= 観測された用量反応は contraction の効果であって tube gate の追加価値ではない。
- (F7) mean delta>0 だが median/10%-trimmed delta が逆符号、または最大|delta| seed 除去で p が α を
  跨ぐ → outlier 駆動。有意でも棄却 (honest disclosure: 内訳を疑う)。

実行::

    py -3.11 research/verified_memory_poc/run_wbar_decision.py

出力::

    research/verified_memory_poc/results_wbar_decision.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[1] / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llcore.fitness import calibrate_baseline  # noqa: E402
from llcore.fitness.tasks import FixedReadout  # noqa: E402
from llcore.state_update import StateUpdateGene, run_sequence  # noqa: E402

from run_3arm_ab import (  # noqa: E402  (Phase 2a と同一構成を保証する import)
    GA_KW,
    STATE_DIM,
    TEST_N_TRIALS,
    TRAIN_N_TRIALS,
    W_BAR,
    R_MAX,
    _READOUT,
    _ensure_utf8_stdout,
    _evolve_arm,
    _fitness_func_for,
)
from run_c_decision import signflip_pvalue  # noqa: E402

# ---- 事前登録パラメータ (docstring と一致させること) ------------------------
W_TASK_VALUES = [0.0, 0.05, 0.10, 0.20]   # DOSE 軸 (task 外乱振幅)
CONFIRMATORY_W = 0.20                       # 主仮説 H1 の対象 (最大外乱)
FIXED_DELAY = 8                             # Phase 2a で効果が立った horizon
FIXED_SEQ_LEN = 32                          # CopyTask 既定
W_SEEDS = list(range(3000, 3020))          # n=20, pilot / run_c と独立
ARMS_2 = ["contraction", "trajectory_tube"]
ARMS_3 = ["none", "contraction", "trajectory_tube"]  # re-skin 点検用 (confirmatory のみ)
PERM_N_RESAMPLES = 100_000
PERM_RNG_SEED = 7
ALPHA = 0.05


@dataclass(frozen=True)
class DisturbedCopyTask:
    """外乱注入版 CopyTask — recurrence 入力に d~U[−w_task,w_task] を加算する.

    ``CopyTask`` (src/llcore/fitness/tasks.py) と error 定義・readout 解釈を一致させ、
    唯一の差分は ``generate`` で入力列に外乱を注入する点のみ (target は **クリーン**)。
    外乱の意味論は ``disturbance_checker.rollout_with_disturbance`` と同型
    (入力加算 d、tube 公式の G=入力ゲインが束ねる外乱)。

    honest 留保: ``CopyTask`` 同様 "fixed-readout probe-based fitness" であり gene 純粋
    fitness ではない。w_task は task 難度軸であって gate の w̄_gate (固定=0.1) とは別物。
    """

    name: str = "disturbed_copy"
    seq_len: int = FIXED_SEQ_LEN
    state_dim: int = STATE_DIM
    out_dim: int = STATE_DIM
    delay: int = FIXED_DELAY
    w_task: float = 0.0
    baseline_mse: float = 1.0  # calibrate_baseline で更新

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """クリーン入力 → target=x[seq_len-1-delay] (clean) → 入力に外乱注入して返す."""
        clean = rng.uniform(-1.0, 1.0, size=(self.seq_len, self.state_dim))
        target_idx = self.seq_len - 1 - self.delay
        target = clean[target_idx].copy()                 # クリーンな保持対象
        if self.w_task > 0.0:
            d = rng.uniform(-self.w_task, self.w_task, size=clean.shape)
            disturbed = clean + d
        else:
            disturbed = clean
        return disturbed, target

    def raw_error(
        self, state_trajectory: np.ndarray, target: np.ndarray, readout: FixedReadout
    ) -> float:
        """CopyTask と同一の raw MSE (calibration / score 共有)."""
        final_state = state_trajectory[-1]
        pred = readout(final_state)
        return float(np.mean((pred - target) ** 2))

    def score(
        self, state_trajectory: np.ndarray, target: np.ndarray, readout: FixedReadout
    ) -> float:
        """CopyTask と同一: raw_error を baseline で正規化、fitness ∈ [0,1]."""
        mse = self.raw_error(state_trajectory, target, readout)
        return float(np.clip(1.0 - mse / max(self.baseline_mse, 1e-9), 0.0, 1.0))


def _build_disturbed_task(w_task: float) -> DisturbedCopyTask:
    """w_task 付き DisturbedCopyTask を calibrated baseline で構築する."""
    t = DisturbedCopyTask(w_task=w_task)
    b = calibrate_baseline(t, _READOUT)
    return replace(t, baseline_mse=float(b))


def _test_fitness_task(gene: StateUpdateGene, task, ga_seed: int) -> float:
    """held-out test fitness (独立 RNG; run_3arm_ab._test_fitness と同型, 任意 task 対応)."""
    from llcore.fitness import evaluate_gene  # 局所 import (run_3arm と同じ評価器)

    rng = np.random.default_rng(900000 + ga_seed)
    return evaluate_gene(gene, task, _READOUT, rng, n_trials=TEST_N_TRIALS)


def _gate_stats_summary(recs: list[dict]) -> dict:
    """degeneracy ガード用に arm の gate 統計を集計する."""
    rej = sum(r.get("n_rejections", 0) for r in recs)
    nch = sum(r.get("n_children_generated", 0) for r in recs)
    fb = sum(r.get("fallback_count", 0) for r in recs)
    return {
        "n_rejections": rej,
        "n_children_generated": nch,
        "fallback_count": fb,
        "reject_rate": (rej / nch) if nch > 0 else None,
        "fallback_rate": (fb / nch) if nch > 0 else None,
    }


def run_all() -> dict:
    _ensure_utf8_stdout()
    t0 = time.time()

    cells: dict = {}
    for w_task in W_TASK_VALUES:
        task = _build_disturbed_task(w_task)
        ff = _fitness_func_for(task)
        arms = ARMS_3 if abs(w_task - CONFIRMATORY_W) < 1e-12 else ARMS_2
        key = f"w{w_task:.2f}"
        cells[key] = {
            "w_task": w_task, "baseline_mse": task.baseline_mse,
            "delay": FIXED_DELAY, "seq_len": FIXED_SEQ_LEN, "arms": {},
        }
        for arm in arms:
            recs = []
            for seed in W_SEEDS:
                res = _evolve_arm(ff, arm, seed)
                best_gene = res.final_best.gene
                rec = {
                    "seed": seed,
                    "best_gene": [best_gene.decay, best_gene.mix, best_gene.gate_str],
                    "train_best_fitness": res.final_best.fitness,
                    "test_best_fitness": _test_fitness_task(best_gene, task, seed),
                }
                if res.gate_stats is not None:
                    rec.update(
                        n_rejections=res.gate_stats.n_rejections,
                        fallback_count=res.gate_stats.fallback_count,
                        n_children_generated=res.gate_stats.n_children_generated,
                    )
                recs.append(rec)
            cells[key]["arms"][arm] = recs
            mean_test = float(np.mean([r["test_best_fitness"] for r in recs]))
            gs = _gate_stats_summary(recs)
            print(f"  [{key}/{arm}] mean test-fit={mean_test:.4f}  "
                  f"rej={gs['n_rejections']} fb={gs['fallback_count']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 用量反応判定: 各 w_task で paired delta + sign-flip permutation -------
    stats = {}
    for key, cell in cells.items():
        tube = np.array([r["test_best_fitness"] for r in cell["arms"]["trajectory_tube"]])
        cont = np.array([r["test_best_fitness"] for r in cell["arms"]["contraction"]])
        deltas = tube - cont                       # paired (同一 seed)
        p = signflip_pvalue(deltas, n_resamples=PERM_N_RESAMPLES, seed=PERM_RNG_SEED)
        # outlier ガード用の robust 統計
        med = float(np.median(deltas))
        k = max(1, int(0.1 * deltas.size))
        trimmed = float(np.sort(deltas)[k:deltas.size - k].mean()) if deltas.size > 2 * k else med
        # 最大|delta| seed 除去で p が α を跨ぐか (単一 seed 依存)
        idx_drop = int(np.argmax(np.abs(deltas)))
        d_drop = np.delete(deltas, idx_drop)
        p_drop = signflip_pvalue(d_drop, n_resamples=PERM_N_RESAMPLES, seed=PERM_RNG_SEED)

        # re-skin (F6): 3-arm がある cell では tube−none と contraction−none を比較
        reskin = None
        if "none" in cell["arms"]:
            none = np.array([r["test_best_fitness"] for r in cell["arms"]["none"]])
            reskin = {
                "tube_minus_none_mean": float((tube - none).mean()),
                "contraction_minus_none_mean": float((cont - none).mean()),
                "tube_specific_mean": float(deltas.mean()),  # = tube − contraction
            }

        # gate binding ガード (両 arm)
        gate_guard = {}
        for arm in ("contraction", "trajectory_tube"):
            gate_guard[arm] = _gate_stats_summary(cell["arms"][arm])

        stats[key] = {
            "w_task": cell["w_task"],
            "mean_delta_tube_minus_contraction": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)),
            "median_delta": med,
            "trimmed_delta_10pct": trimmed,
            "n_positive": int((deltas > 0).sum()),
            "n_negative": int((deltas < 0).sum()),
            "p_signflip_two_sided": p,
            "p_after_dropping_max_abs_seed": p_drop,
            "deltas_per_seed": {str(s): float(d) for s, d in zip(W_SEEDS, deltas)},
            "role": "confirmatory" if abs(cell["w_task"] - CONFIRMATORY_W) < 1e-12 else "exploratory",
            "reskin_3arm": reskin,
            "gate_guard": gate_guard,
        }

    # ---- 用量反応 (単調性) + H1 判定 -----------------------------------------
    ordered = [stats[f"w{w:.2f}"] for w in W_TASK_VALUES]
    means = [s["mean_delta_tube_minus_contraction"] for s in ordered]
    monotone_nondec = all(means[i + 1] >= means[i] - 1e-12 for i in range(len(means) - 1))

    h1 = stats[f"w{CONFIRMATORY_W:.2f}"]
    significant = h1["p_signflip_two_sided"] < ALPHA
    # degeneracy ガード (confirmatory cell)
    g_tt = h1["gate_guard"]["trajectory_tube"]
    binding_ok = (g_tt["reject_rate"] is not None and 0.05 <= g_tt["reject_rate"] <= 0.95)
    fallback_ok = (g_tt["fallback_rate"] is not None and g_tt["fallback_rate"] < 0.01)
    outlier_ok = (np.sign(h1["median_delta"]) == np.sign(h1["mean_delta_tube_minus_contraction"])
                  and np.sign(h1["trimmed_delta_10pct"]) == np.sign(h1["mean_delta_tube_minus_contraction"]))
    guards_pass = bool(binding_ok and fallback_ok and outlier_ok)

    if not guards_pass:
        verdict = ("INVALID (degeneracy guard 不通過): "
                   f"binding_ok={binding_ok} fallback_ok={fallback_ok} outlier_ok={outlier_ok} "
                   "— 統計判定前に内訳を要再設計")
    elif significant and h1["mean_delta_tube_minus_contraction"] > 0:
        verdict = ("H1 supported: w̄_task は用量反応の独立な第二軸 — tube gate の fitness 価値が "
                   f"外乱負荷 w̄_task=0.20 で統計的に検出 (p={h1['p_signflip_two_sided']:.4f}, "
                   f"mean Δ={h1['mean_delta_tube_minus_contraction']:+.4f}, "
                   f"monotone_nondec={monotone_nondec})")
    elif significant:
        verdict = ("H1 reversed: 外乱軸で tube は fitness COST = 用量反応の法則の反例 "
                   f"(mean Δ={h1['mean_delta_tube_minus_contraction']:+.4f}, honest 報告)")
    else:
        verdict = ("(w̄ 軸) 用量反応立たず: gate 価値は memory horizon 固有で外乱負荷には非依存 "
                   f"(p={h1['p_signflip_two_sided']:.4f}); 法則の適用範囲を delay 軸に絞る negative result")

    return {
        "preregistration": {
            "confirmatory_w_task": CONFIRMATORY_W,
            "dose_axis": W_TASK_VALUES,
            "fixed_delay": FIXED_DELAY,
            "fixed_seq_len": FIXED_SEQ_LEN,
            "gate_w_bar": W_BAR, "gate_r_max": R_MAX,
            "alpha": ALPHA, "test": "sign-flip permutation, two-sided",
            "n_resamples": PERM_N_RESAMPLES, "perm_rng_seed": PERM_RNG_SEED,
            "seeds": W_SEEDS, "arms_2": ARMS_2, "arms_3_at_confirmatory": ARMS_3,
            "inherits_config_from": "run_3arm_ab.py (GA_KW/W_BAR/R_MAX/STATE_DIM/readout)",
        },
        "config": {
            "GA_KW": GA_KW, "STATE_DIM": STATE_DIM,
            "TRAIN_N_TRIALS": TRAIN_N_TRIALS, "TEST_N_TRIALS": TEST_N_TRIALS,
            "W_BAR": W_BAR, "R_MAX": R_MAX,
        },
        "cells": cells,
        "stats": stats,
        "dose_response": {
            "means_by_w_task": {f"w{w:.2f}": m for w, m in zip(W_TASK_VALUES, means)},
            "monotone_nondecreasing": monotone_nondec,
        },
        "verdict_wbar": verdict,
        "wall_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    out = run_all()
    out_path = _HERE / "results_wbar_decision.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("\n=== w̄_task dose-response (tube − contraction) ===")
    for w in W_TASK_VALUES:
        s = out["stats"][f"w{w:.2f}"]
        print(f"w_task={w:.2f}: mean Δ={s['mean_delta_tube_minus_contraction']:+.4f}  "
              f"p={s['p_signflip_two_sided']:.4f}  [{s['role']}]  "
              f"+{s['n_positive']}/-{s['n_negative']}  "
              f"med={s['median_delta']:+.4f} trim={s['trimmed_delta_10pct']:+.4f}")
    print(f"\nmonotone non-decreasing: {out['dose_response']['monotone_nondecreasing']}")
    print(f"verdict: {out['verdict_wbar']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
