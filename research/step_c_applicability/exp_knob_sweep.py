# SPDX-License-Identifier: Apache-2.0
"""Step C 適用条件: ③ (behavioral niching / MAP-Elites) が load-bearing になる
**欺瞞性 (deceptiveness) の閾値**を 1 つの tunable knob の sweep で特性化する.

背景 (一次情報, 再導出しない):
- `docs/poc/STEP4_SELECTION_VERDICT.md` — exp4 の **deceptive corridor** で③が decisively
  load-bearing (MAP-E が 3 baseline 全勝, p=1.9e-6, δ=+1.00)。exp5 の **smooth** で消失。
- `docs/poc/E_A_VERDICT.md` / `STEP_C_VERDICT.md` — 実 proxy / 記憶タスクは滑らかで③不要 (honest negative)。

開いた問い: ③が「立つ↔立たない」の二値結果しか無い。本実験は exp4↔exp5 を **1 つの
連続 knob = dip depth d ∈ [0,1]** で内挿し、③の優位 (MAP-E − best baseline + strict gate
pass/fail) を **d の関数として測る**。閾値 d* (③が立ち始める dip depth) を特定する。

== 選んだ knob: dip depth d (ramp に彫った谷の深さ) ==
exp4 の corridor (behavior=mean の genotypic corridor) を保ち、局所峰 (b=0.4, 高さ0.60) と
大域峰 (b=0.9, 高さ1.0) を **固定**したまま、その間を結ぶ登坂 ramp に彫る **谷 (dip) の深さ**
だけを動かす:

    local(b)  = 0.60 * exp(-(b-0.40)^2 / (2*0.08^2))
    glob(b)   = 1.00 * exp(-(b-0.90)^2 / (2*0.06^2))
    t(b)      = clip((b-0.40)/(0.90-0.40), 0, 1)
    ramp(b)   = 0.60 + t(b)*(1.00-0.60)                       # 局所峰→大域峰を結ぶ単調登坂
    dip(b)    = d * exp(-(b-0.65)^2 / (2*0.07^2))             # 谷の中央 b=0.65 に深さ d を彫る
    f(b)      = max(local(b), glob(b), ramp(b)*(1-dip(b))) + noise   for b in [0.40,0.90]

- d=0.0: ramp に谷無し → b: 0.4→0.9 が **厳密に単調増加** (downhill 0 step, 正の勾配が常に存在)。
  = 真の smooth (exp5 相当)。hill-climbing は連続した上り勾配で大域へ登れる → ③不要のはず。
  **注**: 単純な「平床 (flat floor)」では勾配 0 で hill-climb の登坂信号が消え、谷が浅くても
  smooth にならない (平床は弱い罠)。だから床でなく **正の勾配を持つ ramp** を基線にした。
- d=1.0: 谷の床 ≈ 0 → 深い dip。= exp4 の deceptive corridor。hill-climb は downhill 拒否で罠。
- 中間 d: ramp 中央の谷の深さが連続変化。**唯一の自由度が「欺瞞 (dip) の深さ」**になる。

なぜ dip depth を primary に選んだか (justification, design note 参照):
1. exp4↔exp5 の **唯一の差**が dip の有無 (exp5 は smooth_eval で dip を消した)。dip depth は
   その差を連続化した最小の 1 パラメータ → exp4/exp5 を端点として厳密に含む。
2. behavior=mean の **genotypic corridor 構造を全 d で不変**に保つ → `random` の失敗 (b≈0.5 固着で
   大域到達不能) は d 非依存の定数。これにより d を変えたときの MAP-E 優位の変化が
   「dip 越え (③ の本質効果) の難易/不要化」だけに帰属し、corridor 幅などの交絡を排除できる。
3. 峰の高さ・位置を固定するので「大域最適の価値」も d 非依存 → 優位の大小が欺瞞性のみの関数。
4. ramp 基線にすることで d=0 で hill-climb に **連続した上り勾配**を与え「dip が無ければ登れる」
   を保証 → 閾値は「勾配の有無」でなく「dip の深さ」の純粋な関数になる。

== 規律 (一次情報の方針を継承) ==
- ea_lab.py の seed 設計を踏襲: 進化 RNG = SeedSequence([base, method_idx, s]) で一意化、
  honest 再評価は index s で **全 method 共通 (common random numbers)** → paired Wilcoxon の前提充足。
- strict gate (honest_eval §5 完全版): diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15
  ∧ |paired_sign_delta|≥0.147。selection_lab.compare は前 2 条件のみなので **本 module で
  完全 gate を再実装**する (n と効果量も課す)。
- equal budget (全 method 同一 n_evals)。fresh-seed honest 再評価で elitism 持越し artifact 排除。
- selection_lab は read-only import (改変しない)。numpy のみ。CPU 完結。
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# selection_lab (step4) と src/llcore (honest_eval) を import path に。read-only import。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.evolution.honest_eval import (  # noqa: E402
    _cliff_delta,  # = _paired_sign_delta (paired 符号バランス効果量)
    _paired_p,  # 片側 paired Wilcoxon (scipy 不在時は符号検定)
    honest_reevaluate,
)
from selection_lab import (  # noqa: E402
    _clip,
    map_elites,
    panmictic_ga,
    random_restart_hillclimb,
)


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_ensure_utf8_stdout()

# ---------------------------------------------------------------------------
# landscape: genotypic corridor + tunable dip depth d
# ---------------------------------------------------------------------------

D = 24  # exp4 と同一次元 (behavior=mean が CLT で b≈0.5 に固着 → random 不到達 corridor)
_NOISE = 0.008  # exp4/exp5 と同一低ノイズ
_LOCAL_H = 0.60
_GLOB_H = 1.00
_LOCAL_B = 0.40
_GLOB_B = 0.90
_LOCAL_W = 0.08
_GLOB_W = 0.06
_VALLEY_LO = 0.40  # 局所峰→大域峰を結ぶ ramp を張る behavior 区間
_VALLEY_HI = 0.90
_DIP_CENTER = 0.65  # 谷 (dip) を彫る中央位置 (局所/大域の中点付近)
_DIP_W = 0.07  # 谷の Gaussian 幅
_GLOBAL_PEAK_PROXY = 0.8  # honest fitness > これ で大域峰到達と判定 (exp4 と同一 proxy)


def behavior_mean(gene: np.ndarray) -> np.ndarray:
    """behavior = 全 dim の平均 (1D). 高 behavior = genotype 極値 = random 不到達 corridor.

    d に依存しない (corridor 構造を固定)。これにより random の失敗は全 d で定数。
    """
    return np.array([gene.mean()])


def make_corridor_eval(d: float) -> Callable[[np.ndarray, np.random.Generator], float]:
    """dip depth d の corridor fitness を返す.

    局所峰→大域峰を結ぶ単調登坂 ramp に、深さ d の谷 (dip) を彫る:
    - d=0: 谷無し → 厳密単調増加の登坂路 (smooth, exp5 相当)。hill-climb が大域へ登れる。
    - d=1: 谷の床≈0 の深い dip (exp4 deceptive)。hill-climb は downhill 拒否で罠。
    局所峰 (0.60@0.4) と大域峰 (1.0@0.9) は全 d で固定。
    """
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"d must be in [0,1], got {d}")

    def corridor_eval(gene: np.ndarray, rng: np.random.Generator) -> float:
        b = float(gene.mean())
        local = _LOCAL_H * np.exp(-((b - _LOCAL_B) ** 2) / (2 * _LOCAL_W ** 2))
        glob = _GLOB_H * np.exp(-((b - _GLOB_B) ** 2) / (2 * _GLOB_W ** 2))
        if _VALLEY_LO <= b <= _VALLEY_HI:
            t = (b - _VALLEY_LO) / (_VALLEY_HI - _VALLEY_LO)
            ramp = _LOCAL_H + t * (_GLOB_H - _LOCAL_H)  # 単調登坂 (局所峰→大域峰)
            dip = d * np.exp(-((b - _DIP_CENTER) ** 2) / (2 * _DIP_W ** 2))  # 谷
            corridor = ramp * (1.0 - dip)
        else:
            corridor = 0.0
        return float(max(local, glob, corridor) + rng.normal(0, _NOISE))

    return corridor_eval


# ---------------------------------------------------------------------------
# 完全 strict gate (honest_eval §5) — selection_lab.compare の上位互換
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    diff: float
    win_rate: float
    wilcoxon_p: float
    paired_sign_delta: float
    n_seeds: int
    passes: bool  # 完全 gate: diff>0 ∧ p<alpha ∧ n>=min_seeds ∧ |δ|>=min_effect


def strict_gate(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str,
    name_b: str,
    *,
    alpha: float = 0.05,
    min_seeds: int = 15,
    min_effect: float = 0.147,
) -> GateResult:
    """honest_eval §5 の完全 strict gate を a が b を上回るかに適用.

    selection_lab.compare は diff>0 ∧ p<alpha だけ。本 gate は監査 §5 の通り
    n_seeds>=min_seeds と |paired_sign_delta|>=min_effect も課す (overclaim 防止)。
    """
    deltas = scores_a - scores_b
    diff = float(np.mean(deltas))
    p = _paired_p(scores_a, scores_b)
    delta = _cliff_delta(deltas)  # paired_sign_delta
    passes = bool(
        diff > 0.0
        and p < alpha
        and len(scores_a) >= min_seeds
        and abs(delta) >= min_effect
    )
    return GateResult(
        name_a=name_a, name_b=name_b,
        mean_a=float(np.mean(scores_a)), mean_b=float(np.mean(scores_b)),
        diff=diff, win_rate=float(np.mean(scores_a >= scores_b)),
        wilcoxon_p=p, paired_sign_delta=delta, n_seeds=len(scores_a), passes=passes,
    )


# ---------------------------------------------------------------------------
# method runner: ea_lab.py の seed 設計 (SeedSequence + CRN) を踏襲
# ---------------------------------------------------------------------------

_METHODS = ("map_elites", "rr_hillclimb", "panmictic_ga", "random")


def run_methods_crn(
    eval_once: Callable[[np.ndarray, np.random.Generator], float],
    behavior: Callable[[np.ndarray], np.ndarray],
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    behavior_bounds: tuple[np.ndarray, np.ndarray],
    grid_shape: tuple[int, ...],
    n_evals: int,
    n_seeds: int,
    honest_n_trials: int,
    sigma: float,
    base_seed: int,
) -> dict[str, np.ndarray]:
    """MAP-Elites / RR-hillclimb / panmictic-GA / random を n_seeds で走らせ honest 再評価.

    selection_lab.run_methods_over_seeds と同じ 4 method を使うが、seed 設計を ea_lab.py に
    合わせて強化:
    - 進化 RNG = SeedSequence([base, method_idx, s]) で method×seed を一意・無相関化。
    - honest 再評価 RNG = SeedSequence([base, 7, s]) を **全 method 共通 (CRN)** に。
      → index s の 4 method が同一の fresh タスク draw で採点 = 真の matched replicate
      = paired Wilcoxon の前提充足 (selection_lab の元 runner は method 毎に別 seed だった)。
    """
    init_batch = max(20, n_evals // 10)
    out: dict[str, list[float]] = {m: [] for m in _METHODS}

    def _evo_rng(method_idx: int, s: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([base_seed, method_idx, s]))

    def _honest(gene: np.ndarray, s: int) -> float:
        # CRN: 全 method 共通 seed (method_idx を含めない)。
        return honest_reevaluate(
            eval_once, gene, n_trials=honest_n_trials,
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 7, s])),
        )

    for s in range(n_seeds):
        # MAP-Elites (①②③)
        r_me = map_elites(
            eval_once, behavior, dim=dim, bounds=bounds, behavior_bounds=behavior_bounds,
            grid_shape=grid_shape, n_evals=n_evals, init_batch=init_batch,
            sigma=sigma, rng=_evo_rng(0, s),
        )
        out["map_elites"].append(_honest(r_me.best_gene, s))

        # random-restart hill-climb (③なし強 baseline)
        r_rr = random_restart_hillclimb(
            eval_once, dim=dim, bounds=bounds, n_evals=n_evals, sigma=sigma,
            restart_patience=max(10, n_evals // 20), rng=_evo_rng(1, s),
        )
        out["rr_hillclimb"].append(_honest(r_rr.best_gene, s))

        # panmictic GA (①③, ②なし)
        r_ga = panmictic_ga(
            eval_once, dim=dim, bounds=bounds, n_evals=n_evals, pop_size=20,
            tournament_k=3, sigma=sigma, elitism=1, rng=_evo_rng(2, s),
        )
        out["panmictic_ga"].append(_honest(r_ga.best_gene, s))

        # pure random (同予算)
        rrng = _evo_rng(3, s)
        cands = [bounds[0] + (bounds[1] - bounds[0]) * rrng.random(dim) for _ in range(n_evals)]
        best = max(cands, key=lambda g: eval_once(g, rrng))
        out["random"].append(_honest(best, s))

    return {m: np.array(out[m]) for m in _METHODS}


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


@dataclass
class LevelResult:
    d: float  # dip depth knob
    valley_floor: float  # 谷の床の高さ = 0.60*(1-d)
    means: dict[str, float]
    reach_rate: dict[str, float]  # honest fitness > 0.8 (大域峰 proxy) の割合
    me_minus_best_baseline: float  # ③ advantage = MAP-E mean − best baseline mean
    best_baseline_name: str
    gates: dict[str, dict]  # MAP-E vs 各 baseline の完全 strict gate
    load_bearing: bool  # ③が立つ = 3 baseline 全てに strict gate 勝利


def run_sweep(
    *,
    d_levels: list[float],
    n_seeds: int = 20,
    n_evals: int = 6000,
    honest_n_trials: int = 30,
    sigma: float = 0.10,
    base_seed: int = 20260530,
) -> list[LevelResult]:
    bounds = (np.zeros(D), np.ones(D))
    behavior_bounds = (np.zeros(1), np.ones(1))
    grid_shape = (24,)
    baselines = ("rr_hillclimb", "panmictic_ga", "random")

    results: list[LevelResult] = []
    for d in d_levels:
        eval_once = make_corridor_eval(d)
        res = run_methods_crn(
            eval_once, behavior_mean, dim=D, bounds=bounds,
            behavior_bounds=behavior_bounds, grid_shape=grid_shape,
            n_evals=n_evals, n_seeds=n_seeds, honest_n_trials=honest_n_trials,
            sigma=sigma, base_seed=base_seed,
        )
        means = {m: float(res[m].mean()) for m in _METHODS}
        reach = {m: float(np.mean(res[m] > _GLOBAL_PEAK_PROXY)) for m in _METHODS}
        # best baseline = honest 再評価 mean が最大の baseline
        best_b = max(baselines, key=lambda b: means[b])
        advantage = means["map_elites"] - means[best_b]
        gates: dict[str, dict] = {}
        all_pass = True
        for b in baselines:
            g = strict_gate(res["map_elites"], res[b], "map_elites", b)
            gates[b] = asdict(g)
            all_pass = all_pass and g.passes
        results.append(LevelResult(
            d=d, valley_floor=_LOCAL_H * (1.0 - d), means=means, reach_rate=reach,
            me_minus_best_baseline=advantage, best_baseline_name=best_b,
            gates=gates, load_bearing=all_pass,
        ))
        print(f"d={d:.2f} (floor={_LOCAL_H*(1-d):.3f}): "
              f"MAP-E={means['map_elites']:.4f}(reach {reach['map_elites']:.2f}) "
              f"RR={means['rr_hillclimb']:.4f}(r {reach['rr_hillclimb']:.2f}) "
              f"GA={means['panmictic_ga']:.4f} rnd={means['random']:.4f} | "
              f"adv(vs {best_b})={advantage:+.4f} | "
              f"③load-bearing={'YES' if all_pass else 'no '}")
        for b in baselines:
            gg = gates[b]
            print(f"      vs {b:13s}: diff={gg['diff']:+.4f} p={gg['wilcoxon_p']:.3g} "
                  f"δ={gg['paired_sign_delta']:+.2f} pass={gg['passes']}")
    return results


def main() -> int:
    # >=6 levels: smooth(0.0) → deceptive(1.0)。閾値近傍を見るため非等間隔も含め 8 levels。
    d_levels = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    print("Step C 適用条件: dip depth d sweep (smooth d=0 → deceptive d=1)")
    print(f"D={D} behavior=mean genotypic corridor / n_seeds=20 n_evals=6000 "
          f"strict gate (p<0.05 ∧ |δ|>=0.147 ∧ n>=15)")
    print("=" * 88)
    results = run_sweep(d_levels=d_levels)
    print("=" * 88)

    # 閾値 d*: ③が初めて load-bearing になる最小 d
    threshold = None
    for r in results:
        if r.load_bearing:
            threshold = r.d
            break
    print(f"\n③ が load-bearing になり始める閾値 d* = "
          f"{'%.2f' % threshold if threshold is not None else 'なし (全 d で非 load-bearing)'}")

    # transition の sharpness: load-bearing 境界をまたぐ advantage の変化
    advs = [(r.d, r.me_minus_best_baseline, r.load_bearing) for r in results]
    print("transition (d, advantage, load_bearing):")
    for d, a, lb in advs:
        print(f"   d={d:.2f}: adv={a:+.4f} {'LB' if lb else '  '}")

    out_path = Path(__file__).resolve().parent / "exp_knob_sweep_results.json"
    payload = {
        "design": {
            "knob": "dip_depth_d",
            "knob_meaning": "valley floor height = 0.60*(1-d); d=0 smooth (monotone path), d=1 deep dip (exp4)",
            "d_levels": d_levels,
            "D": D, "n_seeds": 20, "n_evals": 6000, "honest_n_trials": 30,
            "sigma": 0.10, "base_seed": 20260530, "noise": _NOISE,
            "global_peak_proxy": _GLOBAL_PEAK_PROXY,
            "strict_gate": "diff>0 ∧ one-sided Wilcoxon p<0.05 ∧ n>=15 ∧ |paired_sign_delta|>=0.147",
        },
        "threshold_d_star": threshold,
        "levels": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
