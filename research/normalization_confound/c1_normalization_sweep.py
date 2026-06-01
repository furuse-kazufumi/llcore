# SPDX-License-Identifier: Apache-2.0
"""C1 midpoint-valley 正規化交絡 sweep — 「滑らか/③不要」は正規化の人工物か?

研究命題 P (falsifiable, DESIGN):
    variable_delay_recall (D∈{15,30,45,60}, distractor_amp=0.2, leaky-delay-line) の
    fitness 地形が「滑らか (C1 で連結 manifold = 単峰 = is_multimodal=False)」という
    現行 honest-negative は、fitness/状態スケール正規化が地形の多峰構造を消した
    『正規化人工物』ではないか。

検定 (否定形):
    baseline (clip=hard, sigma=0.15, 現行 bounds) で C1 が smooth なのに、ある正規化を
    外す (clip=none/soft, sigma 変更, bounds 拡大) と deceptive (valley_fraction>=0.2,
    is_multimodal=True) へ反転すれば → P 採択 = 人工物確定 (→③ablation escalation)。
    全条件で反転しなければ → P 棄却 = robust に滑らか (honest-negative を独立に追認)。

主診断 = C1 midpoint-valley (research/step_c_memory_tasks/landscape_map.py の
multimodality_report を **read-only 流用**)。raw gene 空間で random-restart 山登りし、
収束 optima 間の中点が fitness の谷か (= 分離 peak=多峰=欺瞞 vs 連結 manifold=滑らか) を
**地形幾何から直接**判定。behavior 記述子・grid binning・③の勝敗を一切参照しない (非循環)。

knob 割当 (DESIGN):
- A (clip, 最優先): clip ∈ {hard(=baseline), none(raw R²), soft(tanh)}。
  谷消失の第一容疑 (飽和で谷が床に埋まる) を直撃。
- C (状態スケール): sigma ∈ {0.10, 0.15(現行), 0.20}, bounds ∈ {現行, 拡大 leak_raw[-6,6]}。
  分解能の念押し (山登り 1 step が谷を跨いで見かけ単峰にしていないか)。
- B (記述子/binning): C1 は raw gene 空間で動くため **構造的に不変**。escalation 限定。

破綻ゲート (DESIGN):
- G1: CPU/numpy のみ・現実的時間。検出フェーズ全条件 < 30 分目標。
- G2: seed 固定で再現。base_seed 3 系列で valley_fraction の seed 間変動を報告。
      3 系列で is_multimodal が不一致なら『未確定』として反転と断定しない。
- G3: 主診断が degenerate でない。n_optima>=2 / pairs>0 / R² finite を確認。
      pairs==0 は『判定不能』として smooth とも deceptive とも結論しない。
- G4: 判定は valley_fraction と raw R² spread のみ。③の diff/p/passes を一切参照しない。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- research 隔離: sibling research ディレクトリを import path に載せる (filename import) ---
_HERE = Path(__file__).resolve().parent
_LLCORE_ROOT = _HERE.parents[1]  # research/normalization_confound -> research -> llcore
_RESEARCH = _LLCORE_ROOT / "research"
for _p in (
    _RESEARCH / "step_c_memory_tasks",       # landscape_map, reservoir, memory_tasks
    _RESEARCH / "ea_multitask" / "candidates",  # variable_delay_recall
    _RESEARCH / "ea_multitask",              # task_mixture
    str(_HERE),
):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# read-only 流用 (改変しない)
from landscape_map import multimodality_report  # noqa: E402  (主診断, src 非依存)
from reservoir import LeakyDelayLineReservoir, gene_bounds  # noqa: E402
from variable_delay_recall import VariableDelayRecallTask, make_regimes  # noqa: E402
from memory_tasks import FlipFlopTask  # noqa: E402  (感度確認用 第二候補)
from task_mixture import TaskMixture  # noqa: E402

from c1_clip_eval import make_eval_once_clipswitch  # noqa: E402  (clip 切替版 eval_once)


def _ensure_utf8_stdout() -> None:
    """Windows cp932 console でも UTF-8 で日本語/em-dash を出力する (CLAUDE.md 規律)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - 既に utf-8 / 非対応 stream
            pass


# --------------------------------------------------------------------------- #
# C knob 用の bounds 拡大版 (src 非改変。gene_bounds の戻り値を呼出側で差し替え)
# --------------------------------------------------------------------------- #
def widened_bounds(res: LeakyDelayLineReservoir) -> tuple[np.ndarray, np.ndarray]:
    """leak_raw を [-6,6] に拡大した bounds (時定数レンジ拡大). w_in は現行 [-2,2] 維持."""
    lo = np.concatenate([
        np.full(res.n_taps, -6.0),
        np.full(res.n_taps * res.in_dim, -2.0),
    ])
    hi = np.concatenate([
        np.full(res.n_taps, 6.0),
        np.full(res.n_taps * res.in_dim, 2.0),
    ])
    return lo, hi


# --------------------------------------------------------------------------- #
# 副次診断: 収束 optima の raw R² spread (clip 飽和の直接証拠)
#   landscape_map._hillclimb と同一手順で optima を集め、その raw fitness の散らばりを測る。
#   (multimodality_report は spread を返さないため、同手順を最小再実装する。clip 切替の
#    eval_once を渡すので clip=hard なら床に潰れ、clip=none なら広がるはず。)
# --------------------------------------------------------------------------- #
def _hillclimb_collect_optima(eval_once, *, dim, bounds, n_restarts, n_evals, sigma, base_seed):
    """n_restarts 回 hill-climb し、収束 optima の (gene, fitness) を集める."""
    lo, hi = bounds
    optima = []
    for i in range(n_restarts):
        rng = np.random.default_rng(base_seed + i)
        g = lo + (hi - lo) * rng.random(dim)
        f = eval_once(g, rng)
        for _ in range(n_evals - 1):
            cand = np.clip(g + rng.normal(0, sigma, size=dim), lo, hi)
            cf = eval_once(cand, rng)
            if cf >= f:
                g, f = cand, cf
        optima.append((g, float(f)))
    return optima


def raw_r2_spread(eval_once, *, dim, bounds, n_restarts, n_evals, sigma, base_seed):
    """収束 optima の fitness 値の散らばり (max-min, std, min, max, finite 率)."""
    optima = _hillclimb_collect_optima(
        eval_once, dim=dim, bounds=bounds, n_restarts=n_restarts,
        n_evals=n_evals, sigma=sigma, base_seed=base_seed,
    )
    fits = np.array([f for _, f in optima], dtype=np.float64)
    finite = np.isfinite(fits)
    finite_rate = float(np.mean(finite))
    ff = fits[finite] if finite.any() else fits
    return {
        "n_optima": len(optima),
        "spread": float(ff.max() - ff.min()) if ff.size else float("nan"),
        "std": float(ff.std()) if ff.size else float("nan"),
        "min": float(ff.min()) if ff.size else float("nan"),
        "max": float(ff.max()) if ff.size else float("nan"),
        "finite_rate": finite_rate,
    }


# --------------------------------------------------------------------------- #
# ノイズ検証 (budget_sensitivity_check の決定的所見を harness に内蔵)
#   所見: C1 谷判定の閾値 0.05*fit << 評価ノイズ std。flat+noise を C1 にかけると
#   valley_fraction≈1.0 (偽陽性)、noiseless 単峰は 0.0 (真陰性)。よって stochastic な
#   ridge fitness では「deceptive」verdict が評価ノイズの人工物でないことを **noisy-flat
#   null と比較して** 確認しない限り信用できない。verdict を noise-aware にする。
# --------------------------------------------------------------------------- #
def measure_eval_noise(eval_once, *, dim, bounds, K=16, base_seed=20260530):
    """ランダム gene 1 個を K 回 (異 seed) 評価し fitness 評価ノイズ std を測る."""
    lo, hi = bounds
    g = lo + (hi - lo) * np.random.default_rng(base_seed).random(dim)
    vals = np.array([eval_once(g, np.random.default_rng(base_seed + 5000 + k))
                     for k in range(K)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan"), \
        float(vals.std()) if vals.size else float("nan")


def noisy_flat_null_valley(noise_std, *, dim, bounds, n_restarts, n_evals, sigma,
                           base_seed, mean=0.5):
    """同じ評価ノイズ std を持つ **flat** landscape の C1 valley_fraction (null 分布).

    実 eval と同 (n_restarts/n_evals/sigma) で測る。これが C1 の『ノイズだけで出る谷』の
    基準線。実設定の valley_fraction がこの null を有意に超えない限り deceptive と
    断定してはならない。
    """
    rng_const = np.random.default_rng(base_seed)  # noqa: F841 (mean は固定)

    def flat_eval(g, rng):
        return float(rng.normal(mean, noise_std))

    rep = multimodality_report(flat_eval, dim=dim, bounds=bounds,
                               n_restarts=n_restarts, n_evals=n_evals,
                               sigma=sigma, base_seed=base_seed)
    return float(rep["valley_fraction"])


# --------------------------------------------------------------------------- #
# タスク/基質ファクトリ
# --------------------------------------------------------------------------- #
@dataclass
class TaskSpec:
    name: str
    task: object
    in_dim: int


def build_tasks() -> dict[str, TaskSpec]:
    """C1 を測る対象タスク群を作る (DESIGN: vdr 主, flip_flop 感度確認, mixture)."""
    vdr_regimes = make_regimes(delays=(15, 30, 45, 60), distractor_amp=0.2, in_dim=2)
    # vdr の D は make_regimes の順 (15,30,45,60)
    by_d = {t.seq_len: t for t in vdr_regimes}
    tasks: dict[str, TaskSpec] = {
        "vdr_D60": TaskSpec("vdr_D60", by_d[60], in_dim=2),       # 主 (最難 regime)
        "vdr_D15": TaskSpec("vdr_D15", by_d[15], in_dim=2),       # 対照 (最易 regime)
        "vdr_mixture": TaskSpec(                                  # train mixture (全 regime)
            "vdr_mixture", TaskMixture(regimes=vdr_regimes), in_dim=2),
        "flip_flop": TaskSpec("flip_flop", FlipFlopTask(seq_len=30, in_dim=2), in_dim=2),
    }
    return tasks


# --------------------------------------------------------------------------- #
# 1 設定 (task × clip × sigma × bounds) の C1 + spread を測る
# --------------------------------------------------------------------------- #
@dataclass
class SettingResult:
    setting: str
    task: str
    clip: str
    sigma: float
    bounds: str
    n_train: int
    n_eval: int
    n_restarts: int
    n_evals: int
    # per-seed C1
    valley_fractions: list[float] = field(default_factory=list)
    is_multimodal_flags: list[bool] = field(default_factory=list)
    n_optima_list: list[int] = field(default_factory=list)
    # 集約
    valley_mean: float = float("nan")
    valley_std: float = float("nan")
    is_multimodal_majority: bool = False
    is_multimodal_unanimous: bool = False  # 全 seed 一致 (G2)
    verdict: str = "undetermined"
    # 副次: raw R² spread (代表 seed)
    spread: dict = field(default_factory=dict)
    degenerate: bool = False  # G3: pairs==0 等で判定不能だった seed があるか
    # ノイズ検証 (本研究の主要 caveat)
    eval_noise_std: float = float("nan")     # fitness 評価ノイズ std
    valley_threshold: float = float("nan")   # C1 谷判定の有効閾 ~0.05*|fit|
    noise_dominated: bool = False            # eval_noise_std > valley_threshold か
    noisy_flat_null_vf: float = float("nan") # 同ノイズの flat landscape の valley_fraction (null)
    note: str = ""


def run_setting(
    *, task_spec: TaskSpec, clip: str, sigma: float, bounds_kind: str,
    seeds: list[int], n_restarts: int, n_evals: int, n_train: int, n_eval: int,
) -> SettingResult:
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=task_spec.in_dim)
    bounds = widened_bounds(res) if bounds_kind == "wide" else gene_bounds(res)
    eval_once = make_eval_once_clipswitch(
        res, task_spec.task, n_train=n_train, n_eval=n_eval, clip=clip)
    setting_id = f"{task_spec.name}|clip={clip}|sigma={sigma}|bounds={bounds_kind}"

    r = SettingResult(
        setting=setting_id, task=task_spec.name, clip=clip, sigma=sigma,
        bounds=bounds_kind, n_train=n_train, n_eval=n_eval,
        n_restarts=n_restarts, n_evals=n_evals,
    )
    degenerate_any = False
    for sd in seeds:
        rep = multimodality_report(
            eval_once, dim=res.gene_dim, bounds=bounds,
            n_restarts=n_restarts, n_evals=n_evals, sigma=sigma, base_seed=sd,
        )
        r.valley_fractions.append(float(rep["valley_fraction"]))
        r.is_multimodal_flags.append(bool(rep["is_multimodal"]))
        r.n_optima_list.append(int(rep["n_optima"]))
        # G3: collapse 検知 — n_optima<2 なら必ず判定不能。valley_fraction==0.0 でも
        # multimodality_report は pairs を返さないため、spread 側で pairs を別途見る。
        if int(rep["n_optima"]) < 2:
            degenerate_any = True

    r.valley_mean = float(np.mean(r.valley_fractions)) if r.valley_fractions else float("nan")
    r.valley_std = float(np.std(r.valley_fractions)) if r.valley_fractions else float("nan")
    n_mm = sum(r.is_multimodal_flags)
    r.is_multimodal_majority = n_mm > len(r.is_multimodal_flags) / 2
    r.is_multimodal_unanimous = (n_mm == len(r.is_multimodal_flags)) or (n_mm == 0)

    # 副次 raw R² spread (代表 = 先頭 seed)
    r.spread = raw_r2_spread(
        eval_once, dim=res.gene_dim, bounds=bounds,
        n_restarts=n_restarts, n_evals=n_evals, sigma=sigma, base_seed=seeds[0],
    )
    # G3: finite 率 / n_optima
    if r.spread.get("finite_rate", 1.0) < 1.0 or r.spread.get("n_optima", 0) < 2:
        degenerate_any = True
    r.degenerate = degenerate_any

    # --- ノイズ検証 (本研究の主要 caveat。budget_sensitivity_check の決定的所見を内蔵) ---
    fit_mean, noise_std = measure_eval_noise(
        eval_once, dim=res.gene_dim, bounds=bounds, base_seed=seeds[0])
    r.eval_noise_std = noise_std
    r.valley_threshold = 0.05 * (abs(fit_mean) + 1e-9)
    r.noise_dominated = bool(np.isfinite(noise_std) and noise_std > r.valley_threshold)
    # 同ノイズの flat landscape を C1 にかけた null (ノイズだけで出る谷の基準線)
    r.noisy_flat_null_vf = noisy_flat_null_valley(
        noise_std if np.isfinite(noise_std) else 0.0,
        dim=res.gene_dim, bounds=bounds, n_restarts=n_restarts, n_evals=n_evals,
        sigma=sigma, base_seed=seeds[0])

    # verdict (G2/G3 + noise-aware):
    #  - degenerate → undetermined
    #  - seed 間 is_multimodal 不一致 → undetermined (反転と断定しない)
    #  - **noise_dominated かつ valley_mean が noisy-flat null を上回らない →
    #     noise_confounded** (deceptive と断定しない=本研究の honest 判断)
    #  - 全 seed unimodal → smooth
    #  - 全 seed multimodal かつ null を margin>=0.1 で上回る → deceptive
    margin = r.valley_mean - r.noisy_flat_null_vf
    if degenerate_any:
        r.verdict = "undetermined"
        r.note = "degenerate (n_optima<2 or non-finite R²)"
    elif not r.is_multimodal_unanimous:
        r.verdict = "undetermined"
        r.note = f"seed-inconsistent is_multimodal ({n_mm}/{len(r.is_multimodal_flags)})"
    elif n_mm == 0:
        r.verdict = "smooth"      # 連結 manifold = 単峰 (谷ゼロ)
    elif r.noise_dominated and margin < 0.1:
        # 谷が出ても、同ノイズの flat null と区別できない → ノイズ起源と判断
        r.verdict = "noise_confounded"
        r.note = (f"valley_fraction({r.valley_mean:.2f}) ~ noisy-flat null"
                  f"({r.noisy_flat_null_vf:.2f}); noise_std({noise_std:.3f}) "
                  f">> valley_thr({r.valley_threshold:.4f}). 谷はノイズ人工物の疑い")
    elif n_mm == len(r.is_multimodal_flags) and margin >= 0.1:
        r.verdict = "deceptive"   # null を有意に超える分離 peak
        r.note = f"valley_fraction exceeds noisy-flat null by margin={margin:.2f}"
    else:
        r.verdict = "undetermined"
        r.note = (f"multimodal but within noise null (margin={margin:.2f}); "
                  "cannot confirm geometric valley")
    return r


# --------------------------------------------------------------------------- #
# sweep 本体
# --------------------------------------------------------------------------- #
def build_grid(*, full: bool):
    """sweep する (clip, sigma, bounds) 格子を返す.

    baseline = (clip=hard, sigma=0.15, bounds=current)。
    A: clip ∈ {hard, none, soft} (sigma/bounds は baseline 固定で A を孤立評価)。
    C: sigma ∈ {0.10,0.15,0.20} × bounds ∈ {current,wide} (clip は baseline=hard 固定)。
    全格子 (full) は A×C を組合せず、A 軸と C 軸を baseline 起点の十字で取る (CPU 予算節約)。
    """
    grid = []
    # baseline
    grid.append(("hard", 0.15, "current"))
    # A 軸 (clip を振る; sigma/bounds は baseline)
    grid.append(("none", 0.15, "current"))
    grid.append(("soft", 0.15, "current"))
    # C 軸 (sigma を振る; clip=hard, bounds=current)
    grid.append(("hard", 0.10, "current"))
    grid.append(("hard", 0.20, "current"))
    # C 軸 (bounds 拡大; clip=hard, sigma=baseline)
    grid.append(("hard", 0.15, "wide"))
    if full:
        # 交互作用の最小確認: clip=none と bounds=wide / sigma 端を組合せ
        grid.append(("none", 0.15, "wide"))
        grid.append(("none", 0.10, "current"))
        grid.append(("none", 0.20, "current"))
    # 重複除去 (順序保持)
    seen = set()
    uniq = []
    for g in grid:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq


def main() -> None:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="C1 normalization-confound sweep (vdr)")
    ap.add_argument("--mode", choices=["smoke", "detect", "confirm"], default="detect",
                    help="smoke=tiny / detect=検出 (n_restarts=8) / confirm=確認 (n_restarts>=15)")
    ap.add_argument("--tasks", default="vdr_D60,vdr_D15",
                    help="comma-separated: vdr_D60,vdr_D15,vdr_mixture,flip_flop")
    ap.add_argument("--full-grid", action="store_true",
                    help="A×C 交互作用条件も含める (CPU 予算増)")
    ap.add_argument("--out", default=str(_HERE / "c1_sweep_results.json"))
    args = ap.parse_args()

    # mode 別パラメータ (G1 予算管理)
    # microbench (budget_sensitivity_check): D60 eval ~55-65ms。C1 report = n_restarts*n_evals
    # eval 呼出。n_evals=400 は 1 report ~210s → 全 sweep 30 分超 (G1 違反)。よって detect は
    # n_evals=150 に縮小 (budget_sensitivity §2 で valley_fraction が n_evals 150 vs 300 で
    # 0.893 同値 = 縮小が valley_fraction を 0.05 超動かさないことを確認済 → G1 の縮小条件を満たす)。
    if args.mode == "smoke":
        seeds = [20260530, 20260531]
        n_restarts, n_evals, n_train, n_eval = 4, 120, 24, 24
    elif args.mode == "detect":
        seeds = [20260530, 20260531, 20260601]   # G2: 3 seed 系列
        n_restarts, n_evals, n_train, n_eval = 8, 150, 48, 48
    else:  # confirm
        seeds = [20260530, 20260531, 20260601]
        n_restarts, n_evals, n_train, n_eval = 16, 150, 48, 48

    all_tasks = build_tasks()
    sel = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_specs = []
    for name in sel:
        if name not in all_tasks:
            raise SystemExit(f"unknown task {name!r}; available={list(all_tasks)}")
        task_specs.append(all_tasks[name])

    grid = build_grid(full=args.full_grid)

    print(f"=== C1 正規化交絡 sweep (mode={args.mode}) ===")
    print(f"tasks={[t.name for t in task_specs]} seeds={seeds}")
    print(f"n_restarts={n_restarts} n_evals={n_evals} n_train={n_train} n_eval={n_eval}")
    print(f"grid (clip,sigma,bounds) x{len(grid)}: {grid}")
    print()

    results: list[SettingResult] = []
    t_start = time.time()
    for tspec in task_specs:
        # baseline verdict (clip=hard,sigma=0.15,current) をタスクごとに保持
        baseline_verdict = None
        for (clip, sigma, bounds_kind) in grid:
            t0 = time.time()
            r = run_setting(
                task_spec=tspec, clip=clip, sigma=sigma, bounds_kind=bounds_kind,
                seeds=seeds, n_restarts=n_restarts, n_evals=n_evals,
                n_train=n_train, n_eval=n_eval,
            )
            dt = time.time() - t0
            results.append(r)
            if (clip, sigma, bounds_kind) == ("hard", 0.15, "current"):
                baseline_verdict = r.verdict
            print(f"[{r.setting}] verdict={r.verdict} "
                  f"valley_mean={r.valley_mean:.3f} "
                  f"null_vf={r.noisy_flat_null_vf:.3f} "
                  f"noise_std={r.eval_noise_std:.3f}(thr={r.valley_threshold:.4f}) "
                  f"noise_dom={r.noise_dominated} "
                  f"mm_flags={r.is_multimodal_flags} "
                  f"spread={r.spread.get('spread', float('nan')):.4f} "
                  f"R2[{r.spread.get('min', float('nan')):.3f},{r.spread.get('max', float('nan')):.3f}] "
                  f"({dt:.1f}s)")
        # タスク内 flip 判定
        if baseline_verdict in ("smooth", "noise_confounded"):
            for r in results:
                if r.task == tspec.name and r.verdict == "deceptive":
                    print(f"  >> FLIP[{tspec.name}]: baseline {baseline_verdict} -> {r.setting} deceptive")
    total_dt = time.time() - t_start
    print(f"\ntotal wall-clock = {total_dt:.1f}s")

    # ---- any_flip 判定 (タスクごと baseline=smooth → 同タスクの別設定で deceptive) ----
    by_task: dict[str, list[SettingResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)
    flip_settings = []
    for task, rs in by_task.items():
        base = next((x for x in rs
                     if x.clip == "hard" and abs(x.sigma - 0.15) < 1e-9 and x.bounds == "current"),
                    None)
        if base is None or base.verdict != "smooth":
            continue
        for x in rs:
            if x is base:
                continue
            if x.verdict == "deceptive":
                flip_settings.append(x.setting)
    any_flip = len(flip_settings) > 0

    # clip 飽和の副次裏付け: clip=hard vs clip=none の spread 比 (同 task,sigma,bounds)
    spread_ratios = []
    for task, rs in by_task.items():
        hard = next((x for x in rs if x.clip == "hard" and abs(x.sigma - 0.15) < 1e-9
                     and x.bounds == "current"), None)
        none = next((x for x in rs if x.clip == "none" and abs(x.sigma - 0.15) < 1e-9
                     and x.bounds == "current"), None)
        if hard and none:
            sh = hard.spread.get("spread", float("nan"))
            sn = none.spread.get("spread", float("nan"))
            ratio = (sn / sh) if (sh and np.isfinite(sh) and sh > 1e-9) else float("inf")
            spread_ratios.append({"task": task, "spread_hard": sh, "spread_none": sn,
                                  "ratio_none_over_hard": ratio})

    payload = {
        "proposition": "P: smooth/③不要 は正規化(clip/scale)の人工物か (C1 midpoint-valley で否定形検定)",
        "mode": args.mode,
        "seeds": seeds,
        "n_restarts": n_restarts, "n_evals": n_evals,
        "n_train": n_train, "n_eval": n_eval,
        "grid": [{"clip": c, "sigma": s, "bounds": b} for (c, s, b) in grid],
        "total_wall_clock_s": total_dt,
        "any_flip": any_flip,
        "flip_settings": flip_settings,
        "spread_ratios_clip_none_over_hard": spread_ratios,
        "results": [
            {
                "setting": r.setting, "task": r.task, "clip": r.clip,
                "sigma": r.sigma, "bounds": r.bounds, "verdict": r.verdict,
                "valley_mean": r.valley_mean, "valley_std": r.valley_std,
                "valley_fractions": r.valley_fractions,
                "is_multimodal_flags": r.is_multimodal_flags,
                "is_multimodal_unanimous": r.is_multimodal_unanimous,
                "n_optima_list": r.n_optima_list,
                "spread": r.spread, "degenerate": r.degenerate, "note": r.note,
            }
            for r in results
        ],
        "verdict_summary": (
            "any_flip=True → P 採択 (人工物確定、③ablation escalation へ)"
            if any_flip else
            "any_flip=False → P 棄却 (robust に滑らか / honest-negative を独立追認)"
        ),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=> any_flip={any_flip} flip_settings={flip_settings}")
    print(f"   spread_ratios(none/hard)={[round(x['ratio_none_over_hard'],2) if np.isfinite(x['ratio_none_over_hard']) else 'inf' for x in spread_ratios]}")
    print(f"   wrote {args.out}")


if __name__ == "__main__":
    main()
