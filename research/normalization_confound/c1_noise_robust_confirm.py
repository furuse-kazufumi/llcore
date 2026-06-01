# SPDX-License-Identifier: Apache-2.0
"""any_flip=False の補強 — 「smooth/③不要」は正規化の人工物ではない を honest に固める.

背景 (sweep 結論):
    c1_normalization_sweep.py の 12 設定 (vdr_D60/D15 × clip{hard,none,soft} ×
    sigma{0.10,0.15,0.20} × bounds{current,wide}) は **全て any_flip=False で
    verdict=noise_confounded**。どの正規化を外しても deceptive に反転しない。
    決定的所見 = C1 谷判定閾値 0.05*|fit| (≈0.012-0.039) << fitness 評価ノイズ std
    (0.036-0.21)。budget_sensitivity_check の control が noisy-flat → valley_fraction=1.0
    (偽陽性)、noiseless 単峰 → 0.0 (真陰性) を実証済。

honest correction (この harness の核心):
    任務 branch (any_flip=False) の素案は「smooth のままか確認」だが、**C1 は本 fitness で
    一度も smooth を返していない** (常に noise_confounded)。よって正しい補強は
    「smooth の追認」ではなく「**noise-confound が全正規化・全予算で頑健**であり、
    かつ noise を平均で潰すと谷が noisy-flat null に収束する=幾何ではなくノイズ起源」を
    falsifiable に示すこと。

falsifiable 命題 (この harness):
    P_robust : 既存 C1 の valley≈1.0 は評価ノイズ起源である。
    予測 1 (反証可能): optima と midpoint を **同じ n_avg seed で平均** (CRN-paired) し、
      谷閾を **SEM (noise_std/√n_avg)** でスケールした noise-robust C1 にすると、
      n_avg を上げるほど valley_fraction は **noisy-flat null と同じ軌道**で減衰する。
      → もし実 vdr が幾何的に多峰なら、noise を平均で潰しても valley_fraction は
         null より高い plateau に残る (幾何の谷は averaging で消えない)。これが
         観測されれば P_robust は反証され「真の谷 (人工物でない)」となる。
    予測 2: raw R² spread を n_avg で振っても optima R² 範囲 (~[0.4,0.83] D60) は
      [0,1] 床に届かず、clip 飽和 (交絡 A) は vdr で非 load-bearing のまま。

非循環性 (G4): 判定は valley_fraction / noisy-flat null / R² spread のみ。
③ablation の diff/p/passes を一切参照しない。CRN は同 base_seed の derived seed 列で
実 eval と null を paired にする (共通乱数=分散低減)。

src 非改変・git なし・UTF-8・py -3.11。reservoir / variable_delay_recall / src fit_ridge
を read-only import。c1_clip_eval.make_eval_once_clipswitch を流用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LLCORE_ROOT = _HERE.parents[1]
_RESEARCH = _LLCORE_ROOT / "research"
for _p in (
    _RESEARCH / "step_c_memory_tasks",
    _RESEARCH / "ea_multitask" / "candidates",
    _RESEARCH / "ea_multitask",
    str(_HERE),
):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from reservoir import LeakyDelayLineReservoir, gene_bounds  # noqa: E402
from variable_delay_recall import make_regimes  # noqa: E402
from c1_clip_eval import make_eval_once_clipswitch  # noqa: E402


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            pass


def widened_bounds(res):
    lo = np.concatenate([np.full(res.n_taps, -6.0),
                         np.full(res.n_taps * res.in_dim, -2.0)])
    hi = np.concatenate([np.full(res.n_taps, 6.0),
                         np.full(res.n_taps * res.in_dim, 2.0)])
    return lo, hi


# --------------------------------------------------------------------------- #
# noise-robust hill-climb: optima を n_avg seed 平均 fitness で評価する
# (既存 landscape_map._hillclimb は単一 seed で f を取る → ノイズで谷が出る)
# --------------------------------------------------------------------------- #
def _avg_eval(eval_once, gene, *, n_avg, base_seed, tag):
    """gene を n_avg 個の独立 seed で評価した fitness 平均 (CRN: tag で seed 系列を分ける)."""
    vals = [eval_once(gene, np.random.default_rng(base_seed + tag * 100003 + k))
            for k in range(n_avg)]
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    return float(vals.mean()) if vals.size else float("nan")


def _hillclimb_avg(eval_once, *, dim, bounds, n_evals, sigma, n_avg, base_seed):
    """n_avg seed 平均 fitness で山登り (各候補も平均評価 → ノイズ低減した optima)."""
    lo, hi = bounds
    rng = np.random.default_rng(base_seed)
    g = lo + (hi - lo) * rng.random(dim)
    f = _avg_eval(eval_once, g, n_avg=n_avg, base_seed=base_seed, tag=0)
    for step in range(1, n_evals):
        cand = np.clip(g + rng.normal(0, sigma, size=dim), lo, hi)
        cf = _avg_eval(eval_once, cand, n_avg=n_avg, base_seed=base_seed, tag=step)
        if cf >= f:
            g, f = cand, cf
    return g, f


def noise_robust_c1(eval_once, *, dim, bounds, n_restarts, n_evals, sigma,
                    n_avg, noise_std, base_seed):
    """noise-robust C1: optima/midpoint を n_avg seed 平均で評価し、谷閾を SEM scale.

    既存 C1 との差分:
      - optima endpoint fi,fj も n_avg seed 平均 (既存は単一 seed)。
      - midpoint fm も同じ n_avg seed 平均 (既存は 3 seed 固定)。
      - 谷閾 = max(0.05*|min|, k_sem * SEM)。SEM = noise_std/√n_avg。
        ノイズの 95%CI 相当 (k_sem≈2) を超えた時のみ「谷」と数える=ノイズ偽陽性を抑止。
    """
    lo, hi = bounds
    optima = []
    for i in range(n_restarts):
        g, f = _hillclimb_avg(eval_once, dim=dim, bounds=bounds, n_evals=n_evals,
                              sigma=sigma, n_avg=n_avg, base_seed=base_seed + i)
        optima.append((g, f))
    sem = (noise_std / np.sqrt(max(n_avg, 1))) if np.isfinite(noise_std) else 0.0
    k_sem = 2.0  # 中点 vs 端点 差の有意性: ~2 SEM (片側 ~97.7%)
    valley = 0
    pairs = 0
    for i in range(len(optima)):
        for j in range(i + 1, len(optima)):
            gi, fi = optima[i]
            gj, fj = optima[j]
            if np.allclose(gi, gj, atol=1e-2) or not (np.isfinite(fi) and np.isfinite(fj)):
                continue
            mid = 0.5 * (gi + gj)
            fm = _avg_eval(eval_once, mid, n_avg=n_avg, base_seed=base_seed + 777, tag=i * 31 + j)
            if not np.isfinite(fm):
                continue
            pairs += 1
            fmin = min(fi, fj)
            # 谷閾: 相対 0.05*|fmin| と 絶対 k_sem*SEM の大きい方 (ノイズ床を必ず超える要求)
            thr = max(0.05 * (abs(fmin) + 1e-9), k_sem * sem)
            if fm < fmin - thr:
                valley += 1
    frac = valley / pairs if pairs else 0.0
    return {"n_optima": len(optima), "pairs": pairs, "valley_fraction": frac,
            "is_multimodal": frac >= 0.2, "sem": float(sem),
            "optima_fits": [float(f) for _, f in optima]}


def noisy_flat_null_robust(noise_std, *, dim, bounds, n_restarts, n_evals, sigma,
                           n_avg, base_seed, mean=0.5):
    """同 noise_std の flat landscape を noise-robust C1 にかけた null (ノイズだけの谷の基準)."""
    def flat_eval(g, rng):
        return float(rng.normal(mean, noise_std))
    return noise_robust_c1(flat_eval, dim=dim, bounds=bounds, n_restarts=n_restarts,
                           n_evals=n_evals, sigma=sigma, n_avg=n_avg,
                           noise_std=noise_std, base_seed=base_seed)["valley_fraction"]


def measure_eval_noise(eval_once, *, dim, bounds, K=20, base_seed=20260530):
    lo, hi = bounds
    g = lo + (hi - lo) * np.random.default_rng(base_seed).random(dim)
    vals = np.array([eval_once(g, np.random.default_rng(base_seed + 5000 + k))
                     for k in range(K)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return (float(vals.mean()) if vals.size else float("nan"),
            float(vals.std()) if vals.size else float("nan"))


@dataclass
class NAvgResult:
    setting: str
    task: str
    clip: str
    bounds: str
    n_avg: int
    noise_std: float = float("nan")
    sem: float = float("nan")
    valley_fractions: list[float] = field(default_factory=list)
    valley_mean: float = float("nan")
    mean_pairs: float = float("nan")        # C1 谷判定の分母 (低統計検知)
    null_vf: float = float("nan")
    margin: float = float("nan")            # valley_mean - null_vf
    optima_r2_min: float = float("nan")
    optima_r2_max: float = float("nan")
    verdict: str = "undetermined"
    note: str = ""


def run_navg_point(*, res, task, task_name, clip, bounds_kind, n_avg, seeds, n_restarts,
                   n_evals, sigma, n_train, n_eval):
    bounds = widened_bounds(res) if bounds_kind == "wide" else gene_bounds(res)
    eval_once = make_eval_once_clipswitch(res, task, n_train=n_train, n_eval=n_eval, clip=clip)
    setting = f"{task_name}|clip={clip}|bounds={bounds_kind}|n_avg={n_avg}"
    r = NAvgResult(setting=setting, task=task_name, clip=clip, bounds=bounds_kind, n_avg=n_avg)
    _, noise_std = measure_eval_noise(eval_once, dim=res.gene_dim, bounds=bounds,
                                      base_seed=seeds[0])
    r.noise_std = noise_std
    r.sem = (noise_std / np.sqrt(max(n_avg, 1))) if np.isfinite(noise_std) else float("nan")
    r2_lo, r2_hi, pairs_list = [], [], []
    for sd in seeds:
        rep = noise_robust_c1(eval_once, dim=res.gene_dim, bounds=bounds,
                              n_restarts=n_restarts, n_evals=n_evals, sigma=sigma,
                              n_avg=n_avg, noise_std=noise_std, base_seed=sd)
        r.valley_fractions.append(rep["valley_fraction"])
        pairs_list.append(rep["pairs"])
        ofits = [f for f in rep["optima_fits"] if np.isfinite(f)]
        if ofits:
            r2_lo.append(min(ofits))
            r2_hi.append(max(ofits))
    r.valley_mean = float(np.mean(r.valley_fractions)) if r.valley_fractions else float("nan")
    r.mean_pairs = float(np.mean(pairs_list)) if pairs_list else 0.0
    r.optima_r2_min = float(min(r2_lo)) if r2_lo else float("nan")
    r.optima_r2_max = float(max(r2_hi)) if r2_hi else float("nan")
    # null は flat だが pair 数が少ないと量子化粗いので **実 eval と同じ seed 集合で平均** (CRN)。
    null_vfs = [noisy_flat_null_robust(noise_std if np.isfinite(noise_std) else 0.0,
                                       dim=res.gene_dim, bounds=bounds,
                                       n_restarts=n_restarts, n_evals=min(n_evals, 60),
                                       sigma=sigma, n_avg=n_avg, base_seed=sd)
                for sd in seeds]
    r.null_vf = float(np.mean(null_vfs))
    r.margin = r.valley_mean - r.null_vf
    # verdict (noise-robust, 低統計に頑健化):
    #  geometric_valley は (a) valley>=0.2 (b) margin>=0.15 (c) 十分な pair 数 (mean_pairs>=8)
    #  の全てを要求。pair が少ない (mean_pairs<8) と valley_fraction の量子化が粗く 1 ペアで
    #  margin が跳ねるため、その場合は geometric と断定しない (low_stats)。
    #  - 谷も null も低い (<0.2) → smooth_after_denoise (averaging で谷消失=ノイズ起源確証)
    #  - それ以外 (谷が null と同程度 or low_stats) → noise_confounded (頑健に維持)
    if r.valley_mean >= 0.2 and r.margin >= 0.15 and r.mean_pairs >= 8.0:
        r.verdict = "geometric_valley"
        r.note = (f"valley({r.valley_mean:.2f}) exceeds denoised null({r.null_vf:.2f}) "
                  f"margin={r.margin:.2f}, mean_pairs={r.mean_pairs:.1f}")
    elif r.valley_mean < 0.2 and r.null_vf < 0.2:
        r.verdict = "smooth_after_denoise"
        r.note = f"averaging で谷消失 (valley={r.valley_mean:.2f}, null={r.null_vf:.2f})=ノイズ起源"
    elif r.valley_mean >= 0.2 and r.margin >= 0.15 and r.mean_pairs < 8.0:
        r.verdict = "noise_confounded"
        r.note = (f"valley>null だが low_stats (mean_pairs={r.mean_pairs:.1f}<8); "
                  f"幾何谷と断定不可。margin={r.margin:.2f}")
    else:
        r.verdict = "noise_confounded"
        r.note = (f"valley({r.valley_mean:.2f}) ~ null({r.null_vf:.2f}) margin={r.margin:.2f}; "
                  f"SEM={r.sem:.3f}, mean_pairs={r.mean_pairs:.1f}")
    return r


def main() -> None:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="noise-robust C1 confirmation (any_flip=False 補強)")
    ap.add_argument("--mode", choices=["smoke", "quick", "detect"], default="quick")
    ap.add_argument("--tasks", default="vdr_D60,vdr_D15")
    ap.add_argument("--out", default=str(_HERE / "c1_noise_robust_results.json"))
    args = ap.parse_args()

    if args.mode == "smoke":
        seeds = [20260530, 20260531]
        n_restarts, n_evals, sigma, n_train, n_eval = 4, 40, 0.15, 32, 32
        n_avgs = [1, 4]
        clip_bounds = [("hard", "current")]
    elif args.mode == "quick":
        # G1: n_avg を上げると 1 評価が n_avg 倍重い。n_restarts/n_evals を抑え n_avg 軸に予算配分。
        seeds = [20260530, 20260531, 20260601]
        n_restarts, n_evals, sigma, n_train, n_eval = 4, 60, 0.15, 40, 40
        n_avgs = [1, 4, 16]  # noise を 1/4 まで averaging (SEM = std/√16 = std/4)
        clip_bounds = [("hard", "current"), ("none", "current"), ("hard", "wide")]
    else:  # detect
        seeds = [20260530, 20260531, 20260601]
        n_restarts, n_evals, sigma, n_train, n_eval = 6, 100, 0.15, 48, 48
        n_avgs = [1, 4, 16, 36]
        clip_bounds = [("hard", "current"), ("none", "current"), ("soft", "current"),
                       ("hard", "wide")]

    vdr = {f"vdr_D{t.seq_len}": t for t in make_regimes(delays=(15, 30, 45, 60),
                                                        distractor_amp=0.2, in_dim=2)}
    for t in vdr.values():
        if not hasattr(t, "name"):
            try:
                t.name = f"vdr_D{t.seq_len}"
            except Exception:
                pass
    sel = [s.strip() for s in args.tasks.split(",") if s.strip()]

    print(f"=== noise-robust C1 confirmation (mode={args.mode}) ===")
    print(f"tasks={sel} seeds={seeds} n_avgs={n_avgs} clip_bounds={clip_bounds}")
    print(f"n_restarts={n_restarts} n_evals={n_evals} sigma={sigma} n_train={n_train}")
    print("予測: valley_fraction が n_avg 増で noisy-flat null と同軌道で減衰 → ノイズ起源確証")
    print("反証条件: 谷が null より高い plateau に残る → 真の幾何谷 (正規化人工物でない真の多峰)\n")

    results: list[NAvgResult] = []
    t_start = time.time()
    for name in sel:
        if name not in vdr:
            raise SystemExit(f"unknown task {name!r}; available={list(vdr)}")
        res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
        for (clip, bounds_kind) in clip_bounds:
            for n_avg in n_avgs:
                t0 = time.time()
                r = run_navg_point(res=res, task=vdr[name], clip=clip,
                                   bounds_kind=bounds_kind, n_avg=n_avg, seeds=seeds,
                                   n_restarts=n_restarts, n_evals=n_evals, sigma=sigma,
                                   n_train=n_train, n_eval=n_eval)
                results.append(r)
                dt = time.time() - t0
                print(f"[{r.setting}] verdict={r.verdict} valley_mean={r.valley_mean:.3f} "
                      f"null_vf={r.null_vf:.3f} margin={r.margin:+.3f} "
                      f"noise_std={r.noise_std:.3f} SEM={r.sem:.4f} "
                      f"R2[{r.optima_r2_min:.3f},{r.optima_r2_max:.3f}] ({dt:.1f}s)")
    total_dt = time.time() - t_start
    print(f"\ntotal wall-clock = {total_dt:.1f}s")

    # ---- 集約判定 ----
    # any_flip 踏襲: sweep で any_flip=False。この harness は反転を探さず noise-confound の頑健性を測る。
    # geometric_valley が 1 つでも出れば「真の谷あり=人工物でない真の多峰」(P_robust 反証)。
    has_geometric = any(r.verdict == "geometric_valley" for r in results)
    # 谷が averaging で消える (smooth_after_denoise) が n_avg 最大で出たか (ノイズ起源確証)
    denoise_collapse = {}
    by_setting = {}
    for r in results:
        key = f"{r.task}|clip={r.clip}|bounds={r.bounds}"
        by_setting.setdefault(key, []).append(r)
    for key, rs in by_setting.items():
        rs_sorted = sorted(rs, key=lambda x: x.n_avg)
        v_lo = rs_sorted[0].valley_mean   # n_avg=1
        v_hi = rs_sorted[-1].valley_mean  # n_avg max
        null_hi = rs_sorted[-1].null_vf
        denoise_collapse[key] = {
            "n_avg_min": rs_sorted[0].n_avg, "n_avg_max": rs_sorted[-1].n_avg,
            "valley_at_min_navg": v_lo, "valley_at_max_navg": v_hi,
            "null_at_max_navg": null_hi,
            "tracks_null": abs(v_hi - null_hi) < 0.1,  # 実 valley が null と同軌道
            "decreased": v_hi < v_lo - 0.05,
        }

    payload = {
        "proposition": ("P_robust: 既存 C1 の valley≈1.0 は評価ノイズ起源。noise-robust C1 "
                        "(n_avg seed 平均 + SEM scale 谷閾) で valley_fraction が noisy-flat "
                        "null と同軌道で減衰すれば確証。null より高い plateau なら真の幾何谷 (反証)。"),
        "mode": args.mode, "seeds": seeds, "n_avgs": n_avgs,
        "clip_bounds": clip_bounds, "n_restarts": n_restarts, "n_evals": n_evals,
        "sigma": sigma, "n_train": n_train, "n_eval": n_eval,
        "total_wall_clock_s": total_dt,
        "any_flip": False,  # sweep 踏襲: 全正規化で deceptive 反転なし
        "has_geometric_valley": has_geometric,
        "denoise_collapse_per_setting": denoise_collapse,
        "results": [
            {"setting": r.setting, "task": r.task, "clip": r.clip, "bounds": r.bounds,
             "n_avg": r.n_avg, "noise_std": r.noise_std, "sem": r.sem,
             "valley_fractions": r.valley_fractions, "valley_mean": r.valley_mean,
             "null_vf": r.null_vf, "margin": r.margin,
             "optima_r2_min": r.optima_r2_min, "optima_r2_max": r.optima_r2_max,
             "verdict": r.verdict, "note": r.note}
            for r in results
        ],
        "verdict_summary": (
            "geometric_valley 検出 → 真の多峰 (正規化人工物でない)。③ablation escalation を要検討。"
            if has_geometric else
            "geometric_valley なし。valley_fraction が n_avg 増で null と同軌道なら "
            "『既存 C1 の谷=評価ノイズ起源』を頑健に確証。よって prior の vdr 地形を C1 で "
            "smooth とも deceptive とも言えない (C1 は stochastic fitness に計測不能)。"
            "『③不要』を主張するには noise-robust C1 か deterministic fitness が必要。"
        ),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n=> has_geometric_valley={has_geometric}")
    for key, d in denoise_collapse.items():
        print(f"   [{key}] valley {d['valley_at_min_navg']:.2f}(n_avg={d['n_avg_min']})"
              f" -> {d['valley_at_max_navg']:.2f}(n_avg={d['n_avg_max']}) "
              f"null={d['null_at_max_navg']:.2f} tracks_null={d['tracks_null']}")
    print(f"   wrote {args.out}")


if __name__ == "__main__":
    main()
