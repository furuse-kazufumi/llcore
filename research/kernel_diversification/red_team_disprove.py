# SPDX-License-Identifier: Apache-2.0
"""Red-team (2) — ③(MAP-Elites) が RR を含む 3 baseline 全勝する faithful positive control を探す.

BG9-4 主張「kid-axis 欺瞞 corridor では RR を構造的に排除不能」を **別角度で反証** する試み。
RR を不当に縛らずに (faithful)、MAP-E が RR/panmictic/random を **全て** strict_gate で撃破する
構成を 2-4 個試す。1 個でも見つかれば主張は反証 → 即報告。

faithful 規律 (捏造でない範囲):
  - RR の restart 予算・kernel_id 直接サンプルを人為的に潰さない。
  - landscape は全 method に同一適用。MAP-E 専用の優遇 hook を入れない。
  - behavior descriptor 変更は MAP-E の niche 設計の自由 (探索オペレータ自体は同一)。
    ただし pre-reg の behavior=(kernel_id, theta L1) を変える場合は「pre-reg 逸脱の探索的 probe」
    と明記する。

試す構成 (RR の kid 直接サンプルを「無効化せず」に効かなくする faithful な地形/niche 設計):

  C-A) 多次元 behavior (high-dim theta corridor niche)
      pre-reg 逸脱 probe。behavior = (kernel_id, theta0_frac, theta1_frac, theta2_frac) の 4D。
      target は **target kernel 内の高次元 theta corridor** (各 theta が上端 band)。RR は kid は
      直接引けても、in-basin の高次元 theta corridor は CLT/体積で不到達 (Step4 mean(24) 再現の
      kernel 局所版)。MAP-E は theta 各軸 niche を stepping-stone に corridor を ratchet。

  C-B) sequential-kernel 構造 (kernel_id 遷移コスト)
      target に至るには kernel_id を 0→1→2→3 と **連続経由** せねば fitness が出ない地形
      (各 kernel basin が「直前 kernel を経た theta 痕跡」を要求)。RR の restart 直撃は
      単発の kid=3 を引いても痕跡が無く低 fitness。MAP-E は各 kid niche を保持して相伝。
      → ただし faithful 性は要吟味 (痕跡を gene に持てないなら不当に難しいだけ)。本実装は
      gene 内 junk 次元 (theta[3]) を「経由した最大 kid」の proxy として使わず、純粋に
      「kid と theta の同時条件」で表現する (gene only, 履歴なし=faithful)。

  C-C) target kernel 内 deep theta-corridor (Step4 corridor を特定 kernel の theta 部分空間で再現)
      behavior は pre-reg どおり (kernel_id, theta L1) のまま。ただし target peak を
      「kid≈3.6 ∧ theta が **4 次元すべて高い特定 corner** (L1 高 ∧ 各座標 high)」に置く。
      theta corridor を Step4 並みに締める (_POS_THETA_W を小さく) と BG9-4 は「MAP-E 自身も
      starve」と報告したが、ここでは **theta L1 niche (8 bin) が corridor を ratchet** できるよう
      corridor を L1 軸に整列させて再試行。RR は kid 直撃後の in-basin theta corridor で starve。

  C-D) 多 target + 1 つだけ高 (deceptive multi-basin)
      4 kernel それぞれに局所峰を置き、target kernel のみ最高。RR は最初に当たった kid basin に
      hill-climb で詰まり (他 basin の天井で満足)、restart しても各 basin の局所峰に再収束。
      MAP-E は 4 niche 全保持で最高 basin を必ず archive に持つ。faithful (RR の restart は活きる)。

各構成で MAP-E vs (RR/panmictic/random) を smoke(5 seed/800 evals) で strict_gate 測定。
**③が RR 含む全勝 = 主張反証。**

read-only import: bg9_driver の run_methods_crn / strict_gate / 定数 / kernel_ga_bounds。
新規地形のみ本 module に閉じる。src/既存 .py 無改変。git 非実行。

実行: py -3.11 research/kernel_diversification/red_team_disprove.py [--seeds N] [--n-evals M] [--only C-A]
出力: research/kernel_diversification/red_team_disprove_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import bg9_driver as bg9  # noqa: E402  read-only
from kernel_fitness import kernel_behavior, kernel_behavior_bounds, kernel_ga_bounds  # noqa: E402
from kernels import GA_DIM, N_KERNELS  # noqa: E402

EvalOnce = Callable[[np.ndarray, np.random.Generator], float]

TARGET_KID = 3.6
LOCAL_KID = 0.5
NOISE = 0.008
PROXY = 0.8  # target 到達判定


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:  # pragma: no cover
        pass


def _theta_fracs(gene_vec5: np.ndarray) -> np.ndarray:
    """gene の theta 4 座標を GA bounds の theta box で [0,1] 正規化 (各座標 frac)."""
    lo, hi = kernel_ga_bounds()
    th_lo, th_hi = lo[1:5], hi[1:5]
    th = np.clip(np.asarray(gene_vec5[1:5], dtype=np.float64), th_lo, th_hi)
    return (th - th_lo) / np.maximum(th_hi - th_lo, 1e-12)


# ===========================================================================
# C-A: 多次元 behavior + high-dim theta corridor (in-basin CLT 不到達)
# ===========================================================================

_CA_THETA_W = 0.16   # 各 theta 軸 corridor 幅 (狭く: random が CLT で不到達)
_CA_KID_W = 0.16


def make_ca_eval() -> EvalOnce:
    """target = kid≈3.6 ∧ theta0/1/2 全てが上端 corner (各 frac→1)。

    in-basin でも theta 3 座標を **同時に** 上端へ押す必要があり、random/RR の in-basin draw は
    体積 (各幅 0.16 → 0.16^3 ≈ 0.004) で構造的に届きにくい。MAP-E は theta 各軸 niche を
    stepping-stone に corridor を ratchet。kid 谷も彫り hill-climb を阻む。
    """
    target_corner = np.array([0.9, 0.9, 0.9])  # theta0/1/2 の目標 frac

    def ev(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
        fr = _theta_fracs(gene_vec5)
        # 局所罠: 低 kid, theta 不問
        local = 0.55 * np.exp(-((kid - LOCAL_KID) ** 2) / (2 * 0.70 ** 2))
        # target: kid basin × theta corner Gaussian (3 座標同時)
        kid_g = np.exp(-((kid - TARGET_KID) ** 2) / (2 * _CA_KID_W ** 2))
        d2 = np.sum((fr[:3] - target_corner) ** 2)
        theta_g = np.exp(-d2 / (2 * _CA_THETA_W ** 2))
        # kid 谷 (跨ぐ障壁)
        if LOCAL_KID <= kid <= TARGET_KID:
            dip = np.exp(-((kid - 2.0) ** 2) / (2 * 0.80 ** 2))
            barrier = 1.0 - dip
        else:
            barrier = 1.0
        target = 1.0 * kid_g * theta_g * barrier
        return float(max(local, target) + rng.normal(0, NOISE))

    return ev


def ca_behavior(gene_vec5: np.ndarray) -> np.ndarray:
    """4D behavior = (kernel_id, theta0_frac*scale, theta1_frac*scale, theta2_frac*scale).

    pre-reg 逸脱 probe。kernel_id + theta 3 座標を niche 軸に (高次元 niching)。
    """
    kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
    fr = _theta_fracs(gene_vec5)[:3]
    return np.concatenate([[kid], fr]).astype(np.float64)


def ca_behavior_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([float(N_KERNELS) - 1e-9, 1.0, 1.0, 1.0])
    return lo, hi


CA_GRID = (N_KERNELS, 4, 4, 4)  # kid 4 bin + theta 3 軸 ×4 bin


# ===========================================================================
# C-B: sequential-kernel (kernel_id を連続経由しないと target が出ない, gene only)
# ===========================================================================


def make_cb_eval() -> EvalOnce:
    """target kernel (kid=3) の fitness が、gene の theta が「kid=0,1,2 各 basin の最適 theta を
    内挿した痕跡」を持つときのみ高い、という faithful 表現は gene 履歴なしでは作れない。
    代わりに **kid 軸に 3 連の deceptive dip** を置き、各 dip 谷を跨がないと次 basin に行けない
    multi-barrier corridor を作る (履歴なし = faithful)。RR の restart は単発で kid=3 を引けるが、
    kid 軸の最終 basin (3.6) は **3 連 dip の向こう** にあり、restart が直撃しても周囲が谷で
    hill-climb が即 downhill に落ちる狭い peak。MAP-E は各 kid niche を保持し相伝。
    """
    def ev(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
        fr = _theta_fracs(gene_vec5)
        # 各 kernel basin の中心に小さな峰 (天井は kid とともに上昇 = stepping-stone)
        peaks = []
        ceils = [0.45, 0.60, 0.75, 1.00]
        centers = [0.5, 1.5, 2.5, 3.6]
        for c, ce in zip(centers, ceils):
            peaks.append(ce * np.exp(-((kid - c) ** 2) / (2 * 0.10 ** 2)))
        # target (kid=3.6) は theta corridor も要求 (in-basin starve 用)
        d2 = np.sum((fr[:3] - 0.9) ** 2)
        theta_g = np.exp(-d2 / (2 * 0.20 ** 2))
        peaks[-1] = peaks[-1] * (0.3 + 0.7 * theta_g)
        base = max(peaks)
        return float(base + rng.normal(0, NOISE))

    return ev


# behavior は pre-reg どおり (kernel_id, theta L1)
CB_GRID = bg9.GRID_SHAPE


# ===========================================================================
# C-C: target kernel 内 deep theta-corridor 整列 to L1 軸 (Step4 corridor 再現)
# ===========================================================================

_CC_THETA_W = 0.10


def make_cc_eval() -> EvalOnce:
    """target = kid≈3.6 ∧ theta L1 が上端 (全座標高 = L1 最大 corner)。

    pre-reg behavior=(kernel_id, theta L1) の theta L1 軸に corridor を整列させる。
    MAP-E の theta L1 niche (8 bin) が corridor を ratchet できる設計。RR は kid 直撃後、
    in-basin で theta L1 を上端まで押す必要があるが (1+1) は L1 corridor の狭さで starve。
    """
    def ev(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
        fr = _theta_fracs(gene_vec5)
        l1_frac = float(np.mean(fr[:3]))  # theta L1 の正規化 proxy
        local = 0.55 * np.exp(-((kid - LOCAL_KID) ** 2) / (2 * 0.70 ** 2))
        kid_g = np.exp(-((kid - TARGET_KID) ** 2) / (2 * 0.16 ** 2))
        l1_g = np.exp(-((l1_frac - 0.92) ** 2) / (2 * _CC_THETA_W ** 2))
        if LOCAL_KID <= kid <= TARGET_KID:
            dip = np.exp(-((kid - 2.0) ** 2) / (2 * 0.80 ** 2))
            barrier = 1.0 - dip
        else:
            barrier = 1.0
        target = 1.0 * kid_g * l1_g * barrier
        return float(max(local, target) + rng.normal(0, NOISE))

    return ev


CC_GRID = bg9.GRID_SHAPE


# ===========================================================================
# C-D: deceptive multi-basin (4 kernel に局所峰, target のみ最高)
# ===========================================================================


def make_cd_eval() -> EvalOnce:
    """4 kernel basin それぞれに局所峰、target kernel (kid=3.6) のみ最高 (1.0)。

    各 basin 間に dip。RR は最初に当たった basin に hill-climb で詰まり、restart で別 basin に
    飛んでもその basin の局所峰に再収束 (各 basin 峰が魅力的)。問題は: target basin の天井が
    他より高いだけなら RR は restart で target を引けば climb で届く → faithful には RR も届く。
    そこで target basin のみ **theta corridor (in-basin starve)** を付与し、他 basin は theta 不問
    の即達峰にする → RR は他 basin で高 fitness に満足し target の探索予算が相対的に枯れる。
    MAP-E は 4 niche 保持で target を archive に確保し theta を ratchet。
    """
    ceils = [0.70, 0.75, 0.80, 1.00]
    centers = [0.5, 1.5, 2.5, 3.6]
    widths = [0.30, 0.30, 0.30, 0.16]

    def ev(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
        fr = _theta_fracs(gene_vec5)
        best = 0.0
        for i, (c, ce, w) in enumerate(zip(centers, ceils, widths)):
            g = ce * np.exp(-((kid - c) ** 2) / (2 * w ** 2))
            if i == 3:  # target basin: theta corridor を要求 (starve 用)
                d2 = np.sum((fr[:3] - 0.9) ** 2)
                g = g * np.exp(-d2 / (2 * 0.18 ** 2))
            best = max(best, g)
        return float(best + rng.normal(0, NOISE))

    return ev


CD_GRID = bg9.GRID_SHAPE


# ===========================================================================
# 共通: 1 構成を run_methods_crn で回し strict_gate で MAP-E vs 3 baseline 判定
# ===========================================================================


def run_config(
    name: str,
    eval_once: EvalOnce,
    behavior: Callable[[np.ndarray], np.ndarray],
    behavior_bounds: tuple[np.ndarray, np.ndarray],
    grid_shape: tuple[int, ...],
    *,
    n_evals: int,
    n_seeds: int,
    honest_n_trials: int,
    base_seed: int,
    min_seeds_gate: int,
    pre_reg_deviation: bool,
) -> dict:
    lo, hi = kernel_ga_bounds()
    bounds = (lo, hi)
    res = bg9.run_methods_crn(
        eval_once, behavior, dim=GA_DIM, bounds=bounds, behavior_bounds=behavior_bounds,
        grid_shape=grid_shape, n_evals=n_evals, n_seeds=n_seeds,
        honest_n_trials=honest_n_trials, sigma=bg9.SIGMA, base_seed=base_seed,
    )
    means = {m: float(res[m].mean()) for m in bg9._BASE_METHODS}
    reach = {m: float(np.mean(res[m] > PROXY)) for m in bg9._BASE_METHODS}
    baselines = ("rr_hillclimb", "panmictic_ga", "random")
    gates = {}
    beaten = 0
    for b in baselines:
        g = bg9.strict_gate(res["map_elites"], res[b], "map_elites", b, min_seeds=min_seeds_gate)
        gates[b] = {
            "diff": round(g.diff, 4), "wilcoxon_p": round(g.wilcoxon_p, 4),
            "paired_sign_delta": round(g.paired_sign_delta, 3),
            "mean_a": round(g.mean_a, 4), "mean_b": round(g.mean_b, 4),
            "passes": g.passes,
        }
        beaten += int(g.passes)
    beats_rr = gates["rr_hillclimb"]["passes"]
    all_win = beaten == 3
    return {
        "config": name,
        "pre_reg_deviation": pre_reg_deviation,
        "means": {k: round(v, 4) for k, v in means.items()},
        "reach_rate": {k: round(v, 3) for k, v in reach.items()},
        "gates": gates,
        "n_baselines_beaten": beaten,
        "beats_rr": beats_rr,
        "map_elites_beats_all_3": all_win,
        "raw_scores": {m: res[m].tolist() for m in bg9._BASE_METHODS},
    }


CONFIGS = {
    "C-A_highdim_theta_corridor": dict(
        make_eval=make_ca_eval, behavior=ca_behavior,
        bounds=ca_behavior_bounds, grid=CA_GRID, deviation=True,
        desc="4D behavior (kid, theta0/1/2 frac), in-basin 3D theta corner corridor",
    ),
    "C-B_sequential_kernel_multibarrier": dict(
        make_eval=make_cb_eval, behavior=kernel_behavior,
        bounds=kernel_behavior_bounds, grid=CB_GRID, deviation=False,
        desc="kid 軸 3 連 dip multi-barrier + target theta corridor (pre-reg behavior)",
    ),
    "C-C_inbasin_L1_corridor": dict(
        make_eval=make_cc_eval, behavior=kernel_behavior,
        bounds=kernel_behavior_bounds, grid=CC_GRID, deviation=False,
        desc="target 内 theta-L1 corridor を L1 niche 軸に整列 (pre-reg behavior)",
    ),
    "C-D_deceptive_multibasin": dict(
        make_eval=make_cd_eval, behavior=kernel_behavior,
        bounds=kernel_behavior_bounds, grid=CD_GRID, deviation=False,
        desc="4 basin 局所峰 + target のみ theta corridor (pre-reg behavior)",
    ),
}


def run_all(n_seeds: int, n_evals: int, only: str | None, base_seed: int = 20260602) -> dict:
    t0 = time.time()
    results = {}
    found_disproof = False
    disproof_configs = []
    for name, cfg in CONFIGS.items():
        if only and name != only:
            continue
        bb = cfg["bounds"]()
        r = run_config(
            name, cfg["make_eval"](), cfg["behavior"], bb, cfg["grid"],
            n_evals=n_evals, n_seeds=n_seeds, honest_n_trials=20,
            base_seed=base_seed, min_seeds_gate=3, pre_reg_deviation=cfg["deviation"],
        )
        r["desc"] = cfg["desc"]
        results[name] = r
        if r["map_elites_beats_all_3"]:
            found_disproof = True
            disproof_configs.append(name)
    wall = time.time() - t0
    return {
        "meta": {
            "task": "red_team(2) faithful positive control: ③ が RR 含む 3 baseline 全勝を探す",
            "n_seeds": n_seeds, "n_evals": n_evals, "base_seed": base_seed,
            "min_seeds_gate": 3, "gate_alpha": bg9.GATE_ALPHA,
            "gate_min_effect": bg9.GATE_MIN_EFFECT, "sigma": bg9.SIGMA,
            "wall_clock_sec": round(wall, 1),
            "honest_note": (
                "smoke (5 seed/800 evals)。strict_gate は本番同基準 (p<0.05 ∧ |δ|>=0.147 ∧ diff>0)。"
                "RR は不当に縛らず faithful (restart/kid 直接サンプルそのまま)。"
            ),
        },
        "results": results,
        "verdict": {
            "found_faithful_disproof": found_disproof,
            "disproof_configs": disproof_configs,
            "summary": (
                f"③ が RR 含む 3 baseline 全勝した faithful 構成: {disproof_configs if disproof_configs else 'なし'}。"
                + ("→ BG9-4 主張は反証された (harness は③検出可)。"
                   if found_disproof else
                   "→ faithful には反証できず。BG9-4 構造 N/A 主張を裏付け (RR 排除不能)。")
            ),
        },
    }


def _print(res: dict) -> None:
    print("=" * 84)
    print("RED-TEAM (2) — faithful disproof search (③ vs RR/GA/random)")
    print("=" * 84)
    for name, r in res["results"].items():
        dev = " [PRE-REG DEVIATION]" if r["pre_reg_deviation"] else ""
        print(f"\n[{name}]{dev}  beaten={r['n_baselines_beaten']}/3 "
              f"beats_rr={r['beats_rr']} ALL3={r['map_elites_beats_all_3']}")
        print(f"  {r['desc']}")
        print(f"  means: " + " ".join(f"{m}={r['means'][m]:.3f}" for m in bg9._BASE_METHODS))
        print(f"  reach: " + " ".join(f"{m}={r['reach_rate'][m]:.2f}" for m in bg9._BASE_METHODS))
        for b in ("rr_hillclimb", "panmictic_ga", "random"):
            g = r["gates"][b]
            print(f"    MAP-E vs {b:13s}: diff={g['diff']:+.4f} p={g['wilcoxon_p']:.3g} "
                  f"δ={g['paired_sign_delta']:+.2f} pass={g['passes']}")
    v = res["verdict"]
    print("\n" + "=" * 84)
    print(f"  found_faithful_disproof = {v['found_faithful_disproof']}")
    print(f"  {v['summary']}")
    print(f"  wall = {res['meta']['wall_clock_sec']}s")
    print("=" * 84)


def main() -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-evals", type=int, default=800)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    res = run_all(args.seeds, args.n_evals, args.only)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "red_team_disprove_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(res)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
