# SPDX-License-Identifier: Apache-2.0
"""Phase 0 (F9): 多峰性 instrument の校正 — valley_fraction + n_optima を pos/neg control で検証。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) Phase 0 / red-team F9:
  capability terrain-bet の **前提条件** = 「離散トポロジー軸 (width/branch/op) が多峰か」。
  単峰なら進化 (MAP-Elites 等) は gradient/greedy に勝てない (M3 が離散軸で再現)。
  そこで「多峰性を falsifiable に測る instrument」を **校正** する: eval を決定論化 (eval_noise→
  機械 eps) し、valley_fraction (高 fitness 2 点間に谷=障壁が在る率) と n_optima (multi-start
  hillclimb で見つかる distinct 局所最適数) を、**positive control=合成多峰** と
  **negative control=単峰 (凸 bowl / ESN 風単一 basin)** で判別できるか確認する。

校正の合否:
  - valley_fraction(多峰) ≫ valley_fraction(単峰≈0)、n_optima(多峰)>1 / n_optima(単峰)=1。
  - 決定論性: 同 seed で完全再現、異 seed でも metric が安定 (= landscape の性質を測れている)。

これが PASS すれば、Phase 2 terrain-bet で **実損失地形** に同 instrument を当てて
EXISTS/NULL/ARTIFACT を proper power で判定する土台ができる (本 script はその校正のみ)。

honest:
  - 本校正は連続 proxy 空間 (低次元 box) 上の合成 family で instrument の判別力を確認するもの。
    実トポロジー (離散) 地形への適用は Phase 2。校正と適用を混同しない。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 20260609
DIM = 8                 # 校正用 proxy 空間次元
EVAL_NOISE = 0.0        # 決定論化 (機械 eps; >0 にすると noisy eval の感度試験)


# --------------------------------------------------------------------------- #
# 合成 fitness family (pos = 多峰, neg = 単峰)
# --------------------------------------------------------------------------- #
class MultimodalField:
    """K 個の Gaussian bump の和 = 多峰 (positive control)。"""
    def __init__(self, rng, dim, k=6, width=0.25):
        self.centers = rng.uniform(-1, 1, (k, dim))
        self.amps = rng.uniform(0.6, 1.0, k)
        self.width = width
        self.noise = EVAL_NOISE

    def __call__(self, x):
        x = np.asarray(x)
        d2 = ((self.centers - x) ** 2).sum(axis=1)
        f = float((self.amps * np.exp(-d2 / (2 * self.width ** 2))).sum())
        return f


class UnimodalField:
    """単一 Gaussian bump (凸近傍, 単峰) = negative control。"""
    def __init__(self, rng, dim, width=0.6):
        self.center = rng.uniform(-0.5, 0.5, dim)
        self.width = width
        self.noise = EVAL_NOISE

    def __call__(self, x):
        x = np.asarray(x)
        d2 = ((self.center - x) ** 2).sum()
        return float(np.exp(-d2 / (2 * self.width ** 2)))


class QuadraticBowl:
    """-‖x-c‖² = 厳密単峰 (concave) = 最も clean な negative control。"""
    def __init__(self, rng, dim):
        self.center = rng.uniform(-0.5, 0.5, dim)
        self.noise = EVAL_NOISE

    def __call__(self, x):
        x = np.asarray(x)
        return float(-((self.center - x) ** 2).sum())


# --------------------------------------------------------------------------- #
# instrument: hillclimb → 距離クラスタリングで basin 検出 → basin 間 valley 検定
# --------------------------------------------------------------------------- #
def hillclimb(field, x0, rng, dim, steps=300, sigma=0.2):
    """決定論的 (1+1) greedy ascent。固定 seed の摂動列で局所最適へ。"""
    x = x0.copy()
    fx = field(x)
    for s in range(steps):
        step = sigma * (0.5 ** (s / 80.0))   # 焼きなまし的に縮小
        cand = np.clip(x + step * rng.standard_normal(dim), -1, 1)
        fc = field(cand)
        if fc > fx:
            x, fx = cand, fc
    return x, fx


def find_basins(field, rng, dim, n_starts=40, merge_radius=0.4):
    """multi-start hillclimb → 到達点を **Euclid 距離でクラスタリング** (grid 丸め非依存)。

    各 cluster = 1 basin (内に最良点を代表に)。広い単峰 Gaussian は全 start が単一 center へ
    収束 → cluster=1 (grid 丸めの膨張を回避)。返り値 = fitness 降順の cluster 代表 list。
    """
    opts = []
    for _ in range(n_starts):
        x0 = rng.uniform(-1, 1, dim)
        xo, fo = hillclimb(field, x0, rng, dim)
        opts.append((xo, float(fo)))
    clusters: list[list] = []  # [center(np), best_f, count]
    for xo, fo in sorted(opts, key=lambda t: -t[1]):
        placed = False
        for c in clusters:
            if np.linalg.norm(xo - c[0]) < merge_radius:
                c[2] += 1
                placed = True
                break
        if not placed:
            clusters.append([xo, fo, 1])
    return clusters


def valley_fraction_between_basins(field, clusters, dim, n_steps=24, margin_frac=0.03,
                                   max_basins=8, frange=1.0):
    """**distinct basin 間**の線分に谷 (両端より下の dip=障壁) が在る率。

    basin が 1 個 (単峰) なら pair が無く 0.0 (= 障壁なし=単峰シグナル)。多峰なら真に分離した
    basin 間に障壁が在るはずで高率。anchor を「最高峰 1 つ」でなく「異なる basin」にしたのが
    旧版からの修正点。
    """
    cl = clusters[:max_basins]
    if len(cl) < 2:
        return 0.0, 0
    pairs = 0
    valleys = 0
    ts = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a, b = cl[i][0], cl[j][0]
            seg = np.array([field((1 - t) * a + t * b) for t in ts])
            endpoint_min = min(field(a), field(b))
            if seg.min() < endpoint_min - margin_frac * frange:
                valleys += 1
            pairs += 1
    return (valleys / pairs if pairs else 0.0), pairs


def calibrate_field(name, field, rng, dim):
    probe = np.array([field(rng.uniform(-1, 1, dim)) for _ in range(800)])
    frange = float(probe.max() - probe.min()) + 1e-12
    clusters = find_basins(field, rng, dim)
    n_basins = len(clusters)
    vf, n_pairs = valley_fraction_between_basins(field, clusters, dim, frange=frange)
    return {"name": name, "n_basins": n_basins,
            "valley_fraction_between_basins": vf, "n_basin_pairs": n_pairs,
            "top_basin_counts": [c[2] for c in clusters[:6]]}


def main():
    results = {"meta": {"seed": SEED, "dim": DIM, "eval_noise": EVAL_NOISE,
                        "note": "valley_fraction + n_optima を pos/neg control で校正。決定論的 eval。"},
               "fields": {}, "determinism": {}}

    # 各 field を独立 seed で複数 instance 校正 (metric の安定性も確認)
    field_specs = [
        ("multimodal_pos", lambda r: MultimodalField(r, DIM, k=6)),
        ("unimodal_gauss_neg", lambda r: UnimodalField(r, DIM)),
        ("quadratic_bowl_neg", lambda r: QuadraticBowl(r, DIM)),
    ]
    for fname, ctor in field_specs:
        rows = []
        for inst in range(4):                       # 4 instance で metric 安定性
            rng = np.random.default_rng(SEED + inst * 101)
            field = ctor(rng)
            rng2 = np.random.default_rng(SEED + 7 + inst)   # instrument 用 RNG (field と分離)
            rows.append(calibrate_field(fname, field, rng2, DIM))
        nb = np.array([r["n_basins"] for r in rows])
        vf = np.array([r["valley_fraction_between_basins"] for r in rows])
        results["fields"][fname] = {
            "instances": rows,
            "n_basins_mean": float(nb.mean()), "n_basins_std": float(nb.std()),
            "valley_fraction_between_basins_mean": float(vf.mean()),
        }
        print(f"[{fname:20s}] n_basins={nb.mean():.2f}±{nb.std():.2f}  "
              f"valley_frac(basin間)={vf.mean():.3f}", flush=True)

    # 決定論性: 同 seed 2 回で basin 数完全一致
    f = MultimodalField(np.random.default_rng(SEED), DIM, k=6)
    b1 = len(find_basins(f, np.random.default_rng(123), DIM))
    b2 = len(find_basins(f, np.random.default_rng(123), DIM))
    results["determinism"] = {"same_seed_basins_v1": b1, "same_seed_basins_v2": b2,
                              "identical": bool(b1 == b2)}
    print(f"\n決定論性: 同seed basin数 {b1} == {b2} → {b1==b2}", flush=True)

    # 校正 verdict: 多峰 n_basins>1 ∧ basin間 valley 高 / 単峰 n_basins=1
    pos_nb = results["fields"]["multimodal_pos"]["n_basins_mean"]
    pos_vf = results["fields"]["multimodal_pos"]["valley_fraction_between_basins_mean"]
    neg1_nb = results["fields"]["unimodal_gauss_neg"]["n_basins_mean"]
    neg2_nb = results["fields"]["quadratic_bowl_neg"]["n_basins_mean"]
    discriminates = pos_nb >= 2.0 and pos_vf > 0.5 and neg2_nb <= 1.5
    results["calibration_pass"] = bool(discriminates)
    print(f"\n校正 verdict: 多峰 n_basins={pos_nb:.2f}(valley={pos_vf:.2f}) ≫ "
          f"単峰 n_basins gauss={neg1_nb:.2f}/bowl={neg2_nb:.2f} → instrument 判別力 PASS={discriminates}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase0_multimodality_results.json")
    with open(out, "w", encoding="utf-8") as f2:
        json.dump(results, f2, ensure_ascii=False, indent=2)
    print(f"結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
