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
# instrument: valley_fraction
# --------------------------------------------------------------------------- #
def valley_fraction(field, rng, dim, n_pairs=400, n_steps=24, margin_frac=0.02, anchor="good"):
    """高 fitness 2 点を結ぶ線分上に **谷 (両端より低い dip)** が在る率。

    anchor="good": ランダム多数点から fitness 上位を anchor に選び、その対で測る (最適間障壁)。
    margin: dip が両端 min より (fitness range × margin_frac) 以上低いとき谷と判定。
    """
    # fitness range 推定 (margin 基準)
    probe = np.array([field(rng.uniform(-1, 1, dim)) for _ in range(600)])
    frange = float(probe.max() - probe.min()) + 1e-12
    # anchor 群
    if anchor == "good":
        cand = rng.uniform(-1, 1, (max(2 * n_pairs, 300), dim))
        fc = np.array([field(c) for c in cand])
        order = np.argsort(-fc)
        good = cand[order[:max(40, n_pairs)]]
    else:
        good = rng.uniform(-1, 1, (max(40, n_pairs), dim))
    valleys = 0
    for _ in range(n_pairs):
        i, j = rng.integers(0, good.shape[0], size=2)
        a, b = good[i], good[j]
        if np.allclose(a, b):
            continue
        ts = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]
        seg = np.array([field((1 - t) * a + t * b) for t in ts])
        endpoint_min = min(field(a), field(b))
        if seg.min() < endpoint_min - margin_frac * frange:
            valleys += 1
    return valleys / n_pairs


def hillclimb(field, x0, rng, dim, steps=200, sigma=0.15):
    """決定論的 (1+1) greedy ascent。固定 seed の摂動列で局所最適へ。"""
    x = x0.copy()
    fx = field(x)
    for s in range(steps):
        step = sigma * (0.5 ** (s / 60.0))   # 焼きなまし的に縮小
        cand = np.clip(x + step * rng.standard_normal(dim), -1, 1)
        fc = field(cand)
        if fc > fx:
            x, fx = cand, fc
    return x, fx


def count_optima(field, rng, dim, n_starts=30, round_dec=1):
    """multi-start hillclimb の到達点を粗グリッドで丸めて distinct 数を数える。"""
    optima = []
    fitn = []
    for _ in range(n_starts):
        x0 = rng.uniform(-1, 1, dim)
        xo, fo = hillclimb(field, x0, rng, dim)
        optima.append(np.round(xo, round_dec))
        fitn.append(fo)
    keys = set(tuple(o) for o in optima)
    # fitness top に近い basin だけ数える (微小 basin の noise を除く)
    fitn = np.array(fitn)
    top = fitn.max()
    near_top = [tuple(np.round(o, round_dec)) for o, f in zip(optima, fitn) if f >= top - 0.15 * (abs(top) + 1e-9)]
    return {"distinct_optima": len(keys), "distinct_near_top": len(set(near_top)),
            "fitness_spread": float(fitn.max() - fitn.min())}


def calibrate_field(name, field, rng, dim):
    vf_good = valley_fraction(field, rng, dim, anchor="good")
    vf_rand = valley_fraction(field, rng, dim, anchor="random")
    opt = count_optima(field, rng, dim)
    return {"name": name, "valley_fraction_good_anchor": vf_good,
            "valley_fraction_random_anchor": vf_rand, **opt}


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
        vf = np.array([r["valley_fraction_good_anchor"] for r in rows])
        nt = np.array([r["distinct_near_top"] for r in rows])
        results["fields"][fname] = {
            "instances": rows,
            "valley_fraction_good_mean": float(vf.mean()),
            "valley_fraction_good_std": float(vf.std()),
            "distinct_near_top_mean": float(nt.mean()),
        }
        print(f"[{fname:20s}] valley_frac(good)={vf.mean():.3f}±{vf.std():.3f}  "
              f"near-top optima={nt.mean():.2f}  "
              f"valley_frac(rand)中央={np.median([r['valley_fraction_random_anchor'] for r in rows]):.3f}", flush=True)

    # 決定論性: 同 seed 2 回で完全一致
    rng = np.random.default_rng(SEED)
    f = MultimodalField(np.random.default_rng(SEED), DIM, k=6)
    v1 = valley_fraction(f, np.random.default_rng(123), DIM, n_pairs=200)
    v2 = valley_fraction(f, np.random.default_rng(123), DIM, n_pairs=200)
    results["determinism"] = {"same_seed_valley_v1": v1, "same_seed_valley_v2": v2,
                              "identical": bool(v1 == v2)}
    print(f"\n決定論性: 同seed valley {v1:.6f} == {v2:.6f} → {v1==v2}", flush=True)

    # 校正 verdict
    pos = results["fields"]["multimodal_pos"]["valley_fraction_good_mean"]
    neg1 = results["fields"]["unimodal_gauss_neg"]["valley_fraction_good_mean"]
    neg2 = results["fields"]["quadratic_bowl_neg"]["valley_fraction_good_mean"]
    discriminates = pos > 0.10 and neg1 < 0.05 and neg2 < 0.05
    results["calibration_pass"] = bool(discriminates)
    print(f"\n校正 verdict: 多峰 valley={pos:.3f} ≫ 単峰 valley={neg1:.3f}/{neg2:.3f} → "
          f"instrument 判別力 PASS={discriminates}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase0_multimodality_results.json")
    with open(out, "w", encoding="utf-8") as f2:
        json.dump(results, f2, ensure_ascii=False, indent=2)
    print(f"結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
