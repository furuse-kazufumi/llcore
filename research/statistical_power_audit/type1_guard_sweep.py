# SPDX-License-Identifier: Apache-2.0
"""(D) Type I guard — 各緩和が null landscape での偽陽性率をどれだけ上げるか併測.

これが honest disclosure の核: 「緩めれば進化が見える」だけでは不十分で、緩和の代償
(偽陽性増) を必ず併記する。sweet spot (真陽性回復 ≫ 偽陽性増) が実在するかを出す。

null landscape 群 (真に③優位が無い対照):
- d=0.0 corridor (校正 family の monotone smooth control, ③不要が設計上保証)
- pure-null: eval_once が gene 非依存の定数+noise (= 真の H0)
- shuffle-null: paired permutation で帰無を構成 (Wilcoxon の名目 alpha 検証)

手順 (DESIGN.type1_guard):
(1) baseline (全 knob ON) で各 null の偽陽性率を多 repeat で測定。
(2) ablation で反転を起こした緩和 level を null 群に適用し FPR 再測定。
(3) sweet spot 曲線: 緩和強度 横軸, [真陽性回復率(borderline 検出)] と [偽陽性率(null)] 縦 2 本。
    net gain = TPR - FPR (Youden J 風) 最大化 level を sweet spot。
(4) ROC 的要約: 各 level を (FPR, TPR) 点でプロット。
(5) K1 OFF の null 偽陽性増を最優先測定 (外してはいけない境界)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _audit_common as AC  # noqa: E402

BASELINES = ("rr_hillclimb", "panmictic_ga", "random")


def _load_bearing(scores: dict[str, np.ndarray], **gate_kw) -> bool:
    me = scores["map_elites"]
    return all(AC.eval_gate(me, scores[b], **gate_kw).passes for b in BASELINES)


# ---------------------------------------------------------------------------
# pure-null landscape: gene 非依存の定数+noise (= 真の H0)
# ---------------------------------------------------------------------------


def _pure_null_scores(*, n_seeds: int, n_evals: int, base_seed: int,
                      honest_off: bool = False) -> dict[str, np.ndarray]:
    """gene 非依存 fitness (定数 0.5 + noise) で MAP-E/baseline を走らせる.

    どの method も同分布になるはず → 真の H0。honest_off=True で K1 OFF (noisy best 持越し)
    を再現し、null でも偽 best が水増しされるかを測る (K1 を外す危険性の純粋測定)。
    """
    from selection_lab import map_elites, random_restart_hillclimb, panmictic_ga
    dim = AC.CORRIDOR_D
    bounds = (np.zeros(dim), np.ones(dim))
    bb = (np.zeros(1), np.ones(1))
    init_batch = max(20, n_evals // 10)
    NOISE = 0.10

    def eval_once(gene, rng):
        return float(0.5 + rng.normal(0, NOISE))  # gene 非依存 = 真の H0

    def rng(mi, s):
        return np.random.default_rng(np.random.SeedSequence([base_seed, mi, s]))

    def honest(gene, s):
        return AC.honest_reevaluate(eval_once, gene, n_trials=30,
                                    rng=np.random.default_rng(
                                        np.random.SeedSequence([base_seed, 7, s])))

    out = {m: [] for m in ("map_elites",) + BASELINES}
    for s in range(n_seeds):
        r = map_elites(eval_once, AC.behavior_mean, dim=dim, bounds=bounds,
                       behavior_bounds=bb, grid_shape=(24,), n_evals=n_evals,
                       init_batch=init_batch, sigma=0.10, rng=rng(0, s))
        out["map_elites"].append(float(r.best_fitness) if honest_off else honest(r.best_gene, s))
        r2 = random_restart_hillclimb(eval_once, dim=dim, bounds=bounds, n_evals=n_evals,
                                      sigma=0.10, restart_patience=max(10, n_evals // 20),
                                      rng=rng(1, s))
        out["rr_hillclimb"].append(float(r2.best_fitness) if honest_off else honest(r2.best_gene, s))
        r3 = panmictic_ga(eval_once, dim=dim, bounds=bounds, n_evals=n_evals, pop_size=20,
                          tournament_k=3, sigma=0.10, elitism=1, rng=rng(2, s))
        out["panmictic_ga"].append(float(r3.best_fitness) if honest_off else honest(r3.best_gene, s))
        rr = rng(3, s)
        cands = [bounds[0] + (bounds[1] - bounds[0]) * rr.random(dim) for _ in range(n_evals)]
        fits = [eval_once(g, rr) for g in cands]
        if honest_off:
            out["random"].append(float(max(fits)))
        else:
            best = cands[int(np.argmax(fits))]
            out["random"].append(honest(best, s))
    return {m: np.array(out[m]) for m in out}


def _shuffle_null_fpr(scores: dict[str, np.ndarray], *, n_perm: int, rng,
                      **gate_kw) -> float:
    """paired permutation で帰無を構成し名目 alpha が守られているか直接検定.

    MAP-E vs best baseline の paired delta の符号を seed 内でランダム反転し、
    元 gate (vs best baseline) が PASS する割合 = 実 Type I error。
    """
    me = scores["map_elites"]
    best_b = max(BASELINES, key=lambda b: float(scores[b].mean()))
    b = scores[best_b]
    delta = me - b
    n = len(delta)
    fp = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        pd = delta * signs
        a_perm = b + pd  # 帰無下で符号ランダム
        g = AC.eval_gate(a_perm, b, **gate_kw)
        fp += int(g.passes)
    return fp / n_perm


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    guard = AC.RunGuard.start("type1_guard_sweep")
    if args.smoke:
        n_seeds, n_evals = 15, 600
        null_repeats = [20260530, 777]
        borderline_d = 0.13
        n_perm = 200
    else:
        n_seeds, n_evals = 15, 1500
        null_repeats = [20260530, 777, 31337, 11, 99]
        borderline_d = 0.14
        n_perm = 500

    # 緩和 level セット (ablation で反転を起こす候補)。横軸 = 緩和強度。
    relax_levels = [
        ("baseline_all_on", {}),
        ("K2_min_effect=0.10", {"min_effect": 0.10}),
        ("K2_min_effect=0.05", {"min_effect": 0.05}),
        ("K2_min_effect=0.0", {"min_effect": 0.0}),
        ("K2_alpha=0.10", {"alpha": 0.10}),
        ("K2_alpha=0.20", {"alpha": 0.20}),
        ("K2_min_seeds=10", {"min_seeds": 10}),
        ("K2_min_seeds=5", {"min_seeds": 5}),
    ]

    # ---- (1) 各 null の baseline FPR + 各緩和 level の FPR ----
    # d=0 corridor null
    print(f"[type1] d=0 corridor null over {len(null_repeats)} seeds...")
    d0_scores = [AC.corridor_method_scores(0.0, n_seeds=n_seeds, n_evals=n_evals, base_seed=bs)
                 for bs in null_repeats]
    # pure-null
    print("[type1] pure-null landscape...")
    pure_scores = [_pure_null_scores(n_seeds=n_seeds, n_evals=n_evals, base_seed=bs)
                   for bs in null_repeats]
    # pure-null K1 OFF (noisy best 持越し — K1 を外す危険性)
    print("[type1] pure-null K1 OFF (noisy best 持越し)...")
    pure_k1off = [_pure_null_scores(n_seeds=n_seeds, n_evals=n_evals, base_seed=bs,
                                    honest_off=True) for bs in null_repeats]

    def fpr_over(scores_list, **gate_kw) -> float:
        return float(np.mean([_load_bearing(sc, **gate_kw) for sc in scores_list]))

    # ---- (2) borderline 真陽性検出率 (TPR proxy) ----
    print(f"[type1] borderline d={borderline_d} TPR...")
    bl_scores = [AC.corridor_method_scores(borderline_d, n_seeds=n_seeds, n_evals=n_evals,
                                           base_seed=bs) for bs in null_repeats]

    # ---- sweep: 各 level で FPR (3 null) + TPR (borderline) ----
    rng = np.random.default_rng(20260601)
    level_table: dict[str, dict] = {}
    roc_points = []
    for name, kw in relax_levels:
        fpr_d0 = fpr_over(d0_scores, **kw)
        fpr_pure = fpr_over(pure_scores, **kw)
        fpr_pure_k1off = fpr_over(pure_k1off, **kw)
        # shuffle-null FPR (名目 alpha 直接検定, d0 corridor の delta を使う)
        shuf = float(np.mean([_shuffle_null_fpr(sc, n_perm=n_perm, rng=rng, **kw)
                              for sc in d0_scores]))
        tpr = fpr_over(bl_scores, **kw)  # borderline で load_bearing 検出率
        net_gain = tpr - max(fpr_d0, fpr_pure, shuf)
        level_table[name] = {
            "gate_kw": kw,
            "fpr_d0_corridor": fpr_d0,
            "fpr_pure_null": fpr_pure,
            "fpr_pure_null_K1_OFF": fpr_pure_k1off,
            "fpr_shuffle_null": shuf,
            "tpr_borderline": tpr,
            "net_true_positive_gain": net_gain,
        }
        roc_points.append({"level": name, "fpr": max(fpr_d0, fpr_pure, shuf), "tpr": tpr})
        print(f"  {name}: FPR(d0={fpr_d0:.2f} pure={fpr_pure:.2f} "
              f"k1off={fpr_pure_k1off:.2f} shuf={shuf:.3f}) TPR={tpr:.2f} net={net_gain:+.2f}")

    # sweet spot = net gain 最大 かつ max FPR <= 2*alpha (=0.10)
    candidates = {k: v for k, v in level_table.items()
                  if max(v["fpr_d0_corridor"], v["fpr_pure_null"], v["fpr_shuffle_null"]) <= 0.10}
    if candidates:
        sweet = max(candidates, key=lambda k: candidates[k]["net_true_positive_gain"])
    else:
        sweet = "baseline_all_on"
    for k in level_table:
        level_table[k]["sweet_spot"] = (k == sweet)

    # K1 OFF の null 危険性: pure-null で baseline gate が偽陽性を出すか
    k1_off_fpr = level_table["baseline_all_on"]["fpr_pure_null_K1_OFF"]
    k1_on_fpr = level_table["baseline_all_on"]["fpr_pure_null"]

    meta = guard.finish()
    payload = {
        "_meta": {**meta, "design": "(D) Type I guard sweet spot",
                  "n_seeds": n_seeds, "n_evals": n_evals,
                  "null_repeat_seeds": null_repeats, "borderline_d": borderline_d,
                  "n_perm_shuffle": n_perm, "alpha_nominal": 0.05,
                  "note": "FPR は load_bearing(3 baseline 全勝)で測定。負の結果(=sweet spot=現行設定)も valid。"},
        "level_table": level_table,
        "roc_points": roc_points,
        "sweet_spot": sweet,
        "K1_off_danger": {
            "pure_null_FPR_K1_ON": k1_on_fpr,
            "pure_null_FPR_K1_OFF": k1_off_fpr,
            "K1_off_inflates_fpr": bool(k1_off_fpr > k1_on_fpr + 0.05),
            "verdict": ("K1 (fresh-seed 再評価) を外すと null でも偽 best 水増しで FPR が上がる → "
                        "外してはいけない" if k1_off_fpr > k1_on_fpr + 0.05
                        else "この null/budget では K1 OFF の FPR 増は限定的"),
        },
    }
    out = AC.dump_json(AC.AUDIT_DIR / "type1_guard_sweep_results.json", payload)
    print(f"[type1] wrote {out}  ({meta['wall_clock_s']}s) sweet_spot={sweet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
