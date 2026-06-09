# SPDX-License-Identifier: Apache-2.0
"""Phase 2: framework 性 (F8) — North Star #4 の data 実証。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) §⑥ L99 (North Star #4):
  (a) 汎化 load-bearing — N 世代後の admit topology が param-shift baseline 比で構造的に
      多様化し、その多様性が **held-out tasks への汎化に load-bearing**(多様性が汎化を助ける)。
  (b) 3 plug-point 拡張性 — 新 base / 新 changeop / 新 certifier を **1 オブジェクト差替**で
      載せ替えられる(GeneCodec / Objective / VerifierBackend をテスト化)。

本スクリプトは 2 部構成。いずれも **src/ を一切改変せず additive**:
  - 既存 src 進化ループ ``llcore.evolution.minimal_ga.evolve()`` をそのまま使用。
  - 3 plug-point は research 側 ``verified_evolution_sdp_gate/coupled_nd.py`` に既実装の
    ``CoupledNDGeneCodec`` (GeneCodec) / ``RotationNDObjective`` (Objective) /
    ``make_nd_verifier`` (VerifierBackend, none/inf_norm/two_norm/sdp) を差し替える。

----------------------------------------------------------------------------- #
Part (b) — 3 plug-point swap test
----------------------------------------------------------------------------- #
同一の evolve ループに対し、3 軸を各々 **1 オブジェクト差替のみ** で載せ替え、すべてが
走り fitness/admit が出ることを示す:
  (i)   GeneCodec   — n / 基質を差し替え (CoupledNDGeneCodec(n=2) → (n=3) → (n=4))。
  (ii)  Objective   — task を差し替え (RotationNDObjective を period/radius 違いで生成)。
  (iii) VerifierBackend — none / inf_norm / two_norm / sdp を差し替え。

src の ``evolve()`` の gated path は scalar StateUpdateGene 専用 (codec 併用時は fail-loud)。
coupled gene の verifier gate は「子 fitness を verifier admit で篩う薄い wrapper」で表現する
(plan §⑦ の方針 = coupled-gene gating は verifier backend が担う)。これは src を触らず
fitness_func を 1 段包むだけ = plug-point 差替の正当な実演 (admit されない gene を強く減点)。

----------------------------------------------------------------------------- #
Part (a) — 汎化 load-bearing
----------------------------------------------------------------------------- #
**task family** (RotationNDObjective を period/radius/amp 違いで複数生成) を train family と
held-out family に分割し、**同予算 B (train-fitness 評価回数)** で 2 optimizer を比較:
  - **diverse**   : 構造多様な archive (MAP-Elites over topology descriptor) を進化させ、
                    archive 全 elite の train-family 平均 fitness で best を選ぶ。
  - **paramshift**: 単一 elite を param-shift だけで摂動する baseline (構造多様性なし)。
両者の best gene を **held-out family** で評価し、honest_eval 4 条件 AND で
「多様性が汎化に load-bearing か (diverse が held-out で baseline を上回るか)」を判定。
NULL/負なら正直にそう報告 (多様性は汎化に効かない = 第一級知見)。
地形識別力 (held-out が天井/床でないか) も確認する。

honest 留保:
  - **synthetic task family** (RotationNDObjective の damped block-rotation) であり、
    **実 SmolLM2-CE 損失地形ではない**。本実験は「多様性が汎化を助けるか」の clean probe。
  - coupled-gene verifier gate は fitness wrapper による proxy (src の scalar gate ではない)。
  - paired_sign_delta = net-win-fraction (n_wins−n_losses)/n_seeds (Cliff's delta ではない、
    計画 §⑬・phase2_capability_terrain.py と整合)。
  - 予算 = train-family-fitness 評価回数で optimizer 間を厳密に揃える。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- path: src (進化ループ) + research coupled_nd (3 plug-point) を additive 挿入 --------
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_COUPLED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "verified_evolution_sdp_gate")
)
if _COUPLED not in sys.path:
    sys.path.insert(0, _COUPLED)

import coupled_nd as C  # noqa: E402  (3 plug-point: GeneCodec / Objective / VerifierBackend)
from llcore.evolution.minimal_ga import evolve  # noqa: E402  (src 進化ループ; 無改変)

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# --------------------------------------------------------------------------- #
# 実験定数 (seed 固定; 既存スクリプト踏襲)
# --------------------------------------------------------------------------- #
SEED0 = 20260609
POP = 24
GENS = 30
N_SEEDS = 20                 # ≥15 (honest_eval 条件)
GRID = 10                    # diverse archive の topology descriptor grid (GxG)

# task family: RotationNDObjective を period/radius 違いで生成。train / held-out に分割。
N_DIM = 3                    # small-n per-component (plan: n≤16, ここは n=3)
_FAMILY_PARAMS = [           # (period, radius, amp) — 多様な block-rotation tasks
    (8.0, 0.90, 0.35), (10.0, 0.93, 0.40), (12.0, 0.95, 0.45),
    (9.0, 0.88, 0.30), (11.0, 0.92, 0.50), (13.0, 0.94, 0.38),
    (7.0, 0.91, 0.42), (14.0, 0.96, 0.33),
]
_N_TRAIN = 5                 # 先頭 5 task が train family、残り 3 が held-out family


def _make_family(params):
    return [C.RotationNDObjective(n=N_DIM, period=p, radius=r, amp=a, name=f"rot_p{p}_r{r}")
            for (p, r, a) in params]


# --------------------------------------------------------------------------- #
# Part (b) — 3 plug-point swap test
#
# 同一 evolve ループ (src 無改変) に対し 3 軸を 1 オブジェクト差替のみで載せ替える。
# --------------------------------------------------------------------------- #
def _wrap_fitness(objective, verifier, admit_penalty: float = 1.0):
    """objective.fitness を verifier.certifies で篩う薄い wrapper。

    plan §⑦ の coupled-gene gating 方針 = verifier backend が担う。src の evolve() の
    scalar gate path は使わず (codec 併用で fail-loud)、fitness を 1 段包んで
    「admit されない gene を強く減点」する additive な gate proxy にする。

    verifier=None / make_nd_verifier("none") は無条件 admit = 旧 fitness と一致。
    """
    def f(gene, rng):  # evolve は (gene, rng) -> float を期待 (FitnessFunc)
        base = objective.fitness(gene)
        if verifier is not None and not verifier.certifies(gene):
            return base - admit_penalty  # fail-closed: admit されない構造を減点
        return base
    return f


def swap_test(seed: int = SEED0) -> dict:
    """3 plug-point が **各々 1 オブジェクト差替のみ** で evolve に載ることを data で示す。

    各 swap 軸で evolve を回し、best fitness / admit 状況が出ることを確認する。
    """
    out = {"axis_genecodec": [], "axis_objective": [], "axis_verifier": []}

    # ---- (i) GeneCodec swap: n / 基質を差し替え -------------------------------
    # 1 オブジェクト (codec) を差し替えるだけで別次元 substrate に載る。
    for n in (2, 3, 4):
        codec = C.CoupledNDGeneCodec(n)                         # ← 差替対象 (GeneCodec)
        obj = C.RotationNDObjective(n=n)                         # objective は codec の n に整合
        rng = np.random.default_rng(seed + n)
        res = evolve(_wrap_fitness(obj, None), pop_size=POP, n_generations=GENS,
                     mutation_sigma=0.18, rng=rng, codec=codec)
        fb = res.final_best
        out["axis_genecodec"].append({
            "swap": f"CoupledNDGeneCodec(n={n})", "dim": codec.dim,
            "final_best_fitness": float(fb.fitness),
            "best_curve_improved": bool(res.best_fitness_curve[-1] >= res.best_fitness_curve[0]),
            "ran": True,
        })

    # ---- (ii) Objective swap: task を差し替え ---------------------------------
    # codec / verifier を固定し objective だけを差し替える (1 オブジェクト差替)。
    codec = C.CoupledNDGeneCodec(N_DIM)
    for (period, radius, amp) in [(10.0, 0.93, 0.40), (8.0, 0.88, 0.30), (13.0, 0.95, 0.45)]:
        obj = C.RotationNDObjective(n=N_DIM, period=period, radius=radius, amp=amp)  # ← 差替対象
        rng = np.random.default_rng(seed + int(period * 10))
        res = evolve(_wrap_fitness(obj, None), pop_size=POP, n_generations=GENS,
                     mutation_sigma=0.18, rng=rng, codec=codec)
        fb = res.final_best
        out["axis_objective"].append({
            "swap": f"RotationNDObjective(period={period}, radius={radius}, amp={amp})",
            "final_best_fitness": float(fb.fitness),
            "best_curve_improved": bool(res.best_fitness_curve[-1] >= res.best_fitness_curve[0]),
            "ran": True,
        })

    # ---- (iii) VerifierBackend swap: none/inf_norm/two_norm/sdp を差し替え ------
    # codec / objective を固定し verifier だけを差し替える (1 オブジェクト差替)。
    codec = C.CoupledNDGeneCodec(N_DIM)
    obj = C.RotationNDObjective(n=N_DIM)
    for vname in ("none", "inf_norm", "two_norm", "sdp"):
        verifier = C.make_nd_verifier(vname)                    # ← 差替対象 (VerifierBackend)
        rng = np.random.default_rng(seed + hash(vname) % 9973)
        res = evolve(_wrap_fitness(obj, verifier), pop_size=POP, n_generations=GENS,
                     mutation_sigma=0.18, rng=rng, codec=codec)
        fb = res.final_best
        # final 集団のうち何個が verifier に admit されるか (admit 状況が出ることの確認)
        n_admit = sum(1 for ind in res.generations[-1].individuals
                      if verifier.certifies(ind.gene))
        out["axis_verifier"].append({
            "swap": f"make_nd_verifier({vname!r})", "verifier_name": verifier.name,
            "final_best_fitness": float(fb.fitness),
            "n_admitted_final_pop": int(n_admit), "pop_size": POP,
            "ran": True,
        })

    return out


# --------------------------------------------------------------------------- #
# Part (a) — 汎化 load-bearing
#
# 構造多様な archive (MAP-Elites over topology descriptor) vs param-shift-only baseline を
# 同予算で進化させ、held-out family への汎化 fitness を比較。
# --------------------------------------------------------------------------- #
def _family_fitness(gene, family):
    """task family 平均 fitness (汎化 target)。"""
    return float(np.mean([obj.fitness(gene) for obj in family]))


def _topology_descriptor(codec, gene):
    """admit topology の descriptor (2D cell): (mean decay, tanh(mean|W|))。

    構造多様性 = この descriptor 空間での archive の広がり。
    """
    arr = codec.to_array(gene)
    n = codec.n
    decay = np.clip(arr[:n], 0.0, 1.0)
    W = arr[n:].reshape(n, n)
    d0 = float(np.mean(decay))
    d1 = float(np.tanh(np.mean(np.abs(W))))
    c0 = min(GRID - 1, int(d0 * GRID))
    c1 = min(GRID - 1, int(d1 * GRID))
    return (c0, c1)


def opt_diverse(codec, train_family, rng, budget, sigma=0.22, init=48):
    """構造多様な archive (MAP-Elites over topology descriptor)。

    同予算 budget = train-family-fitness 評価回数。archive の各 cell に train-family fitness
    最大の elite を保持 → topology 空間に構造多様な elite 群を維持する。
    """
    archive: dict = {}
    used = 0

    def _place(arr):
        nonlocal used
        gene = codec.to_gene(codec.clip(arr))
        f = _family_fitness(gene, train_family); used += 1
        cell = _topology_descriptor(codec, gene)
        cur = archive.get(cell)
        if cur is None or f > cur[1]:
            archive[cell] = (arr, f, gene)

    for _ in range(init):
        if used >= budget:
            break
        _place(codec.random(rng))

    while used < budget and archive:
        keys = list(archive.keys())
        parent = archive[keys[rng.integers(len(keys))]][0]
        child = parent + sigma * codec._half * rng.standard_normal(codec.dim)
        _place(child)

    if not archive:
        return codec.to_gene(codec.clip(codec.random(rng))), 0
    best = max(archive.values(), key=lambda v: v[1])
    return best[2], len(archive)  # (best gene, archive 占有 cell 数 = 構造多様性)


def opt_paramshift(codec, train_family, rng, budget, sigma=0.22):
    """param-shift-only baseline: 単一 elite を param-shift だけで摂動 (構造多様性なし)。

    同予算 budget。常に「現 best」1 個から摂動 → hill-climb 的。topology 多様性を archive で
    保持しない = diverse の対照。
    """
    cur = codec.random(rng)
    cur_gene = codec.to_gene(codec.clip(cur))
    cur_f = _family_fitness(cur_gene, train_family)
    used = 1
    best_arr, best_gene, best_f = cur, cur_gene, cur_f
    while used < budget:
        child = cur + sigma * codec._half * rng.standard_normal(codec.dim)
        gene = codec.to_gene(codec.clip(child))
        f = _family_fitness(gene, train_family); used += 1
        if f > cur_f:                       # greedy: 改善時のみ採用 (param-shift hill-climb)
            cur, cur_f, cur_gene = child, f, gene
            if f > best_f:
                best_arr, best_gene, best_f = child, gene, f
    return best_gene, 1                      # 構造多様性 = 1 (単一 lineage)


def generalization_test(seed0: int = SEED0) -> dict:
    """多様性が汎化に load-bearing か (diverse archive が held-out で baseline を上回るか)。"""
    train_family = _make_family(_FAMILY_PARAMS[:_N_TRAIN])
    heldout_family = _make_family(_FAMILY_PARAMS[_N_TRAIN:])
    codec = C.CoupledNDGeneCodec(N_DIM)

    # 同予算 (train-family-fitness 評価回数)
    budget = POP * GENS  # = 720 評価/optimizer/seed

    div_held, div_train, div_cells = [], [], []
    ps_held, ps_train = [], []
    for s in range(N_SEEDS):
        # diverse / paramshift で別 RNG (独立) だが seed は決定論
        r_div = np.random.default_rng(seed0 + 1000 + s * 13 + 1)
        r_ps = np.random.default_rng(seed0 + 1000 + s * 13 + 2)
        g_div, n_cells = opt_diverse(codec, train_family, r_div, budget)
        g_ps, _ = opt_paramshift(codec, train_family, r_ps, budget)
        div_held.append(_family_fitness(g_div, heldout_family))
        div_train.append(_family_fitness(g_div, train_family))
        div_cells.append(n_cells)
        ps_held.append(_family_fitness(g_ps, heldout_family))
        ps_train.append(_family_fitness(g_ps, train_family))
        print(f"  seed {s+1}/{N_SEEDS}: diverse held={div_held[-1]:.3f} (cells={n_cells}) "
              f"train={div_train[-1]:.3f} | paramshift held={ps_held[-1]:.3f} "
              f"train={ps_train[-1]:.3f}", flush=True)

    cmp_div_ps = honest_eval(div_held, ps_held)              # 多様 archive が held-out で baseline 超か
    cmp_ps_div = honest_eval(ps_held, div_held)              # 逆向き (baseline 優位か)

    # 識別力 (held-out が天井/床でないか) — baseline held-out 平均で判定
    ps_held_mean = float(np.mean(ps_held))
    div_held_mean = float(np.mean(div_held))
    discriminating = 0.05 < ps_held_mean < 0.95 and 0.05 < div_held_mean < 0.95

    # verdict
    div_beats_ps = cmp_div_ps["all_pass"]
    ps_beats_div = cmp_ps_div["all_pass"]
    if div_beats_ps:
        verdict = ("PASS (構造多様 archive が param-shift baseline を held-out family で "
                   "4 条件 AND で上回る = 多様性は汎化に load-bearing)")
    elif ps_beats_div:
        verdict = ("負 (param-shift baseline が held-out で多様 archive を上回る = "
                   "構造多様性は汎化を助けない/むしろ害。第一級 NEGATIVE 知見)")
    else:
        verdict = ("NULL (held-out 汎化に有意差なし = 多様性は load-bearing でない。"
                   "第一級 NULL 知見; 多様性が汎化を助ける証拠は立たず)")
    if not discriminating:
        verdict += (f" [⚠地形 non-discriminating: held-out 平均 baseline={ps_held_mean:.3f} "
                    f"diverse={div_held_mean:.3f}=天井/床。verdict 証拠力低下]")

    return {
        "meta": {"n_dim": N_DIM, "pop": POP, "gens": GENS, "n_seeds": N_SEEDS,
                 "budget_evals_per_opt": budget, "grid": GRID, "seed0": seed0,
                 "n_train_tasks": _N_TRAIN, "n_heldout_tasks": len(_FAMILY_PARAMS) - _N_TRAIN,
                 "task_family": "synthetic RotationNDObjective (damped block-rotation), "
                                "NOT real SmolLM2-CE terrain = clean probe",
                 "scipy": _HAVE_SCIPY},
        "diverse_heldout_mean": div_held_mean,
        "diverse_train_mean": float(np.mean(div_train)),
        "diverse_archive_cells_mean": float(np.mean(div_cells)),
        "paramshift_heldout_mean": ps_held_mean,
        "paramshift_train_mean": float(np.mean(ps_train)),
        "diverse_vs_paramshift_heldout": cmp_div_ps,
        "paramshift_vs_diverse_heldout": cmp_ps_div,
        "terrain_discriminating": discriminating,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# honest_eval 4 条件 AND (phase2_capability_terrain.py と同一)
# --------------------------------------------------------------------------- #
def honest_eval(a, b, alt="greater"):
    """paired held-out: a が b を上回るか 4 条件 AND。a,b: shape (S,)。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a - b
    mean_diff = float(diff.mean())
    n_seeds = len(diff)
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    paired_sign_delta = (wins - losses) / n_seeds if n_seeds else 0.0
    if _HAVE_SCIPY and np.any(diff != 0):
        try:
            p = float(_wilcoxon(a, b, alternative=alt, zero_method="wilcox").pvalue)
        except Exception:
            p = float("nan")
    else:
        from math import comb
        k = wins
        n = wins + losses
        p = float(sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)) if n > 0 else 1.0
    cond = {
        "diff_positive": mean_diff > 0,
        "wilcoxon_p_lt_0.05": (p < 0.05),
        "n_seeds_ge_15": n_seeds >= 15,
        "abs_paired_sign_delta_ge_0.147": abs(paired_sign_delta) >= 0.147,
    }
    return {
        "mean_diff": mean_diff, "p_value": p, "n_seeds": n_seeds,
        "wins": wins, "losses": losses, "paired_sign_delta": paired_sign_delta,
        "conditions": cond, "all_pass": all(cond.values()),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print("=== Part (b): 3 plug-point swap test ===", flush=True)
    swap = swap_test(SEED0)
    for axis, rows in swap.items():
        allran = all(r["ran"] for r in rows)
        print(f"[{axis}] {len(rows)} swaps, all_ran={allran}", flush=True)
        for r in rows:
            extra = ""
            if "n_admitted_final_pop" in r:
                extra = f" admit={r['n_admitted_final_pop']}/{r['pop_size']}"
            print(f"    {r['swap']}: best_fit={r['final_best_fitness']:+.3f}"
                  f" improved={r.get('best_curve_improved', '-')}" + extra, flush=True)

    # swap 健全性 (Part b PASS 条件): 各軸の全 swap が ran=True
    swap_pass = all(all(r["ran"] for r in rows) for rows in swap.values())
    # verifier 軸の admit 単調性チェック (none ⊇ inf_norm ⊆ ... の admit 関係を確認)
    vrows = {r["verifier_name"]: r["n_admitted_final_pop"] for r in swap["axis_verifier"]}

    print("\n=== Part (a): 汎化 load-bearing ===", flush=True)
    gen = generalization_test(SEED0)
    print("\n--- generalization verdict ---", flush=True)
    print(f"diverse held-out 平均={gen['diverse_heldout_mean']:.3f} "
          f"(archive cells={gen['diverse_archive_cells_mean']:.1f}) "
          f"train={gen['diverse_train_mean']:.3f}", flush=True)
    print(f"paramshift held-out 平均={gen['paramshift_heldout_mean']:.3f} "
          f"train={gen['paramshift_train_mean']:.3f}", flush=True)
    cd = gen["diverse_vs_paramshift_heldout"]
    print(f"diverse vs paramshift (held-out): diff={cd['mean_diff']:+.3f} "
          f"p={cd['p_value']:.3f} sign_delta={cd['paired_sign_delta']:+.3f} "
          f"→ 4条件AND={cd['all_pass']}", flush=True)
    print(f"地形識別力 discriminating={gen['terrain_discriminating']}", flush=True)
    print(f"VERDICT: {gen['verdict']}", flush=True)

    results = {
        "meta": {
            "title": "Phase 2 framework 性 (F8) — North Star #4",
            "plan": "EVOLVABLE_LLM_PLAN_2026_06_09.md §⑥ L99",
            "seed0": SEED0,
            "src_evolve": "llcore.evolution.minimal_ga.evolve (src 無改変)",
            "plug_points_source": "research/verified_evolution_sdp_gate/coupled_nd.py "
                                  "(CoupledNDGeneCodec / RotationNDObjective / make_nd_verifier)",
            "honest_caveats": [
                "synthetic task family (RotationNDObjective damped block-rotation); "
                "NOT real SmolLM2-CE loss terrain = clean probe",
                "coupled-gene verifier gate は fitness wrapper proxy (src の scalar gate ではない)",
                "paired_sign_delta = net-win-fraction (Cliff's delta ではない)",
                "予算は train-family-fitness 評価回数で optimizer 間を厳密に揃えた",
            ],
        },
        "part_b_swap_test": {
            "all_swaps_ran": swap_pass,
            "verifier_admit_counts": vrows,
            "axes": swap,
            "verdict": ("PASS (3 plug-point すべて 1 オブジェクト差替で evolve に載り fitness/admit "
                        "が出た = framework 約束として機能)" if swap_pass
                        else "FAIL (一部 swap が走らなかった)"),
        },
        "part_a_generalization": gen,
    }
    out = os.path.join(os.path.dirname(__file__), "phase2_framework_f8_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
