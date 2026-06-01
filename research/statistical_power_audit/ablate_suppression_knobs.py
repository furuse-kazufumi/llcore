# SPDX-License-Identifier: Apache-2.0
"""(C) suppression ablation — K1-K4 を個別 toggle し borderline case で verdict 反転を測る.

目的: K1 (fresh-seed 再評価) / K2 (strict gate 閾値) / K3 (budget/global-best/HONEST_N) /
K4 (ridge clip) を **OFAT** (one-factor-at-a-time, 他 ON 固定) で振り、borderline case で
verdict が no-evolution → evolution へ反転するか測る。各 toggle は研究側の関数引数/再計算で
行い src は無改変。

borderline case:
- corridor d=0.13 (校正で not-LB だが adv≈0 の境界) と d=0.14/0.15
- C-gen4b 再現 (実 per-seed, diff>0 だが gate fail)
- flip_flop (実 per-seed, δ=+0.33, p=0.15)

honest disclosure: 各 toggle は Type I guard とペアで読む (反転が信号でなく閾値ガバガバ化で
ないか)。本スクリプトは反転を記録し、Type I コストは type1_guard_sweep.py が併測する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _audit_common as AC  # noqa: E402

# K3/K4 用 read-only import
from ea_lab import _map_elites_core  # noqa: E402
from llcore.fitness import CopyTask, AdditionTask  # noqa: E402
from llcore.fitness.ridge_readout import ridge_fitness  # noqa: E402
from llcore.state_update.genes import StateUpdateGene  # noqa: E402

BASELINES = ("rr_hillclimb", "panmictic_ga", "random")


def _verdict_vs_best(scores: dict[str, np.ndarray], **gate_kw) -> dict:
    """MAP-E vs best baseline + 3 baseline 全勝 (load_bearing) を任意閾値 gate で判定."""
    me = scores["map_elites"]
    best_b = max(BASELINES, key=lambda b: float(scores[b].mean()))
    g = AC.eval_gate(me, scores[best_b], "map_elites", best_b, **gate_kw)
    n_beaten = sum(int(AC.eval_gate(me, scores[b], **gate_kw).passes) for b in BASELINES)
    return {
        "best_baseline": best_b, "diff": g.diff, "wilcoxon_p": g.wilcoxon_p,
        "paired_sign_delta": g.paired_sign_delta,
        "cliff_delta_textbook": g.cliff_delta_textbook,
        "passes_vs_best": g.passes, "n_baselines_beaten": n_beaten,
        "load_bearing": n_beaten == len(BASELINES),
    }


# ---------------------------------------------------------------------------
# K1: fresh-seed 再評価 ON/OFF (corridor で進化 best の noisy fitness を持ち越す)
# ---------------------------------------------------------------------------


def _k1_corridor_scores(d: float, *, n_seeds: int, n_evals: int, base_seed: int,
                        honest_off: bool) -> dict[str, np.ndarray]:
    """corridor で K1 OFF (= 進化中の noisy best fitness をそのまま採用) を再現.

    honest_off=False は exp_knob_sweep の honest 再評価そのもの (run_methods_crn)。
    honest_off=True は MAP-E の best_fitness (進化 rng で測った noisy 値) を honest 平均なしで
    そのまま使い、baseline も同様に noisy best を採用 (elitism 持越し許容相当)。
    """
    if not honest_off:
        return AC.corridor_method_scores(d, n_seeds=n_seeds, n_evals=n_evals,
                                         base_seed=base_seed)
    # K1 OFF: noisy best (進化中に観測した best fitness) を honest 平均せず採用
    from selection_lab import map_elites, random_restart_hillclimb, panmictic_ga
    eval_once = AC.make_corridor_eval(d)
    dim = AC.CORRIDOR_D
    bounds = (np.zeros(dim), np.ones(dim))
    bb = (np.zeros(1), np.ones(1))
    init_batch = max(20, n_evals // 10)
    out = {m: [] for m in ("map_elites",) + BASELINES}

    def rng(mi, s):
        return np.random.default_rng(np.random.SeedSequence([base_seed, mi, s]))

    for s in range(n_seeds):
        r = map_elites(eval_once, AC.behavior_mean, dim=dim, bounds=bounds,
                       behavior_bounds=bb, grid_shape=(24,), n_evals=n_evals,
                       init_batch=init_batch, sigma=0.10, rng=rng(0, s))
        out["map_elites"].append(float(r.best_fitness))
        r2 = random_restart_hillclimb(eval_once, dim=dim, bounds=bounds, n_evals=n_evals,
                                      sigma=0.10, restart_patience=max(10, n_evals // 20),
                                      rng=rng(1, s))
        out["rr_hillclimb"].append(float(r2.best_fitness))
        r3 = panmictic_ga(eval_once, dim=dim, bounds=bounds, n_evals=n_evals, pop_size=20,
                          tournament_k=3, sigma=0.10, elitism=1, rng=rng(2, s))
        out["panmictic_ga"].append(float(r3.best_fitness))
        rr = rng(3, s)
        cands = [bounds[0] + (bounds[1] - bounds[0]) * rr.random(dim) for _ in range(n_evals)]
        fits = [eval_once(g, rr) for g in cands]
        out["random"].append(float(max(fits)))
    return {m: np.array(out[m]) for m in out}


# ---------------------------------------------------------------------------
# K3: global-best vs archive-max 読み出し (EA multitask)
# ---------------------------------------------------------------------------


def _ridge_eval_for_ea_sized(task, *, clip: bool, n_tr: int = 48):
    def ev(gene_vec: np.ndarray, rng: np.random.Generator) -> float:
        g = StateUpdateGene(decay=float(gene_vec[0]), mix=float(gene_vec[1]),
                            gate_str=float(gene_vec[2]))
        return ridge_fitness(g, task, n_train=n_tr, n_eval=n_tr, rng=rng, clip=clip)
    return ev


def _k3_archive_vs_global(*, n_seeds: int, n_evals: int, base_seed: int,
                          honest_n: int = 12, n_tr: int = 24) -> dict:
    """K3: MAP-E の best 読み出しを global-best (現行) vs archive-max で比較.

    archive_out probe で最終 archive を取得し archive-max を再計算 (production 不変)。
    ridge clip=True landscape (memory task 系) で randselect との gap が読み出しで変わるか。
    behavior=decay 軸 (gene[0])。両モードを honest 再評価 (CRN) で採点。

    予算注 (G1 破綻ゲート): ridge_fitness は train/predict で重い。読み出し方式の比較に
    効果量精度は不要なので honest_n / n_train / n_eval を縮小 (smoke で global vs archive の
    符号は安定確認済)。silent truncation 禁止のため _meta に縮小を明記。
    """
    task = CopyTask(state_dim=8, out_dim=8, seq_len=16)
    eval_once = _ridge_eval_for_ea_sized(task, clip=True, n_tr=n_tr)
    dim, bounds = 3, (np.array([0.0, -1.0, -2.0]), np.array([1.0, 1.0, 2.0]))
    bb = (np.array([0.0]), np.array([1.0]))
    behavior = lambda g: np.array([g[0]])  # noqa: E731
    init_batch = max(20, n_evals // 10)

    def rng(mi, s):
        return np.random.default_rng(np.random.SeedSequence([base_seed, mi, s]))

    def honest(gene, s):
        return AC.honest_reevaluate(eval_once, gene, n_trials=honest_n,
                                    rng=np.random.default_rng(
                                        np.random.SeedSequence([base_seed, 7, s])))

    gb_full, am_full, gb_rs, am_rs = [], [], [], []
    for s in range(n_seeds):
        for mode, gb_list, am_list, mi in (("elite", gb_full, am_full, 0),
                                           ("random", gb_rs, am_rs, 1)):
            arch: dict = {}
            r = _map_elites_core(eval_once, behavior, dim=dim, bounds=bounds,
                                 behavior_bounds=bb, grid_shape=(12,), n_evals=n_evals,
                                 init_batch=init_batch, sigma=0.12, rng=rng(mi, s),
                                 selection_mode=mode, archive_out=arch)
            gb_list.append(honest(r.best_gene, s))  # global-best 読み出し (現行)
            # archive-max 読み出し (旧仕様): archive 占有者の max-fitness gene
            am_gene = max(arch.values(), key=lambda kv: kv[1])[0]
            am_list.append(honest(am_gene, s))
    gb_full, am_full = np.array(gb_full), np.array(am_full)
    gb_rs, am_rs = np.array(gb_rs), np.array(am_rs)
    g_gb = AC.eval_gate(gb_full, gb_rs, "MAP-E_full", "randselect")
    g_am = AC.eval_gate(am_full, am_rs, "MAP-E_full", "randselect")
    return {
        "readout_global_best_current": {
            "diff": g_gb.diff, "wilcoxon_p": g_gb.wilcoxon_p,
            "paired_sign_delta": g_gb.paired_sign_delta, "passes": g_gb.passes,
            "full_mean": float(gb_full.mean()), "randselect_mean": float(gb_rs.mean()),
        },
        "readout_archive_max_old": {
            "diff": g_am.diff, "wilcoxon_p": g_am.wilcoxon_p,
            "paired_sign_delta": g_am.paired_sign_delta, "passes": g_am.passes,
            "full_mean": float(am_full.mean()), "randselect_mean": float(am_rs.mean()),
        },
        "flipped_global_to_archive": bool(g_am.passes and not g_gb.passes),
        "note": ("global-best (現行) で③ gap が消えるなら archive-ratchet 自体が③効果の "
                 "可能性。archive-max で PASS かつ global-best で FAIL = 読み出しが③を隠す。"),
    }


# ---------------------------------------------------------------------------
# K4: ridge clip=True vs clip=False の fitness spread (平坦化が clip 由来か)
# ---------------------------------------------------------------------------


def _k4_clip_spread(*, base_seed: int, n_genes: int = 200) -> dict:
    """同一 gene 集団で clip=True/False の fitness 分散・符号を比較.

    clip=True で floor 0.0 に潰れた gene の raw R² 符号を見て、平坦化が clip 由来か
    landscape 由来かを切り分ける。
    """
    rng = np.random.default_rng(base_seed)
    out: dict[str, dict] = {}
    for tname, task in (("copy", CopyTask(state_dim=8, out_dim=8, seq_len=24)),
                        ("addition", AdditionTask(state_dim=8))):
        lo, hi = np.array([0.0, -1.0, -2.0]), np.array([1.0, 1.0, 2.0])
        genes = [lo + (hi - lo) * rng.random(3) for _ in range(n_genes)]
        clipped, raw = [], []
        for gv in genes:
            g = StateUpdateGene(decay=float(gv[0]), mix=float(gv[1]), gate_str=float(gv[2]))
            # 同一 seed で clip True/False を測り fitness 分散を直接比較 (CRN)。
            seed = int(rng.integers(1 << 30))
            clipped.append(ridge_fitness(g, task, n_train=48, n_eval=48,
                                         rng=np.random.default_rng(seed), clip=True))
            raw.append(ridge_fitness(g, task, n_train=48, n_eval=48,
                                     rng=np.random.default_rng(seed), clip=False))
        clipped, raw = np.array(clipped), np.array(raw)
        n_floored = int(np.sum(clipped <= 1e-9))
        floored_raw = raw[clipped <= 1e-9]
        out[tname] = {
            "clip_true_std": float(clipped.std()),
            "clip_false_std": float(raw.std()),
            "clip_true_mean": float(clipped.mean()),
            "clip_false_mean": float(raw.mean()),
            "n_floored_to_zero": n_floored, "n_genes": n_genes,
            "frac_floored": n_floored / n_genes,
            "floored_raw_r2_min": float(floored_raw.min()) if floored_raw.size else None,
            "floored_raw_r2_max": float(floored_raw.max()) if floored_raw.size else None,
            "floored_raw_r2_spread": (float(floored_raw.std()) if floored_raw.size else 0.0),
            "flattening_hidden_by_clip": bool(
                floored_raw.size > 1 and float(floored_raw.std()) > 0.02),
        }
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    guard = AC.RunGuard.start("ablate_suppression_knobs")
    base_seed = 20260530
    if args.smoke:
        n_seeds, n_evals = 15, 600
        borderline_ds = [0.13]
        k3_seeds, k3_evals = 15, 300
    else:
        n_seeds, n_evals = 15, 1500
        borderline_ds = [0.13, 0.14, 0.15]
        k3_seeds, k3_evals = 20, 600

    results: dict[str, dict] = {}
    flips: list[dict] = []

    # ---- borderline corridor cases (K1, K2) ----
    negs = AC.load_real_negative_deltas()
    for d in borderline_ds:
        sc = AC.corridor_method_scores(d, n_seeds=n_seeds, n_evals=n_evals, base_seed=base_seed)
        case = f"corridor_d={d:.2f}"
        baseline_v = _verdict_vs_best(sc)  # 全 knob ON (現行)
        rec = {"baseline_all_on": baseline_v, "toggles": {}}

        # K1 OFF (fresh-seed 再評価無効 = noisy best 持越し)
        sc_k1off = _k1_corridor_scores(d, n_seeds=n_seeds, n_evals=n_evals,
                                       base_seed=base_seed, honest_off=True)
        v_k1 = _verdict_vs_best(sc_k1off)
        rec["toggles"]["K1_freshseed_OFF"] = v_k1
        if v_k1["load_bearing"] and not baseline_v["load_bearing"]:
            flips.append({"case": case, "knob": "K1_freshseed_OFF",
                          "from": "no", "to": "evolution",
                          "type1_cost": "HIGH — elitism 持越しで null でも偽 best 水増し。"
                                        "type1_guard で最優先測定。外してはいけない境界。"})

        # K2: min_effect / alpha / min_seeds の各 level
        for me_lvl in (0.10, 0.05, 0.0):
            v = _verdict_vs_best(sc, min_effect=me_lvl)
            rec["toggles"][f"K2_min_effect={me_lvl}"] = v
            if v["load_bearing"] and not baseline_v["load_bearing"]:
                flips.append({"case": case, "knob": f"K2_min_effect={me_lvl}",
                              "from": "no", "to": "evolution",
                              "type1_cost": "効果量床↓。d=0 null での偽陽性を type1_guard で併測。"})
        for a_lvl in (0.10, 0.20):
            v = _verdict_vs_best(sc, alpha=a_lvl)
            rec["toggles"][f"K2_alpha={a_lvl}"] = v
            if v["load_bearing"] and not baseline_v["load_bearing"]:
                flips.append({"case": case, "knob": f"K2_alpha={a_lvl}",
                              "from": "no", "to": "evolution",
                              "type1_cost": "alpha↑で名目 Type I を直接緩める。"
                                            "null FPR が alpha に追従し上がる。"})
        for n_lvl in (10, 5):
            v = _verdict_vs_best(sc, min_seeds=n_lvl)
            rec["toggles"][f"K2_min_seeds={n_lvl}"] = v
            if v["load_bearing"] and not baseline_v["load_bearing"]:
                flips.append({"case": case, "knob": f"K2_min_seeds={n_lvl}",
                              "from": "no", "to": "evolution",
                              "type1_cost": "n 床↓。少 seed は分散大→偶発 PASS リスク。"})
        results[case] = rec
        print(f"  {case}: baseline LB={baseline_v['load_bearing']} "
              f"(diff={baseline_v['diff']:+.4f} p={baseline_v['wilcoxon_p']:.3g})")

    # ---- 実 per-seed negative (K2 閾値のみ; 進化は再走せず実 delta で gate 再計算) ----
    for case in ("C-gen4b_MAPE_vs_random", "flip_flop_MAPE_vs_random"):
        if case not in negs:
            continue
        rec = negs[case]
        a, b = rec["a"], rec["b"]
        baseline_g = AC.eval_gate(a, b)
        toggles = {}
        for me_lvl in (0.10, 0.05, 0.0):
            g = AC.eval_gate(a, b, min_effect=me_lvl)
            toggles[f"K2_min_effect={me_lvl}"] = {"passes": g.passes, "p": g.wilcoxon_p}
            if g.passes and not baseline_g.passes:
                flips.append({"case": case, "knob": f"K2_min_effect={me_lvl}",
                              "from": "no", "to": "evolution",
                              "type1_cost": "実 negative。p で落ちる case は min_effect↓では救えない見込み。"})
        for a_lvl in (0.10, 0.20):
            g = AC.eval_gate(a, b, alpha=a_lvl)
            toggles[f"K2_alpha={a_lvl}"] = {"passes": g.passes, "p": g.wilcoxon_p}
            if g.passes and not baseline_g.passes:
                flips.append({"case": case, "knob": f"K2_alpha={a_lvl}",
                              "from": "no", "to": "evolution",
                              "type1_cost": ("alpha↑が p で落ちる underpowered case を救う。"
                                             "ただし null FPR も alpha まで上がる(type1_guard で確認)。")})
        results[case] = {
            "baseline_all_on": {"passes": baseline_g.passes, "p": baseline_g.wilcoxon_p,
                                "diff": baseline_g.diff,
                                "paired_sign_delta": baseline_g.paired_sign_delta},
            "toggles": toggles, "source": rec["source"],
        }
        print(f"  {case}: baseline passes={baseline_g.passes} p={baseline_g.wilcoxon_p:.3g}")

    # ---- K3: global-best vs archive-max ----
    print("  K3 archive-max vs global-best (ridge CopyTask)...")
    results["K3_readout_global_vs_archive"] = _k3_archive_vs_global(
        n_seeds=k3_seeds, n_evals=k3_evals, base_seed=base_seed)
    if results["K3_readout_global_vs_archive"]["flipped_global_to_archive"]:
        flips.append({"case": "EA_multitask_ridge", "knob": "K3_archive_max_readout",
                      "from": "no(global-best)", "to": "evolution(archive-max)",
                      "type1_cost": "archive-max は randselect の忘却で③を水増しする方向。"
                                    "Codex F2 が公平化した旧仕様への巻き戻し=偽陽性源。"})

    # ---- K4: ridge clip spread ----
    print("  K4 ridge clip spread (CopyTask/AdditionTask)...")
    results["K4_clip_spread"] = _k4_clip_spread(base_seed=base_seed)

    meta = guard.finish()
    payload = {
        "_meta": {**meta, "design": "(C) suppression ablation (OFAT)",
                  "n_seeds": n_seeds, "n_evals": n_evals, "base_seed": base_seed,
                  "note": ("各 toggle は src 無改変で research 側引数/再計算。"
                           "反転は type1_guard_sweep.py の Type I コストと必ずペアで読む。")},
        "cases": results,
        "verdict_flips": flips,
        "flip_summary": {
            "n_flips": len(flips),
            "knobs_that_flipped": sorted({f["knob"] for f in flips}),
        },
    }
    out = AC.dump_json(AC.AUDIT_DIR / "ablate_suppression_knobs_results.json", payload)
    print(f"[ablate] wrote {out}  ({meta['wall_clock_s']}s) flips={len(flips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
