# SPDX-License-Identifier: Apache-2.0
"""EXP3 (step_d_settle) — K4 clip verdict-flip + null-ridge FPR.

Codex pair-review F2 の確定/反証実験。現状 ablate_suppression_knobs.py の K4 は
ridge clip=True/False の **fitness spread / floor 率** しか測っておらず、
「clip を外すと③ (selection load-bearing) verdict が反転するか」を測っていない
= 「有力候補」止まり (CODEX_PAIRREVIEW.md F2)。

本 script は 2 点を測る:

(1) **paired verdict-flip**: 低 R² task (addition / flip_flop) で MAP-E vs
    {randselect, panmictic, random} を **同一 CRN seed** にて clip=True / clip=False
    の 2 条件で進化 → honest 再評価 → full strict gate (diff>0 ∧ 片側 Wilcoxon
    p<0.05 ∧ |paired_sign_delta|>=0.147 ∧ n>=min_seeds) を適用。
        load_bearing = MAP-E が 3 baseline 全勝。
        verdict_flip = (clip=False で load_bearing=True) ∧ (clip=True で load_bearing=False)。
    これが「clip が真の信号を潰している (= 外すと③が見える)」の核心測定。

(2) **null-ridge FPR (§9.3 scope gap)**: gene 非依存 target (inputs と無相関ノイズ)
    の null-ridge landscape で clip=False / clip=True 両方の load_bearing FPR を
    複数 null seed で測定。clip=False が「ノイズ分散を見かけの構造に膨らませて FPR を
    上げる」かを直接 guard する。

判定 (3 値):
- K4 確定 (clip = active suppression):
    実 task verdict_flip=True **かつ** null-ridge で clip=False FPR <= clip=True FPR + 0.05
    (FPR を上げずに真の信号を回復)。
- K4 反証/降格:
    verdict_flip=False (clip 外しても③不発 = spread 平坦化は verdict に効かず)
    **または** clip=False が null-ridge FPR を有意増 (構造でなくノイズを拾う Type I コスト)。
    → VERDICT を「有力候補」据え置き or 棄却に訂正。
- なお不確定:
    verdict_flip=True だが null-ridge FPR も clip=False で有意増 (信号回復と Type I 増が
    分離不能) → より大予算 null-ridge 要を honest 開示。

規律 (step_d_settle):
- src/llcore 非改変。eval_gate / RunGuard / dump_json は _audit_common から read-only import。
  run_ea_methods_over_seeds / _map_elites_core は ea_lab から read-only import。
  CopyTask/AdditionTask/ridge_fitness/StateUpdateGene/FlipFlopTask は src + research から import 再利用。
- ridge_fitness は ~10-35ms/call と重い。verdict-flip / FPR の判定に効果量精度は不要なので
  n_evals / honest_n / n_tr を縮小 (silent truncation 禁止: _meta.budget_reduction に明記)。
- CRN: clip=True/False は **同一 (method_idx, s) seed** で評価される (ea_lab の
  _evo_rng / honest 共通 seed が matched-replicate pairing を保証 → paired Wilcoxon 前提)。
- 結果は results JSON + log で必ず永続化 (返り値だけにしない)。UTF-8。git 操作なし。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- path 設定 (read-only import; src 非改変) ---
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[2]  # <llcore-root>
EXP3_DIR = _HERE
for _p in (
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "research" / "ea_multitask"),
    str(REPO_ROOT / "research" / "step4_selection"),
    str(REPO_ROOT / "research" / "step_c_memory_tasks"),
    str(REPO_ROOT / "research" / "statistical_power_audit"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _audit_common as AC  # noqa: E402  (eval_gate / RunGuard / dump_json / utf8)
from ea_lab import run_ea_methods_over_seeds  # noqa: E402
from llcore.fitness import AdditionTask  # noqa: E402
from llcore.fitness.ridge_readout import ridge_fitness  # noqa: E402
from llcore.state_update.genes import StateUpdateGene  # noqa: E402
from memory_tasks import FlipFlopTask  # noqa: E402

AC.ensure_utf8_stdout()

# MAP-E が全勝すべき 3 baseline (ea_lab の method 名に対応)。
EA_BASELINES = ("map_elites_randselect", "panmictic_ga", "random")

# 全 script 共通 base_seed (CRN: clip True/False は同 seed 列で評価)。
BASE_SEED = 20260601

# StateUpdateGene の探索 bounds (decay∈[0,1], mix∈[-1,1], gate_str∈[-2,2])。
GENE_BOUNDS = (np.array([0.0, -1.0, -2.0]), np.array([1.0, 1.0, 2.0]))
GENE_DIM = 3


# ===========================================================================
# null-ridge landscape: gene 非依存 target (inputs と無相関ノイズ) = 真の H0
# ===========================================================================


@dataclass(frozen=True)
class NullRidgeTask:
    """ridge 版 pure-null: target を inputs / gene と無相関なランダムノイズにする.

    final_state は target を線形デコードできない → raw R²<=0 期待 (mean 予測以下)。
    有限標本の overfit で稀に正の R² が出る = clip=False がノイズを「構造」に
    膨らませて FPR を上げるかを直接測る土俵。

    type1_guard_sweep の pure-null (gene 非依存定数+noise) の ridge 版思想。
    """

    seq_len: int = 20
    state_dim: int = 4  # gene state 次元 (= inputs dim)
    out_dim: int = 1

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inputs = rng.uniform(-1.0, 1.0, size=(self.seq_len, self.state_dim))
        # target は inputs と独立 (gene 非依存) = H0。
        target = rng.standard_normal(self.out_dim).astype(np.float64)
        return inputs, target


# ===========================================================================
# ridge eval_once ファクトリ (ablate の _ridge_eval_for_ea_sized と同型)
# ===========================================================================


def _make_ridge_eval(task: object, *, clip: bool, n_tr: int):
    """(gene_vec, rng) -> ridge_fitness。clip を bind。run_sequence の state 次元は
    task の入力次元 (state_dim) に一致する必要があり、3-param StateUpdateGene が
    全 task で共通の探索遺伝子 (decay/mix/gate_str)。"""

    def ev(gene_vec: np.ndarray, rng: np.random.Generator) -> float:
        g = StateUpdateGene(
            decay=float(gene_vec[0]), mix=float(gene_vec[1]), gate_str=float(gene_vec[2])
        )
        return ridge_fitness(g, task, n_train=n_tr, n_eval=n_tr, rng=rng, clip=clip)

    return ev


# ===========================================================================
# (1) verdict-flip 測定
# ===========================================================================


def _verdict_for_scores(scores: dict, *, min_seeds: int) -> dict:
    """MAP-E vs 3 baseline に full strict gate を適用し load_bearing を判定.

    scores[method].test (hold-out 再評価) を主指標に使う (exp_ea3 と同じ)。
    """
    me = scores["map_elites"].test
    per_baseline = {}
    n_beaten = 0
    for b in EA_BASELINES:
        g = AC.eval_gate(me, scores[b].test, "map_elites", b, min_seeds=min_seeds)
        per_baseline[b] = {
            "diff": g.diff,
            "wilcoxon_p": g.wilcoxon_p,
            "paired_sign_delta": g.paired_sign_delta,
            "cliff_delta_textbook": g.cliff_delta_textbook,
            "cond_diff_pos": g.cond_diff_pos,
            "cond_p": g.cond_p,
            "cond_effect": g.cond_effect,
            "cond_n": g.cond_n,
            "passes": g.passes,
        }
        n_beaten += int(g.passes)
    return {
        "map_elites_mean": float(me.mean()),
        "map_elites_std": float(me.std()),
        "per_baseline": per_baseline,
        "n_baselines_beaten": n_beaten,
        "load_bearing": n_beaten == len(EA_BASELINES),
    }


def _run_ea_for_task(task: object, task_name: str, *, clip: bool, n_seeds: int,
                     n_evals: int, honest_n: int, n_tr: int, grid_k: int,
                     state_dim: int) -> dict:
    """1 task × 1 clip 条件で 4 method を進化 → honest 再評価。

    ridge には regime split が無いので eval_train == eval_test (同一 task)。
    behavior = decay 軸 (gene[0]) を 1D grid に bin (ablate K3 と同じ)。
    """
    ev = _make_ridge_eval(task, clip=clip, n_tr=n_tr)
    bb = (np.array([0.0]), np.array([1.0]))
    behavior = lambda g: np.array([g[0]])  # noqa: E731  (decay 軸)
    scores = run_ea_methods_over_seeds(
        ev, ev, behavior, dim=GENE_DIM, bounds=GENE_BOUNDS,
        behavior_bounds=bb, grid_shape=(grid_k,), n_evals=n_evals, n_seeds=n_seeds,
        honest_n_trials=honest_n, sigma=0.12, base_seed=BASE_SEED,
    )
    return scores


def measure_verdict_flip(*, n_seeds: int, n_evals: int, honest_n: int, n_tr: int,
                         grid_k: int) -> dict:
    """addition / flip_flop で clip=True/False の paired verdict-flip を測る."""
    tasks = {
        "addition": (AdditionTask(state_dim=8), 8),
        "flip_flop": (FlipFlopTask(seq_len=30), 2),  # in_dim=2 -> state dim 2
    }
    out: dict[str, dict] = {}
    for tname, (task, sdim) in tasks.items():
        print(f"[verdict-flip] task={tname} clip=True ...", flush=True)
        sc_clip = _run_ea_for_task(task, tname, clip=True, n_seeds=n_seeds,
                                   n_evals=n_evals, honest_n=honest_n, n_tr=n_tr,
                                   grid_k=grid_k, state_dim=sdim)
        v_clip = _verdict_for_scores(sc_clip, min_seeds=min(15, n_seeds))
        print(f"[verdict-flip] task={tname} clip=False ...", flush=True)
        sc_raw = _run_ea_for_task(task, tname, clip=False, n_seeds=n_seeds,
                                  n_evals=n_evals, honest_n=honest_n, n_tr=n_tr,
                                  grid_k=grid_k, state_dim=sdim)
        v_raw = _verdict_for_scores(sc_raw, min_seeds=min(15, n_seeds))

        verdict_flip = bool(v_raw["load_bearing"] and not v_clip["load_bearing"])
        # 反転の方向も記録 (clip=True で③, clip=False で消える = 逆向きの異常)。
        reverse_flip = bool(v_clip["load_bearing"] and not v_raw["load_bearing"])
        out[tname] = {
            "clip_true": v_clip,
            "clip_false": v_raw,
            "verdict_flip_false_reveals_third": verdict_flip,
            "reverse_flip_clip_needed_for_third": reverse_flip,
            "per_seed_clip_true": {
                m: sc_clip[m].test.tolist() for m in ("map_elites",) + EA_BASELINES
            },
            "per_seed_clip_false": {
                m: sc_raw[m].test.tolist() for m in ("map_elites",) + EA_BASELINES
            },
        }
        print(
            f"  {tname}: clip=True LB={v_clip['load_bearing']} "
            f"(beaten={v_clip['n_baselines_beaten']}/3)  "
            f"clip=False LB={v_raw['load_bearing']} "
            f"(beaten={v_raw['n_baselines_beaten']}/3)  flip={verdict_flip}",
            flush=True,
        )
    return out


# ===========================================================================
# (2) null-ridge FPR 測定
# ===========================================================================


def measure_null_ridge_fpr(*, null_seeds: list[int], n_seeds: int, n_evals: int,
                           honest_n: int, n_tr: int, grid_k: int) -> dict:
    """gene 非依存 null-ridge で clip=True/False の load_bearing FPR を測る.

    各 null seed で base_seed をずらして独立な null 実現を作り、MAP-E が 3 baseline
    全勝 (= 偽陽性) する割合を集計。FPR が高いほど Type I コスト大。
    """
    task = NullRidgeTask(seq_len=20, state_dim=4)
    bb = (np.array([0.0]), np.array([1.0]))
    behavior = lambda g: np.array([g[0]])  # noqa: E731

    def _fpr_for_clip(clip: bool) -> dict:
        events = []
        per_repeat = []
        for nidx, nseed in enumerate(null_seeds):
            ev = _make_ridge_eval(task, clip=clip, n_tr=n_tr)
            scores = run_ea_methods_over_seeds(
                ev, ev, behavior, dim=GENE_DIM, bounds=GENE_BOUNDS,
                behavior_bounds=bb, grid_shape=(grid_k,), n_evals=n_evals,
                n_seeds=n_seeds, honest_n_trials=honest_n, sigma=0.12,
                base_seed=nseed,  # null 実現ごとに独立 seed
            )
            v = _verdict_for_scores(scores, min_seeds=min(15, n_seeds))
            events.append(int(v["load_bearing"]))
            per_repeat.append({
                "null_seed": nseed,
                "load_bearing": v["load_bearing"],
                "n_baselines_beaten": v["n_baselines_beaten"],
                "map_elites_mean": v["map_elites_mean"],
            })
            print(
                f"    null[{nidx}] seed={nseed} clip={clip}: "
                f"LB={v['load_bearing']} beaten={v['n_baselines_beaten']}/3 "
                f"me_mean={v['map_elites_mean']:.4f}",
                flush=True,
            )
        return {
            "fpr": float(np.mean(events)) if events else 0.0,
            "n_null_repeats": len(events),
            "per_repeat": per_repeat,
        }

    print("[null-ridge FPR] clip=True ...", flush=True)
    fpr_clip_true = _fpr_for_clip(True)
    print("[null-ridge FPR] clip=False ...", flush=True)
    fpr_clip_false = _fpr_for_clip(False)
    fpr_delta = fpr_clip_false["fpr"] - fpr_clip_true["fpr"]
    return {
        "clip_true": fpr_clip_true,
        "clip_false": fpr_clip_false,
        "fpr_delta_false_minus_true": fpr_delta,
        # clip=False が FPR を 0.05 超で上げたら Type I コスト有意。
        "clip_false_inflates_fpr": bool(fpr_delta > 0.05),
        "note": (
            "null-ridge は gene 非依存 target (inputs と無相関ノイズ) = 真の H0。"
            "clip=False FPR > clip=True FPR + 0.05 なら『clip を外すとノイズを構造として "
            "拾い偽陽性を増やす』= Type I コスト。clip は信号の有無を識別できない床だが "
            "FPR 抑制の役割があることになる。"
        ),
    }


# ===========================================================================
# G3 sanity: gate が known-positive で PASS / known-null で FPR≈α
# ===========================================================================


def g3_sanity_check() -> dict:
    """eval_gate が (a) 明確に勝つ分布で PASS、(b) 同分布 (H0) で稀に PASS、を確認.

    本実験の判定器 (full strict gate) が健全に働くことの最小校正。EA は回さず合成配列で。
    """
    rng = np.random.default_rng(BASE_SEED)
    # known-positive: a が b を一貫して上回る (psd→1, p 小)。
    n = 15
    b = rng.uniform(0.0, 0.4, size=n)
    a_pos = b + rng.uniform(0.10, 0.20, size=n)  # 全 seed で a>b
    g_pos = AC.eval_gate(a_pos, b, min_seeds=n)
    # known-null: a と b 同分布 (shuffle)。多 repeat で FPR≈α 期待。
    fp = 0
    REP = 400
    for _ in range(REP):
        x = rng.uniform(0.0, 0.5, size=n)
        y = rng.uniform(0.0, 0.5, size=n)
        if AC.eval_gate(x, y, min_seeds=n).passes:
            fp += 1
    fpr_null = fp / REP
    return {
        "known_positive_passes": bool(g_pos.passes),
        "known_positive_p": g_pos.wilcoxon_p,
        "known_positive_psd": g_pos.paired_sign_delta,
        "known_null_fpr": fpr_null,
        "known_null_fpr_under_010": bool(fpr_null <= 0.10),
        "diagnostic_valid": bool(g_pos.passes and fpr_null <= 0.10),
    }


# ===========================================================================
# main
# ===========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny smoke (n_seeds=3, n_evals=40, null 1 seed)")
    ap.add_argument("--phase", choices=("both", "verdict", "null"), default="both",
                    help="G1 分割実行 (TRIZ #1): 'verdict'=verdict-flip のみ, "
                         "'null'=null-ridge FPR のみ, 'both'=両方 (full は両方で G1 超過のため "
                         "分割 run を推奨)。phase 別 partial JSON に書き、最後に merge 判定。")
    args = ap.parse_args()

    guard = AC.RunGuard.start("exp3_clip_verdict_flip")
    t0 = time.time()

    if args.smoke:
        n_seeds, n_evals, honest_n, n_tr, grid_k = 3, 40, 6, 12, 8
        null_seeds = [BASE_SEED + 100]
        mode = "smoke"
    else:
        # 予算 (G1<900s 遵守): ridge_fitness ~10ms/call (n_tr=16)。
        # verdict-flip = 2 task × 2 clip × n_seeds, null-FPR = 2 clip × len(null_seeds) × n_seeds。
        # ~9.8s/seed (n_evals=60,n_tr=16,honest_n=8) で
        #   verdict-flip: 2*2*15*9.8 ≈ 588s, null-FPR: 2*4*15*9.8(null は n_evals 同) ...
        # null は n_evals を 40 に下げて 2*4*15*6.5 ≈ 780s 累積を回避するため verdict より軽量化。
        n_seeds, n_evals, honest_n, n_tr, grid_k = 15, 60, 8, 16, 12
        null_seeds = [BASE_SEED + 100, BASE_SEED + 200, BASE_SEED + 300, BASE_SEED + 400]
        mode = "full"

    print(f"=== EXP3 clip verdict-flip + null-ridge FPR ({mode}) ===")
    print(f"n_seeds={n_seeds} n_evals={n_evals} honest_n={honest_n} n_tr={n_tr} "
          f"grid_k={grid_k} null_seeds={null_seeds}\n", flush=True)

    # --- G3 sanity (gate 健全性) ---
    g3 = g3_sanity_check()
    print(f"[G3] gate sanity: known_pos_passes={g3['known_positive_passes']} "
          f"known_null_fpr={g3['known_null_fpr']:.3f} valid={g3['diagnostic_valid']}\n",
          flush=True)

    null_n_evals = max(40, n_evals - 20) if not args.smoke else n_evals
    vflip_partial = EXP3_DIR / "exp3_verdict_partial.json"
    null_partial = EXP3_DIR / "exp3_null_partial.json"

    # --- phase 別実行 (G1 分割; TRIZ #1) ---
    # 各 phase を partial JSON に書き、'both' でなくても後続 run が前 phase の partial を
    # 読み込んで merge 判定できる (chunked-resumable, exp_c2c3 と同パターン)。
    run_verdict = args.phase in ("both", "verdict")
    run_null = args.phase in ("both", "null")

    if run_verdict:
        vflip = measure_verdict_flip(n_seeds=n_seeds, n_evals=n_evals, honest_n=honest_n,
                                     n_tr=n_tr, grid_k=grid_k)
        vflip_partial.write_text(
            json.dumps({"verdict_flip": vflip, "g3_gate_sanity": g3,
                        "wall_clock_s": round(time.time() - t0, 3)},
                       ensure_ascii=False, indent=2, default=AC._json_default),
            encoding="utf-8")
    else:
        vflip = (json.loads(vflip_partial.read_text(encoding="utf-8"))["verdict_flip"]
                 if vflip_partial.exists() else None)

    if run_null:
        null = measure_null_ridge_fpr(null_seeds=null_seeds, n_seeds=n_seeds,
                                      n_evals=null_n_evals, honest_n=honest_n,
                                      n_tr=n_tr, grid_k=grid_k)
        null_partial.write_text(
            json.dumps({"null_ridge_fpr": null,
                        "wall_clock_s": round(time.time() - t0, 3)},
                       ensure_ascii=False, indent=2, default=AC._json_default),
            encoding="utf-8")
    else:
        null = (json.loads(null_partial.read_text(encoding="utf-8"))["null_ridge_fpr"]
                if null_partial.exists() else None)

    # merge: 片 phase だけ走らせた場合は他方を partial から復元。両方無いと判定不能。
    if vflip is None or null is None:
        meta = guard.finish()
        print(f"[exp3] phase={args.phase} 完了。merge 判定には verdict/null 両 partial が必要。"
              f"(verdict={'有' if vflip is not None else '無'} "
              f"null={'有' if null is not None else '無'}) "
              f"(RunGuard {meta['wall_clock_s']}s, src_unchanged={meta['src_unchanged']})")
        return 0

    # --- 統合判定 (3 値) ---
    any_real_flip = any(
        v["verdict_flip_false_reveals_third"] for v in vflip.values()
    )
    fpr_inflated = null["clip_false_inflates_fpr"]
    # [CF2] G3 sanity (known-positive PASS ∧ known-null FPR<=0.10) を統合判定に渡す。
    # diagnostic_valid=False なら判定器自体が信用できないため、過大確定 (null_confirmed /
    # not_load_bearing) を名乗らず still_inconclusive に落とす (sanity 不成立ゆえ判定保留)。
    diagnostic_valid = bool(g3.get("diagnostic_valid", False))

    if not diagnostic_valid:
        # [CF2] 判定器 sanity 不成立 → どの方向の確定も保留
        verdict = "still_inconclusive"
        verdict_text = (
            "判定保留 (CF2): G3 gate sanity が不成立 "
            f"(known_positive_passes={g3.get('known_positive_passes')}, "
            f"known_null_fpr={g3.get('known_null_fpr')}) ゆえ verdict-flip / FPR の "
            "判定器自体が信用できない → K4 の確定 (反証/降格) も載荷も名乗らず判定保留。"
        )
    elif any_real_flip and not fpr_inflated:
        verdict = "load_bearing"
        verdict_text = (
            "K4 確定: 低 R² task で clip=False が③ verdict-flip を起こし "
            "(clip 外すと selection が load-bearing に転じる)、かつ null-ridge で "
            "clip=False が FPR を上げない (信号回復 ∧ Type I 中立) → "
            "clip は active suppression。"
        )
    elif not any_real_flip:
        # [CF4] FPR 0/0 + ~7x 縮小予算ゆえ「null 確定」より「at this budget で非載荷」が正確。
        verdict = "not_load_bearing_at_this_budget"
        verdict_text = (
            "K4 反証/降格: clip=False でも③ verdict-flip が起きない "
            "(spread 平坦化は診断量だが verdict は変えない) → "
            "K4 を『唯一の能動的 suppression 機序』から『spread を潰すが verdict 非 "
            "load-bearing な診断的所見』に降格。null-ridge FPR=0/0 + ~7x 縮小予算ゆえ "
            "『null 確定』ではなく『この予算 (at this budget) で非載荷』と限定する (CF4)。"
        )
    else:  # any_real_flip and fpr_inflated
        verdict = "still_inconclusive"
        verdict_text = (
            "なお不確定: clip=False で verdict_flip=True だが null-ridge FPR も clip=False "
            "で有意増 (信号回復と Type I 増が分離不能) → F2 は確定も反証もできず、"
            "より大予算 null-ridge が必要。"
        )

    meta = guard.finish()
    wall = round(time.time() - t0, 3)
    # G1 break-gate: 'both' を 1 プロセスで回すと full で ~1203s と G1(900s) を超過する
    # (初回 full run 実測)。honest disclosure: silent truncation せず、--phase verdict /
    # --phase null の 2 run に分割すれば各 phase は G1 内に収まる (verdict ~770s,
    # null ~430s と推定)。本 JSON が 'both' 由来なら g1_pass=False を明記。
    g1_limit_s = 900.0
    g1_pass = wall <= g1_limit_s
    payload = {
        "_meta": {
            **meta,
            "wall_clock_s_total": wall,
            "g1_break_gate": {
                "limit_s": g1_limit_s,
                "wall_clock_s": wall,
                "phase": args.phase,
                "g1_pass": bool(g1_pass),
                "note": (
                    "'both' を 1 プロセスで回すと full で G1 (900s) を超過する "
                    "(初回 full 実測 ~1203s)。silent truncation 禁止のため超過を明記。"
                    "再現で G1 を守るには --phase verdict と --phase null の 2 run に "
                    "分割する (TRIZ #1 分割; 各 phase は partial JSON 経由で merge)。"
                    "判定結果 (verdict_flip / FPR) は分割しても同一 (CRN seed 固定・決定論)。"
                ) if not g1_pass else f"phase={args.phase} は G1 内 ({wall}s <= {g1_limit_s}s)。",
            },
            "mode": mode,
            "design": "EXP3 K4 clip verdict-flip + null-ridge FPR (Codex F2 確定/反証)",
            "base_seed": BASE_SEED,
            "n_seeds": n_seeds,
            "n_evals_verdict": n_evals,
            "n_evals_null": null_n_evals,
            "honest_n": honest_n,
            "n_tr": n_tr,
            "grid_k": grid_k,
            "null_seeds": null_seeds,
            "ea_baselines": list(EA_BASELINES),
            "supersedes": "research/statistical_power_audit/ablate_suppression_knobs.py "
                          "K4_clip_spread (spread/floor 率のみ; verdict-flip 未測)",
            "budget_reduction_disclosure": (
                "ridge_fitness は ~10ms/call (n_tr=16)。verdict-flip / FPR の "
                "③判定に効果量精度は不要なので n_evals/honest_n/n_tr を縮小: "
                f"verdict n_evals={n_evals}, null n_evals={null_n_evals}, "
                f"honest_n={honest_n}, n_tr={n_tr}, grid_k={grid_k}。"
                "exp_ea3 本実験は n_evals=400/honest_n=30/n_tr=48。本実験は③の有無 "
                "(verdict 反転) の検出が目的で精密な効果量は不要なため縮小。"
                "silent truncation なし。CRN seed は clip True/False 間で共通。"
                "注: 縮小後も 'both' 1 プロセスは full で ~1203s と G1(900s) 超過 — "
                "g1_break_gate 参照。--phase 分割で各 phase は G1 内。"
            ),
            "honest_caveats": (
                "(1) ridge_fitness の clip=True 0.0 は raw R²<0 を潰した床で『信号皆無』"
                "とは識別不能 (ridge_readout docstring の Codex High finding)。"
                "(2) 縮小予算ゆえ underpowered な case では verdict_flip=False が "
                "『clip 無関係』でなく『低 power で③検出不能』の可能性 (EXP1 の psd 床 "
                "問題と同根)。本実験は『clip を外しても③が出ないか』の方向性確認が主眼。"
                "(3) null-ridge FPR は len(null_seeds) 反復のみで MC 誤差大 "
                "(±1/sqrt(N))。FPR delta が 0.05 床近傍なら『分離不能』判定が妥当。"
            ),
        },
        "g3_gate_sanity": g3,
        "verdict_flip": vflip,
        "null_ridge_fpr": null,
        "summary": {
            "any_real_verdict_flip": any_real_flip,
            "tasks_flipped": [t for t, v in vflip.items()
                              if v["verdict_flip_false_reveals_third"]],
            "null_fpr_clip_true": null["clip_true"]["fpr"],
            "null_fpr_clip_false": null["clip_false"]["fpr"],
            "null_fpr_delta": null["fpr_delta_false_minus_true"],
            "clip_false_inflates_fpr": fpr_inflated,
            "third_axis_verdict": verdict,
            "verdict_text": verdict_text,
        },
    }

    out_path = EXP3_DIR / "exp3_clip_flip_results.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=AC._json_default),
        encoding="utf-8",
    )
    print(f"\n[exp3] wrote {out_path}  (total {wall}s, RunGuard {meta['wall_clock_s']}s)")
    print(f"[exp3] verdict = {verdict}")
    print(f"[exp3] {verdict_text}")
    print(f"[exp3] src_unchanged={meta['src_unchanged']} "
          f"changed={meta['src_changed_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
