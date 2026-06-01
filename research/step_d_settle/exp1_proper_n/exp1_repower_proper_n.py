# SPDX-License-Identifier: Apache-2.0
"""EXP1 (step_d_settle) — proper-n 再検定: ③ が有意化するか null-at-power か.

背景: 統計検出力監査 (STATISTICAL_POWER_VERDICT.md) で C-gen4b(MAP-E vs random) /
flip_flop(vs random / vs RR) の「③不要」判定が n=15 では underpowered (power 0.31 /
0.27 / 0.15) と判明。本実験は n を n80 域まで上げて再検定し、3 値で着地する:
  ③load_bearing確定 / null_confirmed_at_power / still_inconclusive (psd 床律速)。

TRIZ #1 分割 — 単一 n80 fresh-run は CPU 非現実的 (実測 C-gen4b ~56s/seed,
flip_flop @n_evals=2000 ~266s/seed) なので 2 層に分離:

A 層 [bootstrap 再確認, 安価, 主証拠]:
  既存 results JSON から **実 per-seed paired delta** を復元 (_audit_common.
  load_real_negative_deltas)。各 case を n∈{15,30,64,82,128,169,255} で B=5000 bootstrap、
  CRN として B 個の resample index を seed=20260601 で固定共有 (case 間 paired)。full gate
  (片側 Wilcoxon p<0.05 ∧ |psd|≥0.147 ∧ diff>0) を vectorized 適用し power 曲線・n80 を算出。
  観測単発判定は src exact scipy.wilcoxon (_paired_p) を使用。

B 層 [fresh 真再走, G1 適合 chunk, 確証/反証]:
  C-gen4b は ea_lab.run_ea_methods_over_seeds を n_seeds 拡張 (SeedSequence で
  s>=15 は非 alias の fresh replicate)、flip_flop は selection_lab.run_methods_over_seeds
  を非衝突 base_seed の追加 chunk で回し per-seed score を partial JSON に追記
  (chunked-resumable, exp_c2c3 と同パターン)。到達 n は honest disclosure で明記。

honest caveat (§9 psd床): bootstrap で C-gen4b median psd は n を上げても ~0.20 で動かず、
P(|psd|≥0.147) は ~0.80 に頭打ち。= psd≈0.20 床は p<0.05 飽和後に binding に転じ大 n でも
power が 0.80 を超えない構造的天井 → 「中効果ゆえ n では限界」を honest negative とする。
flip_flop vs RR は psd=0.067<0.147 で床未満 = どんな n でも gate 不可 (null 寄り) と明示。

src 非改変。既存 lab は read-only import 再利用のみ。書込は本ディレクトリのみ。git 非実行。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# --- repo root / 隔離 dir ---
_THIS = Path(__file__).resolve()
EXP1_DIR = _THIS.parent  # research/step_d_settle/exp1_proper_n/
STEP_D_DIR = EXP1_DIR.parent  # research/step_d_settle/
REPO_ROOT = STEP_D_DIR.parents[1]  # D:/projects/llcore
SRC_DIR = REPO_ROOT / "src"
AUDIT_DIR = REPO_ROOT / "research" / "statistical_power_audit"
EA_DIR = REPO_ROOT / "research" / "ea_multitask"
STEP_C_DIR = REPO_ROOT / "research" / "step_c_memory_tasks"
STEP4_DIR = REPO_ROOT / "research" / "step4_selection"
EA_CANDIDATES_DIR = EA_DIR / "candidates"  # variable_delay_recall.py はここ (exp_ea3 L29)

# read-only import path (src/既存 lab 改造禁止)
for _p in (str(SRC_DIR), str(AUDIT_DIR), str(EA_DIR), str(EA_CANDIDATES_DIR),
           str(STEP_C_DIR), str(STEP4_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_ensure_utf8_stdout()

# --- read-only import: A 層 power machinery (改造禁止) ---
import _audit_common as AC  # noqa: E402
from _audit_common import (  # noqa: E402
    eval_gate,
    load_real_negative_deltas,
    textbook_cliff_delta,
    wilcoxon_p_greater_batch,
)
from llcore.evolution.honest_eval import _paired_p, _paired_sign_delta  # noqa: E402

ALPHA = 0.05
MIN_EFFECT = 0.147
# n80 域までの sweep (DESIGN: 既知 n80 = C-gen4b 255 / ff_vs_random 82)
N_SWEEP = [15, 30, 64, 82, 128, 169, 255]
CRN_SEED = 20260601  # case 間共有の resample index seed (paired bootstrap)
EXP1_TARGETS = (
    "C-gen4b_MAPE_vs_random",       # diff>0 psd=0.200 underpowered 候補
    "flip_flop_MAPE_vs_random",     # diff>0 psd=0.333 underpowered 候補
    "flip_flop_MAPE_vs_rr_hillclimb",  # psd=0.067 床未満 (null 寄り)
)


# ===========================================================================
# G4: src 不変監査 + 書込先 assert (本 dir 限定)
# ===========================================================================
def _src_tree_fingerprint() -> dict[str, tuple[int, float]]:
    fp: dict[str, tuple[int, float]] = {}
    for f in sorted((SRC_DIR / "llcore").rglob("*.py")):
        st = f.stat()
        fp[str(f.relative_to(REPO_ROOT))] = (st.st_size, st.st_mtime)
    return fp


def _assert_write_path(path: Path) -> None:
    rp = Path(path).resolve()
    if EXP1_DIR not in rp.parents and rp != EXP1_DIR:
        raise AssertionError(f"G4 違反: 書込先が exp1 dir 外 {rp}")


def _json_default(o: object):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _dump_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    _assert_write_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")
    return path


class _Tee:
    """stdout を log file にも複製 (silent truncation 禁止の証跡保全)."""

    def __init__(self, log_path: Path):
        _assert_write_path(log_path)
        self._f = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._f.write(s)
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class RunGuard:
    def __init__(self, name: str):
        self.name = name
        self.t0 = time.time()
        self.src_fp_start = _src_tree_fingerprint()

    def finish(self) -> dict:
        elapsed = time.time() - self.t0
        fp_end = _src_tree_fingerprint()
        changed = [k for k in fp_end if fp_end.get(k) != self.src_fp_start.get(k)]
        return {
            "script": self.name,
            "wall_clock_s": round(elapsed, 3),
            "src_unchanged": fp_end == self.src_fp_start,
            "src_changed_files": changed,
            "exit_ok": True,
        }


# ===========================================================================
# A 層: bootstrap re-power (CRN paired)
# ===========================================================================
def _batch_gate_powers(samples: np.ndarray, *, alpha: float = ALPHA,
                       min_effect: float = MIN_EFFECT) -> dict[str, float]:
    """(B, n) paired-delta 標本群に full gate を vectorized 適用し条件別 power を返す.

    repower_real_negatives._batch_gate_powers と同一ロジック (read-only 再現)。
    片側 Wilcoxon は AC.wilcoxon_p_greater_batch (正規近似, G3 校正済) を使う。
    """
    diffs = samples.mean(axis=1)
    pos = (samples > 0).sum(axis=1)
    neg = (samples < 0).sum(axis=1)
    n = samples.shape[1]
    psd = (pos - neg) / n
    p = wilcoxon_p_greater_batch(samples)
    c_diff = diffs > 0.0
    c_p = p < alpha
    c_eff = np.abs(psd) >= min_effect
    return {
        "full": float(np.mean(c_diff & c_p & c_eff)),
        "p_only": float(np.mean(c_p)),
        "effect_only": float(np.mean(c_eff)),
        "diff_only": float(np.mean(c_diff)),
        # psd 床到達確率 (P(|psd|>=min_effect)) = §9 床天井の直接観測
        "p_psd_floor": float(np.mean(np.abs(psd) >= min_effect)),
        "median_psd": float(np.median(psd)),
    }


def _crn_resample_idx(npop: int, n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """(B, n) の復元抽出 index。CRN: 同 (n, B, seed) で全 case 共有 → paired bootstrap."""
    return rng.integers(0, npop, size=(B, n))


def _calibrate_power_engine(B: int) -> dict:
    """G3: power 計算器の妥当性校正 (null / 大効果 / C-gen4b 模擬)。repower と同基準."""
    rng = np.random.default_rng(424242)

    def _boot(pop, n):
        idx = rng.integers(0, len(pop), size=(B, n))
        return _batch_gate_powers(pop[idx])["full"]

    def _param(mean, sd, n):
        return _batch_gate_powers(rng.normal(mean, sd, size=(B, n)))["full"]

    null_pop = rng.normal(0.0, 0.10, size=200)
    null_power = _boot(null_pop, 15)
    big_pop = np.abs(rng.normal(0.20, 0.05, size=200)) + 0.05
    big_power = _boot(big_pop, 20)
    cg_15 = _param(0.06255, 0.12, 15)
    cg_30 = _param(0.06255, 0.12, 30)
    valid = (null_power <= 0.06 and big_power >= 0.95
             and abs(cg_15 - 0.59) <= 0.08 and abs(cg_30 - 0.83) <= 0.08)
    return {
        "null_power": null_power, "big_effect_power": big_power,
        "cgen4b_sim_n15": cg_15, "cgen4b_sim_n30": cg_30,
        "checks": {
            "null_le_0.06": bool(null_power <= 0.06),
            "big_ge_0.95": bool(big_power >= 0.95),
            "cgen4b_n15_059pm008": bool(abs(cg_15 - 0.59) <= 0.08),
            "cgen4b_n30_083pm008": bool(abs(cg_30 - 0.83) <= 0.08),
        },
        "power_engine_valid": bool(valid),
    }


def _n80_from_curve(ns: list[int], powers: list[float]) -> float | None:
    for i in range(len(ns)):
        if powers[i] >= 0.80:
            if i == 0:
                return float(ns[0])
            p0, p1 = powers[i - 1], powers[i]
            n0, n1 = ns[i - 1], ns[i]
            if p1 == p0:
                return float(n1)
            frac = (0.80 - p0) / (p1 - p0)
            return float(n0 + frac * (n1 - n0))
    return None


def layer_a_bootstrap(B: int, n_sweep: list[int]) -> dict:
    """A 層: 実 per-seed delta から CRN paired bootstrap re-power."""
    negs = load_real_negative_deltas()
    out: dict[str, dict] = {}
    # CRN: 各 n ごとに 1 つの resample-index seed を全 case で共有 → paired
    n_seed_rngs = {n: np.random.default_rng([CRN_SEED, n]) for n in n_sweep}

    for name in EXP1_TARGETS:
        if name not in negs:
            print(f"  [A] WARN: {name} が results JSON に無い → skip")
            continue
        rec = negs[name]
        delta = np.asarray(rec["delta"], dtype=np.float64)
        a = np.asarray(rec["a"], dtype=np.float64)
        b = np.asarray(rec["b"], dtype=np.float64)
        npop = len(delta)
        mean = float(delta.mean())
        sd = float(np.std(delta, ddof=1))
        dz = float(mean / sd) if sd > 0 else (np.inf if mean > 0 else 0.0)

        # 観測単発判定は src exact (_paired_p)
        p_exact = _paired_p(a, b)
        psd_obs = _paired_sign_delta(delta)
        cd = textbook_cliff_delta(a, b)

        full_curve, ppsd_curve, median_psd_curve = [], [], []
        cond_p_curve, cond_eff_curve = [], []
        for n in n_sweep:
            idx = _crn_resample_idx(npop, n, B, n_seed_rngs[n])
            pw = _batch_gate_powers(delta[idx])
            full_curve.append(pw["full"])
            ppsd_curve.append(pw["p_psd_floor"])
            median_psd_curve.append(pw["median_psd"])
            cond_p_curve.append(pw["p_only"])
            cond_eff_curve.append(pw["effect_only"])

        n80 = _n80_from_curve(n_sweep, full_curve)
        floor_below = abs(psd_obs) < MIN_EFFECT  # psd が床未満なら gate は n に依らず不可
        ppsd_ceiling = max(ppsd_curve)  # P(|psd|>=床) の到達上限

        out[name] = {
            "note": rec.get("note", ""),
            "source": rec["source"],
            "n_observed": npop, "diff": mean, "sd": sd, "cohen_dz": dz,
            "paired_sign_delta_obs": psd_obs,
            "cliff_delta_textbook": cd,
            "wilcoxon_p_exact_obs": p_exact,
            "n_sweep": n_sweep,
            "power_curve_full": full_curve,
            "p_psd_floor_curve": ppsd_curve,
            "median_psd_curve": median_psd_curve,
            "cond_p_only_curve": cond_p_curve,
            "cond_effect_only_curve": cond_eff_curve,
            "n80_full_gate": n80,
            "psd_floor_below_threshold": bool(floor_below),
            "p_psd_floor_ceiling": ppsd_ceiling,
        }
        print(f"  [A] {name}: diff={mean:+.4f} psd_obs={psd_obs:+.3f} p_exact={p_exact:.4f} "
              f"dz={dz:+.2f}")
        print(f"       power(full) @ {n_sweep} = "
              f"{['%.3f' % x for x in full_curve]}  n80={n80}")
        print(f"       P(|psd|>={MIN_EFFECT}) @ n = "
              f"{['%.3f' % x for x in ppsd_curve]}  ceiling={ppsd_ceiling:.3f}")
        print(f"       median psd @ n = {['%.3f' % x for x in median_psd_curve]}  "
              f"floor_below_threshold={floor_below}")
    return out


# ===========================================================================
# B 層: fresh 真再走 (chunked-resumable, partial JSON 追記)
# ===========================================================================
def _load_partial(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _cgen4b_fresh_chunk(target_total_n: int, partial_path: Path, *,
                        chunk_seconds_budget: float) -> dict:
    """C-gen4b fresh: ea_lab.run_ea_methods_over_seeds を n_seeds 拡張で回し追記.

    SeedSequence([base_seed, method_idx, s]) ゆえ s=0..14 は元 exp_ea3 を厳密再現、
    s>=15 は非 alias の fresh replicate。1 seed ずつ回し chunk 予算内で追記。
    """
    from ea_lab import run_ea_methods_over_seeds
    from reservoir import LeakyDelayLineReservoir, gene_bounds, make_behavior, make_eval_once
    from task_mixture import TaskMixture, split_regimes
    from variable_delay_recall import VariableDelayRecallTask

    # exp_ea3_ablation.py と同一 config
    N_TAPS, IN_DIM, DAMP = 8, 2, 0.2
    TRAIN_D, TEST_D = (15, 30), (45, 60)
    N_EVALS, HONEST_N, SIGMA, GRID = 400, 30, 0.12, (6, 6)
    BASE_SEED = 20260530

    res = LeakyDelayLineReservoir(n_taps=N_TAPS, in_dim=IN_DIM)
    bounds = gene_bounds(res); behavior = make_behavior(res); dim = res.gene_dim
    all_D = tuple(sorted(set(TRAIN_D) | set(TEST_D)))
    regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=DAMP, in_dim=IN_DIM)
               for D in all_D]
    test_idx = [i for i, D in enumerate(all_D) if D in TEST_D]
    train_regimes, test_regimes = split_regimes(regimes, test_idx=test_idx)
    ev_tr = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    ev_te = make_eval_once(res, TaskMixture(test_regimes), n_train=48, n_eval=48)

    state = _load_partial(partial_path)
    block = state.get("C-gen4b", {"per_seed": {"map_elites": [], "random": [],
                                               "panmictic_ga": [], "map_elites_randselect": []},
                                  "n_done": 0, "base_seed": BASE_SEED,
                                  "config": {"task": "variable_delay_recall", "n_evals": N_EVALS,
                                             "honest_n": HONEST_N, "train_D": list(TRAIN_D),
                                             "test_D": list(TEST_D), "grid": list(GRID)}})
    n_done = block["n_done"]
    t_chunk0 = time.time()
    added = 0
    # 1 seed ずつ: run_ea_methods_over_seeds(n_seeds=s+1) は s 番目だけ新規計算しない
    # (全 s を毎回回す) ため、ここでは 1 seed 専用に SeedSequence を直接使い per-seed 評価する。
    # → run_ea_methods_over_seeds を n_seeds=1 で「s 固定」呼びはできないので、
    #   ea_lab の seed 設計と完全一致する薄い 1-seed runner を read-only primitives で組む。
    from ea_lab import map_elites_full, map_elites_randselect
    from llcore.evolution.honest_eval import honest_reevaluate
    from selection_lab import panmictic_ga as _panmictic

    def _evo_rng(method_idx: int, s: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([BASE_SEED, method_idx, s]))

    def _honest_test(gene, s: int) -> float:
        return honest_reevaluate(
            ev_te, gene, n_trials=HONEST_N,
            rng=np.random.default_rng(np.random.SeedSequence([BASE_SEED, 7, s])))

    init_batch = max(20, N_EVALS // 10)
    while n_done < target_total_n:
        if added > 0 and (time.time() - t_chunk0) > chunk_seconds_budget:
            print(f"  [B/C-gen4b] chunk 予算到達 ({chunk_seconds_budget}s) → 中断 (n_done={n_done})")
            break
        s = n_done
        r_me = map_elites_full(ev_tr, behavior, dim=dim, bounds=bounds,
                               behavior_bounds=(np.zeros(2), np.ones(2)), grid_shape=GRID,
                               n_evals=N_EVALS, init_batch=init_batch, sigma=SIGMA,
                               rng=_evo_rng(0, s))
        me = _honest_test(r_me.best_gene, s)
        r_rs = map_elites_randselect(ev_tr, behavior, dim=dim, bounds=bounds,
                                     behavior_bounds=(np.zeros(2), np.ones(2)), grid_shape=GRID,
                                     n_evals=N_EVALS, init_batch=init_batch, sigma=SIGMA,
                                     rng=_evo_rng(1, s))
        rs = _honest_test(r_rs.best_gene, s)
        r_ga = _panmictic(ev_tr, dim=dim, bounds=bounds, n_evals=N_EVALS, pop_size=20,
                          tournament_k=3, sigma=SIGMA, elitism=1, rng=_evo_rng(2, s))
        ga = _honest_test(r_ga.best_gene, s)
        rrng = _evo_rng(3, s)
        cands = [bounds[0] + (bounds[1] - bounds[0]) * rrng.random(dim) for _ in range(N_EVALS)]
        best = max(cands, key=lambda g: ev_tr(g, rrng))
        rnd = _honest_test(best, s)

        block["per_seed"]["map_elites"].append(float(me))
        block["per_seed"]["map_elites_randselect"].append(float(rs))
        block["per_seed"]["panmictic_ga"].append(float(ga))
        block["per_seed"]["random"].append(float(rnd))
        n_done += 1; added += 1
        block["n_done"] = n_done
        state["C-gen4b"] = block
        _dump_json(partial_path, state)  # 途中クラッシュ耐性
        print(f"  [B/C-gen4b] seed {s}: mape={me:.4f} rnd={rnd:.4f} "
              f"diff={me - rnd:+.4f} (n_done={n_done})")
    block["wall_clock_chunk_s"] = round(time.time() - t_chunk0, 1)
    return block


def _flipflop_fresh_chunk(target_extra_seeds: int, partial_path: Path, *,
                          chunk_seconds_budget: float) -> dict:
    """flip_flop fresh: selection_lab.run_methods_over_seeds を非衝突 base_seed で追記.

    run_methods_over_seeds は base_seed+s でシードするため元 (base=20260530, s∈0..14) と
    衝突しない離れた base (20271000 + chunk*100) を使い 1 seed ずつ fresh replicate を追加。
    DESIGN 準拠で n_evals=2000 を維持 (~266s/seed)。
    """
    from selection_lab import run_methods_over_seeds
    from memory_tasks import FlipFlopTask
    from reservoir import LeakyDelayLineReservoir, gene_bounds, make_behavior, make_eval_once

    N_EVALS, HONEST_N, SIGMA = 2000, 20, 0.15
    FRESH_BASE = 20271000  # 元 20260530 系列と非衝突 (差 >> n_seeds)

    task = FlipFlopTask(seq_len=30, in_dim=2)
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=task.in_dim)
    eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
    behavior = make_behavior(res); lo, hi = gene_bounds(res)

    state = _load_partial(partial_path)
    block = state.get("flip_flop", {"per_seed": {"map_elites": [], "random": [],
                                                 "rr_hillclimb": [], "panmictic_ga": []},
                                    "n_done": 0, "fresh_base_seed": FRESH_BASE,
                                    "config": {"task": "flip_flop", "seq_len": 30,
                                               "n_evals": N_EVALS, "honest_n": HONEST_N,
                                               "sigma": SIGMA}})
    n_done = block["n_done"]
    t_chunk0 = time.time()
    added = 0
    while added < target_extra_seeds:
        if added > 0 and (time.time() - t_chunk0) > chunk_seconds_budget:
            print(f"  [B/flip_flop] chunk 予算到達 ({chunk_seconds_budget}s) → 中断 "
                  f"(n_done={n_done})")
            break
        # 各 fresh replicate に固有 base_seed を割り当て (1 seed/call, 衝突防止)
        bs = FRESH_BASE + n_done
        sc = run_methods_over_seeds(
            eval_once, behavior, dim=res.gene_dim, bounds=(lo, hi),
            behavior_bounds=(np.array([0.0, 0.0]), np.array([1.0, 0.5])),
            grid_shape=(12, 12), n_evals=N_EVALS, n_seeds=1, honest_n_trials=HONEST_N,
            sigma=SIGMA, base_seed=bs)
        me = float(sc["map_elites"][0]); rnd = float(sc["random"][0])
        rr = float(sc["rr_hillclimb"][0]); ga = float(sc["panmictic_ga"][0])
        block["per_seed"]["map_elites"].append(me)
        block["per_seed"]["random"].append(rnd)
        block["per_seed"]["rr_hillclimb"].append(rr)
        block["per_seed"]["panmictic_ga"].append(ga)
        n_done += 1; added += 1
        block["n_done"] = n_done
        state["flip_flop"] = block
        _dump_json(partial_path, state)
        print(f"  [B/flip_flop] fresh seed {n_done - 1} (base={bs}): mape={me:.4f} "
              f"rnd={rnd:.4f} rr={rr:.4f} diff_rnd={me - rnd:+.4f}")
    block["wall_clock_chunk_s"] = round(time.time() - t_chunk0, 1)
    return block


def _evaluate_fresh_blocks(partial_path: Path) -> dict:
    """B 層 partial の到達済 per-seed を full gate で評価 (C-gen4b 元15 + fresh 合算)."""
    state = _load_partial(partial_path)
    out: dict[str, dict] = {}

    # --- C-gen4b: 元 exp_ea3 (s0..14) を fresh が拡張 ---
    if "C-gen4b" in state:
        ps = state["C-gen4b"]["per_seed"]
        me = np.asarray(ps["map_elites"], dtype=np.float64)
        rnd = np.asarray(ps["random"], dtype=np.float64)
        n = len(me)
        if n >= 2:
            g = eval_gate(me, rnd, "MAP-E", "random", min_seeds=15)
            out["C-gen4b_MAPE_vs_random"] = {
                "n_total": n, "mean_mape": float(me.mean()), "mean_random": float(rnd.mean()),
                "diff": g.diff, "wilcoxon_p_exact": g.wilcoxon_p,
                "paired_sign_delta": g.paired_sign_delta, "cohen_dz": g.cohen_dz,
                "gate_passes": g.passes, "cond_diff_pos": g.cond_diff_pos,
                "cond_p": g.cond_p, "cond_n": g.cond_n, "cond_effect": g.cond_effect,
            }

    # --- flip_flop: 元 exp_c2c3 (15) + fresh 追加。CRN は異なる (元は base20260530, fresh は別) ---
    if "flip_flop" in state:
        # 元 15 seed を exp_c2c3_results から取り、fresh を append
        cc_path = STEP_C_DIR / "exp_c2c3_results.json"
        orig = {"map_elites": [], "random": [], "rr_hillclimb": []}
        if cc_path.exists():
            cc = json.loads(cc_path.read_text(encoding="utf-8"))
            ff = cc["flip_flop"]["per_seed_scores"]
            for k in orig:
                orig[k] = list(ff[k])
        fresh = state["flip_flop"]["per_seed"]
        for bname in ("random", "rr_hillclimb"):
            me = np.asarray(orig["map_elites"] + fresh["map_elites"], dtype=np.float64)
            bb = np.asarray(orig[bname] + fresh[bname], dtype=np.float64)
            n = len(me)
            if n >= 2:
                g = eval_gate(me, bb, "MAP-E", bname, min_seeds=15)
                out[f"flip_flop_MAPE_vs_{bname}"] = {
                    "n_total": n, "n_orig": len(orig["map_elites"]),
                    "n_fresh": len(fresh["map_elites"]),
                    "mean_mape": float(me.mean()), "mean_baseline": float(bb.mean()),
                    "diff": g.diff, "wilcoxon_p_exact": g.wilcoxon_p,
                    "paired_sign_delta": g.paired_sign_delta, "cohen_dz": g.cohen_dz,
                    "gate_passes": g.passes,
                    "note": ("元15 (base20260530) + fresh は別 base のため厳密 CRN ではないが "
                             "matched-replicate ではなく独立追加 → effect size の安定性確認用。"),
                }
    return out


# ===========================================================================
# 3 値判定 (falsifiable_criteria)
# ===========================================================================
def _verdict(layer_a: dict, fresh_eval: dict) -> dict:
    """各 case を {load_bearing / null_confirmed_at_power / still_inconclusive} に着地."""
    verdicts: dict[str, dict] = {}

    def _case(name: str) -> dict:
        a = layer_a.get(name, {})
        fr = fresh_eval.get(name, {})
        psd_obs = a.get("paired_sign_delta_obs", 0.0)
        diff_obs = a.get("diff", 0.0)
        floor_below = a.get("psd_floor_below_threshold", False)
        ceiling = a.get("p_psd_floor_ceiling", 0.0)
        n80 = a.get("n80_full_gate")
        fr_pass = fr.get("gate_passes", False)
        fr_diff = fr.get("diff", None)
        fr_n = fr.get("n_total", 0)

        # (1) psd 床未満 → どんな n でも gate 不可 = null 寄り (= still_inconclusive の床律速亜種)
        if floor_below:
            v = "still_inconclusive"
            reason = (f"psd_obs={psd_obs:+.3f} は床 {MIN_EFFECT} 未満 → full gate は n に依らず不可 "
                      f"(効果が床以下 = null 寄り、しかし『効果無し』の積極的確証ではない)。")
            return {"verdict": v, "reason": reason, "psd_obs": psd_obs, "diff_obs": diff_obs,
                    "p_psd_floor_ceiling": ceiling, "n80": n80, "fresh_n": fr_n,
                    "fresh_gate_passes": fr_pass}

        # (2) fresh 真再走で diff<=0 へ収束 → null_confirmed_at_power
        if fr_diff is not None and fr_n >= 15 and fr_diff <= 0:
            v = "null_confirmed_at_power"
            reason = (f"fresh 真再走 n={fr_n} で diff={fr_diff:+.4f}<=0 へ収束 → ③不在を確証 "
                      f"(STATISTICAL_POWER_VERDICT §6 準拠)。")
            return {"verdict": v, "reason": reason, "psd_obs": psd_obs, "diff_obs": diff_obs,
                    "fresh_diff": fr_diff, "fresh_n": fr_n, "fresh_gate_passes": fr_pass,
                    "n80": n80}

        # (3) A 層 power>=0.80 到達 ∧ fresh full gate PASS → load_bearing
        a_reaches_80 = n80 is not None and ceiling >= 0.80
        if a_reaches_80 and fr_pass:
            v = "load_bearing"
            reason = (f"A 層 n80={n80} かつ P(|psd|床)上限={ceiling:.3f}>=0.80、"
                      f"fresh 真再走 n={fr_n} で full gate PASS。")
            return {"verdict": v, "reason": reason, "psd_obs": psd_obs, "n80": n80,
                    "p_psd_floor_ceiling": ceiling, "fresh_n": fr_n, "fresh_gate_passes": True}

        # (4) still_inconclusive — 律速がどちらか (psd 床天井 vs p) で文面を分ける
        v = "still_inconclusive"
        psd_floor_caps = ceiling < 0.80  # P(|psd|>=床) 自体が 0.80 未満 = 床が構造天井
        if psd_floor_caps:
            reason = (f"psd_obs={psd_obs:+.3f} (床 {MIN_EFFECT} 以上だが中効果帯)、"
                      f"P(|psd|>=床) 上限={ceiling:.3f}<0.80 → psd 床が構造的律速、"
                      f"full gate の効果量条件が n を上げても飽和して 0.80 power に届かない "
                      f"(中効果ゆえ psd 床が binding = honest negative)。")
        else:
            # psd 床は満たせるが n80 到達せず or fresh 未確認 = p (検定力) 律速
            reason = (f"psd_obs={psd_obs:+.3f} で効果量条件は n と共に満たせる "
                      f"(P(|psd|>=床) 上限={ceiling:.3f}>=0.80) が、A 層 full power n80={n80} "
                      f"(検定 p が律速)、fresh 真再走未到達/未 PASS のため確定保留。")
        reason += (f" fresh n={fr_n} で gate_pass={fr_pass}, diff={fr_diff}." if fr_diff is not None
                   else " (fresh 未到達/不足)")
        return {"verdict": v, "reason": reason, "psd_obs": psd_obs, "diff_obs": diff_obs,
                "p_psd_floor_ceiling": ceiling, "psd_floor_caps_power": bool(psd_floor_caps),
                "n80": n80, "fresh_n": fr_n, "fresh_gate_passes": fr_pass,
                "fresh_diff": fr_diff}

    for name in EXP1_TARGETS:
        verdicts[name] = _case(name)
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny: B=1000, B層スキップ")
    ap.add_argument("--layer", choices=["a", "b", "both"], default="both")
    ap.add_argument("--cgen4b-target-n", type=int, default=64,
                    help="C-gen4b fresh の到達目標 n (元 15 含む累積)")
    ap.add_argument("--flipflop-extra", type=int, default=0,
                    help="flip_flop fresh で追加する seed 数 (~266s/seed)")
    ap.add_argument("--chunk-seconds", type=float, default=820.0,
                    help="B 層 1 run の壁時計予算 (G1<900s)")
    args = ap.parse_args()

    log_path = EXP1_DIR / "exp1_repower_proper_n.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    guard = RunGuard("exp1_repower_proper_n")
    B = 1000 if args.smoke else 5000
    n_sweep = [15, 30, 64] if args.smoke else N_SWEEP

    try:
        print(f"=== EXP1 proper-n 再検定 (smoke={args.smoke} layer={args.layer} B={B}) ===")
        print(f"n_sweep={n_sweep} alpha={ALPHA} min_effect={MIN_EFFECT} CRN_seed={CRN_SEED}")

        # G3: power engine 校正
        calib = _calibrate_power_engine(B)
        print(f"[G3] power-engine: null={calib['null_power']:.3f} "
              f"big={calib['big_effect_power']:.3f} cg4b_n15={calib['cgen4b_sim_n15']:.3f} "
              f"cg4b_n30={calib['cgen4b_sim_n30']:.3f} VALID={calib['power_engine_valid']}")

        # A 層は常に算出 (cheap ~18s)。verdict は A 層の psd_obs/床天井に依存するため
        # --layer b 単独でも判定根拠を欠かさない (前 run の空 layer_a → psd_obs=0 バグ回避)。
        print("\n--- A 層: CRN paired bootstrap re-power ---")
        layer_a = layer_a_bootstrap(B, n_sweep)

        fresh_eval = {}
        partial_path = EXP1_DIR / "exp1_freshrun_partial.json"
        if args.layer in ("b", "both") and not args.smoke:
            print("\n--- B 層: fresh 真再走 (chunked-resumable) ---")
            cb = _cgen4b_fresh_chunk(args.cgen4b_target_n, partial_path,
                                     chunk_seconds_budget=args.chunk_seconds)
            print(f"  [B/C-gen4b] n_done={cb['n_done']} "
                  f"(chunk {cb.get('wall_clock_chunk_s', 0)}s)")
            if args.flipflop_extra > 0:
                fb = _flipflop_fresh_chunk(args.flipflop_extra, partial_path,
                                           chunk_seconds_budget=args.chunk_seconds)
                print(f"  [B/flip_flop] n_done={fb['n_done']} "
                      f"(chunk {fb.get('wall_clock_chunk_s', 0)}s)")
            fresh_eval = _evaluate_fresh_blocks(partial_path)
            print("\n--- fresh 評価 (full gate) ---")
            for name, fr in fresh_eval.items():
                print(f"  {name}: n={fr['n_total']} diff={fr['diff']:+.4f} "
                      f"p={fr['wilcoxon_p_exact']:.4f} psd={fr['paired_sign_delta']:+.3f} "
                      f"pass={fr['gate_passes']}")
        elif args.layer == "both":
            # smoke: 既存 partial があれば評価のみ
            fresh_eval = _evaluate_fresh_blocks(partial_path)

        verdicts = _verdict(layer_a, fresh_eval)
        print("\n=== 3 値判定 ===")
        for name, v in verdicts.items():
            print(f"  {name}: [{v['verdict']}]")
            print(f"     {v['reason']}")

        meta = guard.finish()
        g1_ok = meta["wall_clock_s"] < 900.0
        payload = {
            "_meta": {
                **meta, "experiment": "EXP1 proper-n 再検定 (C-gen4b / flip_flop)",
                "smoke": args.smoke, "layer": args.layer, "B": B, "n_sweep": n_sweep,
                "alpha": ALPHA, "min_effect": MIN_EFFECT, "crn_seed": CRN_SEED,
                "G1_wall_clock_lt_900": bool(g1_ok),
                "cgen4b_target_n": args.cgen4b_target_n,
                "flipflop_extra": args.flipflop_extra,
                "chunk_seconds_budget": args.chunk_seconds,
                "honest_disclosure": (
                    "psd≈0.20 床は p<0.05 飽和後に binding に転じ、大 n でも P(|psd|>=0.147) が "
                    "~0.80 で頭打ち = power の構造的天井。中効果ゆえ現実的 n では確定不能。"
                    "flip_flop vs RR は psd=0.067<0.147 で床未満 = 効果が床以下 (null 寄り)。"),
            },
            "power_engine_calibration_G3": calib,
            "layer_a_bootstrap": layer_a,
            "layer_b_fresh_eval": fresh_eval,
            "verdicts": verdicts,
        }
        out = _dump_json(EXP1_DIR / "exp1_repower_proper_n_results.json", payload)
        print(f"\n[exp1] wrote {out}  ({meta['wall_clock_s']}s, G1<900={g1_ok}, "
              f"src_unchanged={meta['src_unchanged']})")
        return 0
    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
