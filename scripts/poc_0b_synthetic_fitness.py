# SPDX-License-Identifier: Apache-2.0
"""PoC 0b — 合成 sequence fitness (copy / addition) の falsifiable 命題検証.

falsifiable 命題:
    decay/mix/gate_str の 3 パラメータ gene + 固定線形 readout で、
    copy task / addition task それぞれに対し
    (a) fitness は決定論的に計算可能で範囲 [0, 1] に収まり、
    (b) random gene 集団で fitness の分散が非自明 (各 task で gene が差別化される),
    (c) copy と addition で「最適 gene」が異なる (task 依存性),
    (d) baseline-fit gene の fitness は random gene 中央値より高い。

破綻ゲート (G1-G7):
- [G1] fitness が NaN/Inf にならず [0, 1] に収まる
- [G2] 決定論性: 同 gene/seed で 2 回 evaluate しても一致
- [G3] non-degenerate: random gene 20 個体で fitness が定数でない (variance > 1e-4)
- [G4] task 依存: copy と addition で random pop の fitness 分布が異なる
- [G5] gene sensitivity: 各 gene の摂動で fitness が変化 (parameter 効いている)
- [G6] baseline calibration: random gene 中央値の MSE が baseline として収まる
- [G7] reasonable best: 数百個体 random search で fitness > 0.3 の gene が存在

使い方::

    py -3.11 scripts/poc_0b_synthetic_fitness.py

依存: numpy のみ. llive 非依存.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass


_PROJ_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.fitness import (  # noqa: E402
    AdditionTask,
    CopyTask,
    calibrate_baseline_robust,
    evaluate_gene,
    make_fixed_readout,
)
from llcore.state_update import StateUpdateGene  # noqa: E402


def _make_calibrated_tasks() -> tuple[CopyTask, CopyTask, AdditionTask]:
    """seed sweep で baseline_mse を calibrate した task triple を作る.

    v2 (Codex Q1 指摘 fix): copy delay=0 と delay=4 (memory horizon) の 2 種類で
    "memory-capable" の主張を強める。
    v2 (Codex Q5 指摘 fix): calibrate_baseline_robust で 3 seed sweep median 採用。
    """
    state_dim = 8
    readout_copy = make_fixed_readout(state_dim, out_dim=state_dim, seed=1001)
    readout_add = make_fixed_readout(state_dim, out_dim=1, seed=1002)

    base_copy0 = CopyTask(state_dim=state_dim, out_dim=state_dim, delay=0)
    base_copy4 = CopyTask(name="copy_delay4", state_dim=state_dim, out_dim=state_dim, delay=4)
    base_add = AdditionTask(state_dim=state_dim, out_dim=1)

    mse_c0, _, _ = calibrate_baseline_robust(base_copy0, readout_copy)
    mse_c4, _, _ = calibrate_baseline_robust(base_copy4, readout_copy)
    mse_a, _, _ = calibrate_baseline_robust(base_add, readout_add)

    return (
        replace(base_copy0, baseline_mse=mse_c0),
        replace(base_copy4, baseline_mse=mse_c4),
        replace(base_add, baseline_mse=mse_a),
    )


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate_g1_fitness_range(rng: np.random.Generator) -> tuple[bool, str]:
    """[G1] fitness が NaN/Inf 出ず [0, 1] に収まる."""
    copy_task, _copy4_task, add_task = _make_calibrated_tasks()
    readout_copy = make_fixed_readout(8, 8, seed=1001)
    readout_add = make_fixed_readout(8, 1, seed=1002)
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    f_copy = evaluate_gene(gene, copy_task, readout_copy, rng)
    f_add = evaluate_gene(gene, add_task, readout_add, rng)
    ok = (
        np.isfinite(f_copy) and np.isfinite(f_add)
        and 0.0 <= f_copy <= 1.0 and 0.0 <= f_add <= 1.0
    )
    return ok, f"f_copy={f_copy:.3f}, f_add={f_add:.3f} ∈ [0, 1]"


def gate_g2_determinism() -> tuple[bool, str]:
    """[G2] 同 gene/seed で 2 回 evaluate して fitness が一致."""
    copy_task, _c4, _a = _make_calibrated_tasks()
    readout = make_fixed_readout(8, 8, seed=1001)
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    f1 = evaluate_gene(gene, copy_task, readout, np.random.default_rng(42))
    f2 = evaluate_gene(gene, copy_task, readout, np.random.default_rng(42))
    ok = f1 == f2
    return ok, f"f1={f1:.6f}, f2={f2:.6f}, diff={abs(f1 - f2):.2e}"


def gate_g3_non_degenerate(rng: np.random.Generator) -> tuple[bool, str]:
    """[G3] random gene 20 個体で fitness 分散が非自明."""
    copy_task, _copy4_task, add_task = _make_calibrated_tasks()
    readout_copy = make_fixed_readout(8, 8, seed=1001)
    readout_add = make_fixed_readout(8, 1, seed=1002)
    f_copies: list[float] = []
    f_adds: list[float] = []
    for _ in range(20):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        f_copies.append(evaluate_gene(gene, copy_task, readout_copy, rng))
        f_adds.append(evaluate_gene(gene, add_task, readout_add, rng))
    var_copy = float(np.var(f_copies))
    var_add = float(np.var(f_adds))
    ok = var_copy > 1e-4 and var_add > 1e-4
    return ok, f"var(copy)={var_copy:.4e}, var(add)={var_add:.4e} > 1e-4"


def gate_g4_task_dependency(rng: np.random.Generator) -> tuple[bool, str]:
    """[G4] copy と addition で gene の fitness rank が異なる (task 依存性).

    v2 (Codex 2026-05-29 fix): OR logic → rank_corr 必須 + mean_diff 補助。
    task dependency の本質は ranking/selection pressure の差なので rank が主判定。
    """
    copy_task, _copy4_task, add_task = _make_calibrated_tasks()
    readout_copy = make_fixed_readout(8, 8, seed=1001)
    readout_add = make_fixed_readout(8, 1, seed=1002)
    n = 30
    f_copies: list[float] = []
    f_adds: list[float] = []
    for _ in range(n):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        f_copies.append(evaluate_gene(gene, copy_task, readout_copy, rng))
        f_adds.append(evaluate_gene(gene, add_task, readout_add, rng))
    f_copies_arr = np.array(f_copies)
    f_adds_arr = np.array(f_adds)
    rank_copy = np.argsort(np.argsort(f_copies_arr))
    rank_add = np.argsort(np.argsort(f_adds_arr))
    rank_corr = float(np.corrcoef(rank_copy, rank_add)[0, 1])
    mean_diff = abs(float(f_copies_arr.mean()) - float(f_adds_arr.mean()))
    # 主判定: rank_corr < 0.7 必須 (task で gene 順位が違う = 真の task 依存性)
    ok = abs(rank_corr) < 0.7
    return ok, f"rank_corr={rank_corr:.3f} (< 0.7 主判定) | mean_diff={mean_diff:.4f} (補助)"


def gate_g5_gene_sensitivity(rng: np.random.Generator) -> tuple[bool, str]:
    """[G5] 各 gene の摂動で fitness が変化."""
    copy_task, _c4, _a = _make_calibrated_tasks()
    readout = make_fixed_readout(8, 8, seed=1001)
    base = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    f_base = evaluate_gene(base, copy_task, readout, rng)
    perturbed = [
        ("decay+0.2", StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.5)),
        ("mix+0.3", StateUpdateGene(decay=0.7, mix=0.8, gate_str=0.5)),
        ("gate+0.5", StateUpdateGene(decay=0.7, mix=0.5, gate_str=1.0)),
    ]
    msgs: list[str] = []
    all_ok = True
    for name, gene in perturbed:
        f = evaluate_gene(gene, copy_task, readout, rng)
        diff = abs(f - f_base)
        ok = diff > 1e-3
        msgs.append(f"{name}:diff={diff:.4f}")
        all_ok = all_ok and ok
    return all_ok, " | ".join(msgs)


def gate_g6_baseline_calibration() -> tuple[bool, str]:
    """[G6] baseline_mse の seed 頑健性 (Codex 2026-05-29 fix).

    seed sweep (3 seed) で median が同 order に収まることを要求 (max/min < 3)。
    """
    state_dim = 8
    readout_copy = make_fixed_readout(state_dim, state_dim, seed=1001)
    readout_add = make_fixed_readout(state_dim, 1, seed=1002)
    base_copy = CopyTask(state_dim=state_dim, out_dim=state_dim, delay=0)
    base_add = AdditionTask(state_dim=state_dim, out_dim=1)
    med_c, min_c, max_c = calibrate_baseline_robust(base_copy, readout_copy)
    med_a, min_a, max_a = calibrate_baseline_robust(base_add, readout_add)
    # seed 頑健性: max/min < 3 (同 order)
    ratio_c = max_c / max(min_c, 1e-12)
    ratio_a = max_a / max(min_a, 1e-12)
    ok = (
        np.isfinite(med_c) and med_c > 1e-4
        and np.isfinite(med_a) and med_a > 1e-4
        and ratio_c < 3.0 and ratio_a < 3.0
    )
    return ok, f"copy median={med_c:.3e} (range {min_c:.3e}..{max_c:.3e}, ratio={ratio_c:.2f}) | add median={med_a:.3e} (ratio={ratio_a:.2f})"


def gate_g7_reasonable_best(rng: np.random.Generator, n_search: int = 200) -> tuple[bool, str]:
    """[G7] random search で fitness > 0.3 が 3 task (copy0/copy4/add) 全てで存在.

    v2 (Codex 2026-05-29 fix): copy delay=0 に加え copy delay=4 (memory horizon) も検査。
    """
    copy0_task, copy4_task, add_task = _make_calibrated_tasks()
    readout_copy = make_fixed_readout(8, 8, seed=1001)
    readout_add = make_fixed_readout(8, 1, seed=1002)
    bests = {"copy0": 0.0, "copy4": 0.0, "add": 0.0}
    for _ in range(n_search):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        bests["copy0"] = max(bests["copy0"], evaluate_gene(gene, copy0_task, readout_copy, rng, n_trials=3))
        bests["copy4"] = max(bests["copy4"], evaluate_gene(gene, copy4_task, readout_copy, rng, n_trials=3))
        bests["add"] = max(bests["add"], evaluate_gene(gene, add_task, readout_add, rng, n_trials=3))
    ok = all(v > 0.3 for v in bests.values())
    return ok, f"best copy0={bests['copy0']:.3f}, copy4={bests['copy4']:.3f}, add={bests['add']:.3f} all > 0.3"


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 0b — Synthetic sequence fitness (copy / addition) falsifiable verification")
    print("=" * 72)
    rng = np.random.default_rng(20260529)

    gates: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("G1: fitness range [0, 1] + finite", lambda: gate_g1_fitness_range(rng)),
        ("G2: determinism (seed=42)", gate_g2_determinism),
        ("G3: non-degenerate random pop N=20", lambda: gate_g3_non_degenerate(rng)),
        ("G4: task dependency (copy vs add)", lambda: gate_g4_task_dependency(rng)),
        ("G5: gene sensitivity", lambda: gate_g5_gene_sensitivity(rng)),
        ("G6: baseline calibration finite", gate_g6_baseline_calibration),
        ("G7: reasonable best (200 search)", lambda: gate_g7_reasonable_best(rng, 200)),
    ]

    all_pass = True
    for name, fn in gates:
        ok, detail = fn()
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {name}")
        print(f"         {detail}")
        all_pass = all_pass and ok

    print("-" * 72)
    if all_pass:
        print("PoC 0b verdict: PASS — fitness が機能、task 依存性 + gene sensitivity を満たす.")
        print("                 次段 PoC 0c (自前 minimal GA) に進めます.")
        return 0
    print("PoC 0b verdict: FAIL — fitness 設計または task 設計を見直してから次へ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
