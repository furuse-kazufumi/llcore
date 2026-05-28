# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 0b synthetic fitness (copy / addition).

破綻ゲート G1-G7 を pytest 経由でも独立検証。実行::

    pytest tests/unit/test_poc_0b_fitness.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from llcore.fitness import (
    AdditionTask,
    CopyTask,
    FixedReadout,
    calibrate_baseline,
    evaluate_gene,
    make_fixed_readout,
)
from llcore.state_update import StateUpdateGene


@pytest.fixture
def calibrated_copy() -> CopyTask:
    readout = make_fixed_readout(8, 8, seed=1001)
    base = CopyTask(state_dim=8, out_dim=8, delay=0)
    mse = calibrate_baseline(base, readout)
    return replace(base, baseline_mse=mse)


@pytest.fixture
def calibrated_add() -> AdditionTask:
    readout = make_fixed_readout(8, 1, seed=1002)
    base = AdditionTask(state_dim=8, out_dim=1)
    mse = calibrate_baseline(base, readout)
    return replace(base, baseline_mse=mse)


@pytest.fixture
def readout_copy() -> FixedReadout:
    return make_fixed_readout(8, 8, seed=1001)


@pytest.fixture
def readout_add() -> FixedReadout:
    return make_fixed_readout(8, 1, seed=1002)


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------


def test_fixed_readout_shape() -> None:
    """readout(state) の shape は (out_dim,)."""
    r = make_fixed_readout(8, 4, seed=42)
    state = np.zeros(8)
    out = r(state)
    assert out.shape == (4,)
    assert np.all(out == 0)


def test_fixed_readout_deterministic() -> None:
    """同 seed で同 matrix."""
    r1 = make_fixed_readout(8, 4, seed=42)
    r2 = make_fixed_readout(8, 4, seed=42)
    assert np.array_equal(r1.matrix, r2.matrix)


# ---------------------------------------------------------------------------
# G1 fitness range + finite
# ---------------------------------------------------------------------------


def test_g1_fitness_in_range(
    calibrated_copy: CopyTask,
    calibrated_add: AdditionTask,
    readout_copy: FixedReadout,
    readout_add: FixedReadout,
) -> None:
    """[G1] fitness が NaN/Inf 出ず [0, 1] に収まる."""
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    rng = np.random.default_rng(20260529)
    f_c = evaluate_gene(gene, calibrated_copy, readout_copy, rng)
    f_a = evaluate_gene(gene, calibrated_add, readout_add, rng)
    assert np.isfinite(f_c)
    assert np.isfinite(f_a)
    assert 0.0 <= f_c <= 1.0
    assert 0.0 <= f_a <= 1.0


# ---------------------------------------------------------------------------
# G2 determinism
# ---------------------------------------------------------------------------


def test_g2_determinism(
    calibrated_copy: CopyTask, readout_copy: FixedReadout
) -> None:
    """[G2] 同 gene/seed で 2 回 evaluate して完全一致."""
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    f1 = evaluate_gene(gene, calibrated_copy, readout_copy, np.random.default_rng(42))
    f2 = evaluate_gene(gene, calibrated_copy, readout_copy, np.random.default_rng(42))
    assert f1 == f2


# ---------------------------------------------------------------------------
# G3 non-degenerate
# ---------------------------------------------------------------------------


def test_g3_non_degenerate_population(
    calibrated_copy: CopyTask,
    calibrated_add: AdditionTask,
    readout_copy: FixedReadout,
    readout_add: FixedReadout,
) -> None:
    """[G3] random gene 20 個体で fitness 分散 > 1e-4 (両 task)."""
    rng = np.random.default_rng(20260529)
    f_copies: list[float] = []
    f_adds: list[float] = []
    for _ in range(20):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        f_copies.append(evaluate_gene(gene, calibrated_copy, readout_copy, rng))
        f_adds.append(evaluate_gene(gene, calibrated_add, readout_add, rng))
    assert float(np.var(f_copies)) > 1e-4
    assert float(np.var(f_adds)) > 1e-4


# ---------------------------------------------------------------------------
# G4 task dependency
# ---------------------------------------------------------------------------


def test_g4_task_dependency_rank_corr_required(
    calibrated_copy: CopyTask,
    calibrated_add: AdditionTask,
    readout_copy: FixedReadout,
    readout_add: FixedReadout,
) -> None:
    """[G4] v2: rank_corr 必須 (主判定) — task dependency の本質は ranking 差.

    v1 では OR logic だったが、Codex 2026-05-29 指摘で「rank_corr 必須 + mean_diff 補助」に
    変更。本 test は主判定 rank_corr < 0.7 のみ assert。
    """
    rng = np.random.default_rng(20260529)
    f_copies: list[float] = []
    f_adds: list[float] = []
    for _ in range(30):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        f_copies.append(evaluate_gene(gene, calibrated_copy, readout_copy, rng))
        f_adds.append(evaluate_gene(gene, calibrated_add, readout_add, rng))
    rank_corr = float(np.corrcoef(
        np.argsort(np.argsort(f_copies)),
        np.argsort(np.argsort(f_adds)),
    )[0, 1])
    # 主判定: rank_corr < 0.7 (task で gene 順位が異なる = 真の task 依存性)
    assert abs(rank_corr) < 0.7


# ---------------------------------------------------------------------------
# G5 gene sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,decay,mix,gate_str",
    [
        ("decay+0.2", 0.9, 0.5, 0.5),
        ("mix+0.3", 0.7, 0.8, 0.5),
        ("gate+0.5", 0.7, 0.5, 1.0),
    ],
)
def test_g5_gene_sensitivity(
    calibrated_copy: CopyTask,
    readout_copy: FixedReadout,
    name: str,
    decay: float,
    mix: float,
    gate_str: float,
) -> None:
    """[G5] 各 gene の摂動で fitness が変化."""
    base = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    rng_b = np.random.default_rng(20260529)
    rng_p = np.random.default_rng(20260529)
    f_base = evaluate_gene(base, calibrated_copy, readout_copy, rng_b)
    f_pert = evaluate_gene(
        StateUpdateGene(decay=decay, mix=mix, gate_str=gate_str),
        calibrated_copy,
        readout_copy,
        rng_p,
    )
    assert abs(f_pert - f_base) > 1e-3, f"{name} dead"


# ---------------------------------------------------------------------------
# G6 baseline calibration
# ---------------------------------------------------------------------------


def test_g6_baseline_finite(calibrated_copy: CopyTask, calibrated_add: AdditionTask) -> None:
    """[G6] baseline_mse が有限・正値."""
    assert np.isfinite(calibrated_copy.baseline_mse)
    assert calibrated_copy.baseline_mse > 1e-4
    assert calibrated_copy.baseline_mse < 1e4
    assert np.isfinite(calibrated_add.baseline_mse)
    assert calibrated_add.baseline_mse > 1e-4
    assert calibrated_add.baseline_mse < 1e4


def test_g6_baseline_seed_robust(
    readout_copy: FixedReadout, readout_add: FixedReadout
) -> None:
    """[G6] v2: seed sweep で baseline_mse が同 order (max/min < 3) — 頑健性 gate.

    Codex 2026-05-29 指摘で追加 (v1 baseline_mse の単純 range check は弱い)。
    """
    from llcore.fitness import calibrate_baseline_robust

    base_copy = CopyTask(state_dim=8, out_dim=8, delay=0)
    base_add = AdditionTask(state_dim=8, out_dim=1)
    med_c, min_c, max_c = calibrate_baseline_robust(base_copy, readout_copy)
    med_a, min_a, max_a = calibrate_baseline_robust(base_add, readout_add)
    ratio_c = max_c / max(min_c, 1e-12)
    ratio_a = max_a / max(min_a, 1e-12)
    assert ratio_c < 3.0, f"copy seed sweep ratio too high: {ratio_c}"
    assert ratio_a < 3.0, f"add seed sweep ratio too high: {ratio_a}"


def test_g7_reasonable_best_3tasks(
    readout_copy: FixedReadout, readout_add: FixedReadout
) -> None:
    """[G7] v2: random search で 3 task (copy0/copy4/add) 全部で fitness > 0.3.

    Codex 2026-05-29 指摘で copy delay=4 追加 → "memory-capable" を honest 主張。
    """
    from dataclasses import replace as dc_replace

    from llcore.fitness import calibrate_baseline

    copy0 = dc_replace(
        CopyTask(state_dim=8, out_dim=8, delay=0),
        baseline_mse=calibrate_baseline(CopyTask(state_dim=8, out_dim=8, delay=0), readout_copy),
    )
    copy4 = dc_replace(
        CopyTask(state_dim=8, out_dim=8, delay=4),
        baseline_mse=calibrate_baseline(CopyTask(state_dim=8, out_dim=8, delay=4), readout_copy),
    )
    add = dc_replace(
        AdditionTask(state_dim=8, out_dim=1),
        baseline_mse=calibrate_baseline(AdditionTask(state_dim=8, out_dim=1), readout_add),
    )
    rng = np.random.default_rng(20260529)
    bests = {"copy0": 0.0, "copy4": 0.0, "add": 0.0}
    for _ in range(150):  # CI 軽量化 (200 → 150)
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        bests["copy0"] = max(bests["copy0"], evaluate_gene(gene, copy0, readout_copy, rng, n_trials=3))
        bests["copy4"] = max(bests["copy4"], evaluate_gene(gene, copy4, readout_copy, rng, n_trials=3))
        bests["add"] = max(bests["add"], evaluate_gene(gene, add, readout_add, rng, n_trials=3))
    assert bests["copy0"] > 0.3, f"copy delay=0 best too low: {bests}"
    assert bests["copy4"] > 0.3, f"copy delay=4 (memory horizon) best too low: {bests}"
    assert bests["add"] > 0.3, f"add best too low: {bests}"


def test_raw_error_consistency_addition(
    calibrated_add: AdditionTask, readout_add: FixedReadout
) -> None:
    """v2 fix: AdditionTask の raw_error と score が同じ MSE 定義を共有.

    Codex 2026-05-29 blocker fix の test 追随 — abs(pred) 経由の error 定義一貫性。
    """
    from llcore.state_update import run_sequence

    rng = np.random.default_rng(42)
    inputs, target = calibrated_add.generate(rng)
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    trajectory = run_sequence(inputs, gene)
    raw_err = calibrated_add.raw_error(trajectory, target, readout_add)
    # score = 1 - clip(raw_err / baseline, 0, 1) と一致するか
    expected_score = float(np.clip(1.0 - raw_err / max(calibrated_add.baseline_mse, 1e-9), 0.0, 1.0))
    actual_score = calibrated_add.score(trajectory, target, readout_add)
    assert abs(actual_score - expected_score) < 1e-12, (
        f"score / raw_error inconsistency: actual={actual_score}, expected={expected_score}"
    )


# ---------------------------------------------------------------------------
# Task generate API
# ---------------------------------------------------------------------------


def test_copy_task_shapes(calibrated_copy: CopyTask) -> None:
    """copy task の generate shape."""
    inputs, target = calibrated_copy.generate(np.random.default_rng(0))
    assert inputs.shape == (calibrated_copy.seq_len, calibrated_copy.state_dim)
    assert target.shape == (calibrated_copy.state_dim,)


def test_addition_task_shapes(calibrated_add: AdditionTask) -> None:
    """addition task の generate shape."""
    inputs, target = calibrated_add.generate(np.random.default_rng(0))
    assert inputs.shape == (calibrated_add.seq_len, calibrated_add.state_dim)
    assert target.shape == (1,)
    assert target[0] >= 0  # L2 norm is non-negative


def test_calibrate_baseline_seed_stable(readout_copy: FixedReadout) -> None:
    """calibrate_baseline は seed 固定で再現性あり."""
    base = CopyTask(state_dim=8, out_dim=8)
    mse1 = calibrate_baseline(base, readout_copy, seed=42)
    mse2 = calibrate_baseline(base, readout_copy, seed=42)
    assert mse1 == mse2
