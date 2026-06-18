# SPDX-License-Identifier: Apache-2.0
"""Tests for ``llcore.fitness.memory_objective`` (branch A: メモリ効率 fitness, PoC P1)."""
from __future__ import annotations

import numpy as np
import pytest

from llcore.fitness import (
    CopyTask,
    MemoryEfficiencyObjective,
    evaluate_gene,
    make_fixed_readout,
    state_boundedness_footprint,
)
from llcore.state_update import StateUpdateGene
from llcore.verifier.invariants import _lipschitz_upper_bound


def _contractive() -> StateUpdateGene:
    # L = max(0.5, 0.5 + 0.5*0) = 0.5 < 1 (収縮的・有界状態)
    return StateUpdateGene(decay=0.5, mix=0.0, gate_str=0.0)


def _divergent() -> StateUpdateGene:
    # L = max(0.0, 0.0 + 1.0*2.0) = 2.0 >= l_cap (発散しうる)
    return StateUpdateGene(decay=0.0, mix=0.0, gate_str=2.0)


def test_state_boundedness_footprint_in_unit_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        gene = StateUpdateGene(
            decay=float(rng.uniform(-0.5, 1.5)),
            mix=float(rng.uniform(-2, 2)),
            gate_str=float(rng.uniform(-3, 3)),
        )
        f = state_boundedness_footprint(gene)
        assert 0.0 <= f <= 1.0


def test_footprint_matches_verifier_closed_form() -> None:
    # drift guard: proxy は verifier の閉形式上界 / l_cap の正規化と一致する。
    rng = np.random.default_rng(7)
    for _ in range(30):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0, 1)),
            mix=float(rng.uniform(-1, 1)),
            gate_str=float(rng.uniform(-2, 2)),
        )
        g = gene.clipped()
        expected = min(_lipschitz_upper_bound(g.decay, g.gate_str) / 2.0, 1.0)
        assert state_boundedness_footprint(gene, l_cap=2.0) == pytest.approx(expected)


def test_contractive_gene_has_lower_footprint_than_divergent() -> None:
    assert state_boundedness_footprint(_contractive()) < state_boundedness_footprint(_divergent())
    assert state_boundedness_footprint(_contractive()) == pytest.approx(0.25)  # 0.5/2.0
    assert state_boundedness_footprint(_divergent()) == pytest.approx(1.0)  # clip(2.0/2.0)


def test_memory_objective_fitness_in_unit_range() -> None:
    task = CopyTask()
    obj = MemoryEfficiencyObjective(base_task=task, w_acc=0.7, w_mem=0.3, n_trials=3)
    readout = make_fixed_readout(task.state_dim, task.out_dim, seed=1)
    rng = np.random.default_rng(123)
    for _ in range(20):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0, 1)),
            mix=float(rng.uniform(-1, 1)),
            gate_str=float(rng.uniform(-2, 2)),
        )
        f = obj.fitness(gene, readout, np.random.default_rng(99))
        assert 0.0 <= f <= 1.0


def test_w_mem_zero_reduces_to_retention_baseline() -> None:
    # w_mem=0 ⇒ fitness は純 retention(evaluate_gene)に縮退する(floor 包含)。
    task = CopyTask()
    obj = MemoryEfficiencyObjective(base_task=task, w_acc=1.0, w_mem=0.0, n_trials=4)
    readout = make_fixed_readout(task.state_dim, task.out_dim, seed=2)
    gene = _contractive()
    got = obj.fitness(gene, readout, np.random.default_rng(2026))
    want = evaluate_gene(gene, task, readout, np.random.default_rng(2026), n_trials=4)
    assert got == pytest.approx(want)


def test_memory_objective_rejects_bad_weights() -> None:
    task = CopyTask()
    with pytest.raises(ValueError):
        MemoryEfficiencyObjective(base_task=task, w_acc=0.0, w_mem=0.0)
    with pytest.raises(ValueError):
        MemoryEfficiencyObjective(base_task=task, w_acc=-1.0, w_mem=0.5)


def test_higher_w_mem_rewards_contractive_over_divergent_at_equal_retention() -> None:
    # メモリ項を重くすると、retention が同程度でも収縮的 gene が有利になる(gate の意味の芽)。
    task = CopyTask()
    obj_mem = MemoryEfficiencyObjective(base_task=task, w_acc=0.3, w_mem=0.7, n_trials=3)
    readout = make_fixed_readout(task.state_dim, task.out_dim, seed=3)
    # footprint 項だけを比較するため、retention を共通化した合成は難しいので
    # footprint 寄与の符号(収縮的 > 発散的)が fitness にも表れることを確認する。
    f_contractive = obj_mem.fitness(_contractive(), readout, np.random.default_rng(5))
    f_divergent = obj_mem.fitness(_divergent(), readout, np.random.default_rng(5))
    # 収縮的は footprint 0.25(報酬 0.75)、発散的は footprint 1.0(報酬 0.0)。
    # retention 差が footprint 差(0.7*0.75=0.525)を逆転させない限り収縮的が上。
    assert f_contractive > f_divergent


# --- PoC script (scripts/poc_branch_a_memory_fitness.py) ---


def _load_poc() -> object:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "poc_branch_a_memory_fitness.py"
    spec = importlib.util.spec_from_file_location("poc_branch_a", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_cell(mod: object, gate_mode: str) -> dict[str, object]:
    task = CopyTask()
    readout = make_fixed_readout(task.state_dim, task.out_dim, seed=2026)
    obj = MemoryEfficiencyObjective(base_task=task, w_acc=0.7, w_mem=0.3, n_trials=3)
    return mod.run_cell(  # type: ignore[attr-defined]
        gate_mode, obj, readout,
        n_seeds=2, pop_size=6, n_generations=2, resample_cap=10, base_seed=0,
    )


def test_run_cell_control_has_no_gate_stats_and_unit_metrics() -> None:
    mod = _load_poc()
    cell = _tiny_cell(mod, "none")
    assert "gate_stats" not in cell
    for k in ("safe_rate", "mean_footprint", "mean_retention"):
        assert 0.0 <= cell[k] <= 1.0  # type: ignore[operator]


def test_run_cell_gated_records_gate_stats_without_spinning() -> None:
    mod = _load_poc()
    cell = _tiny_cell(mod, "contraction")  # G3/G6: fail-closed, bounded resamples
    gs = cell["gate_stats"]
    assert gs["n_resamples"] <= gs["max_possible_resamples"]  # type: ignore[index,call-overload]
    assert gs["n_children_generated"] > 0  # type: ignore[index,operator]


def test_run_cell_is_deterministic_g1() -> None:
    mod = _load_poc()
    assert _tiny_cell(mod, "none") == _tiny_cell(mod, "none")  # same seed -> identical


def test_poc_main_smoke_writes_verdict(tmp_path: object) -> None:
    mod = _load_poc()
    out = tmp_path / "poc.json"  # type: ignore[attr-defined]
    import json as _json

    rc = mod.main([  # type: ignore[attr-defined]
        "--n-seeds", "2", "--pop-size", "6", "--n-generations", "2",
        "--falsify-seeds", "3", "--honest-trials", "3", "--json", str(out),
    ])
    assert rc == 0
    payload = _json.loads(out.read_text(encoding="utf-8"))
    v = payload["verdict"]
    # G5 は記録のみ(bool 値が入るだけで合否に使わない)。
    assert "memory_passes" in v["G5_falsification_recorded"]
    assert isinstance(v["functional_min"], bool)
