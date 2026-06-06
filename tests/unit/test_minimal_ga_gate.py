# SPDX-License-Identifier: Apache-2.0
"""Unit tests for T1 Phase 1 (a) — 証明ゲートの evolve() 本配線 (additive).

検証範囲:
- (i)   gate off (``gate_mode="none"``) = 旧挙動 byte-identical (seed 固定)
        + ``gate_stats is None`` (後方互換)。
- (ii)  gate on (``contraction``) で非契約 gene が reject され、admit された全 (非 elite)
        子が L<1 certified であること + 集計が立つこと。
- (iii) fallback 経路 (resample_cap=0 で reject → known-safe fallback) が発火し
        ``fallback_count > 0``、fallback gene は契約済みであること。
- (iv)  src 配線 evolve() と research wrapper gated_evolve() の挙動一致 (全モード)。
- (v)   誤用 (codec + gated mode) を fail-loud で弾く。

実行::

    pytest tests/unit/test_minimal_ga_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from llcore.evolution import GateStats, evolve
from llcore.kernel.protocol import GeneCodec  # noqa: F401  (型/誤用テスト用)
from llcore.state_update import StateUpdateGene
from llcore.verifier import verify_gene_safe, verify_lipschitz_contraction


# ---------------------------------------------------------------------------
# fitness helpers (deterministic with seed, exercises rng draw order)
# ---------------------------------------------------------------------------


def _ff(gene: StateUpdateGene, rng: np.random.Generator) -> float:
    """rng を引いて確率的 fitness draw 順も再現する小さな決定論 fitness."""
    x = rng.uniform(-1, 1, size=(4, 3))
    g = gene.clipped()
    return float(np.tanh(g.decay + g.mix - g.gate_str + x.sum()))


_KW = dict(
    pop_size=10,
    n_generations=8,
    tournament_k=3,
    mutation_sigma=0.15,
    crossover_rate=0.5,
    elitism=1,
)


# ---------------------------------------------------------------------------
# (i) gate off = byte-identical to old behavior + gate_stats None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 2020])
def test_gate_none_byte_identical_to_default(seed: int) -> None:
    """[i] gate_mode 既定 ('none') と明示 'none' が完全一致 = ゲート無効で旧挙動保存."""
    r_default = evolve(_ff, rng=np.random.default_rng(seed), **_KW)
    r_none = evolve(_ff, rng=np.random.default_rng(seed), gate_mode="none", **_KW)
    assert r_default.best_fitness_curve == r_none.best_fitness_curve
    assert r_default.diversity_curve == r_none.diversity_curve
    assert r_default.final_best.fitness == r_none.final_best.fitness


def test_gate_none_stats_is_none() -> None:
    """[i] gate off では gate_stats=None (additive フィールド既定; 後方互換)."""
    r = evolve(_ff, rng=np.random.default_rng(1000), **_KW)
    assert r.gate_stats is None
    r2 = evolve(_ff, rng=np.random.default_rng(1000), gate_mode="none", **_KW)
    assert r2.gate_stats is None


def test_default_evolutionresult_construction_backward_compatible() -> None:
    """[i] EvolutionResult を新フィールド無しの keyword 構築できる (既存呼び出し互換)."""
    from llcore.evolution import EvolutionResult, Individual, Population

    pop = Population(individuals=(Individual(StateUpdateGene(0.5, 0.0, 0.0), 1.0),))
    res = EvolutionResult(
        generations=(pop,),
        best_fitness_curve=(1.0,),
        diversity_curve=(0.0,),
    )
    assert res.gate_stats is None  # default
    assert res.final_best.fitness == 1.0


# ---------------------------------------------------------------------------
# (ii) gate on (contraction) rejects non-contract genes; admitted are certified
# ---------------------------------------------------------------------------


def test_contraction_gate_admits_only_certified() -> None:
    """[ii] contraction gate on で admit された全 (非 elite) 子が L<1 certified."""
    r = evolve(_ff, rng=np.random.default_rng(1000), gate_mode="contraction", **_KW)
    assert isinstance(r.gate_stats, GateStats)
    assert r.gate_stats.gate_mode == "contraction"
    # 最終世代の elite を除く子 (elitism=1) はすべて契約済みであるべき。
    # elite は前世代から持ち越すため、起点世代の non-certified 個体が残ることはあるが、
    # gate が child admission を支配するため世代を経た全集団は契約済みへ収束する。
    final = r.generations[-1]
    sorted_inds = sorted(final.individuals, key=lambda ind: -ind.fitness)
    children = sorted_inds[1:]  # elitism=1 → 先頭 1 個が elite
    for ind in children:
        assert verify_lipschitz_contraction(ind.gene).contraction is True, (
            f"admitted child not certified: {ind.gene}"
        )


def test_contraction_gate_rejects_noncontract_children() -> None:
    """[ii] 非契約初期集団 + contraction gate で reject が実際に発生する."""
    # gate_str=2 は L>=1 (非契約)。mutation で多くの子が gate に弾かれる。
    init = [StateUpdateGene(decay=0.05, mix=0.0, gate_str=2.0) for _ in range(10)]
    r = evolve(
        _ff,
        rng=np.random.default_rng(7),
        pop_size=10,
        n_generations=5,
        initial_pop=init,
        gate_mode="contraction",
        mutation_sigma=0.05,
        resample_cap=50,
    )
    assert r.gate_stats is not None
    assert r.gate_stats.n_rejections > 0, "expected the gate to reject some children"


def test_state_norm_gate_runs_and_reports() -> None:
    """[ii] state_norm gate も配線されて集計を返す (admit された子は state_norm ok)."""
    r = evolve(_ff, rng=np.random.default_rng(1000), gate_mode="state_norm", **_KW)
    assert isinstance(r.gate_stats, GateStats)
    assert r.gate_stats.gate_mode == "state_norm"
    final = r.generations[-1]
    sorted_inds = sorted(final.individuals, key=lambda ind: -ind.fitness)
    for ind in sorted_inds[1:]:
        assert verify_gene_safe(ind.gene).ok


# ---------------------------------------------------------------------------
# (iii) fallback path
# ---------------------------------------------------------------------------


def test_fallback_path_on_resample_cap_zero() -> None:
    """[iii] resample_cap=0 + 非契約初期集団 → reject 即 fallback (fallback_count>0)."""
    init = [StateUpdateGene(decay=0.05, mix=0.0, gate_str=2.0) for _ in range(10)]
    r = evolve(
        lambda g, rng: float(rng.uniform()),
        rng=np.random.default_rng(7),
        pop_size=10,
        n_generations=5,
        initial_pop=init,
        gate_mode="contraction",
        resample_cap=0,
        mutation_sigma=0.05,
    )
    assert r.gate_stats is not None
    assert r.gate_stats.fallback_count > 0, "expected the resample cap to trigger fallback"
    assert r.gate_stats.n_resamples == 0, "resample_cap=0 means zero resample attempts"


def test_fallback_gene_is_contraction_certified() -> None:
    """[iii] known-safe fallback gene (decay=0.5, mix=0, gate_str=0) は契約済み."""
    from llcore.evolution.minimal_ga import _FALLBACK_GENE

    lr = verify_lipschitz_contraction(_FALLBACK_GENE)
    assert lr.contraction is True
    assert verify_gene_safe(_FALLBACK_GENE).ok
    # all population members under contraction gate must be certified
    # (children gated; fallback certified; elites originate from gated populations).
    assert _FALLBACK_GENE.decay == 0.5
    assert _FALLBACK_GENE.mix == 0.0
    assert _FALLBACK_GENE.gate_str == 0.0


# ---------------------------------------------------------------------------
# (iv) src evolve() == research gated_evolve() (両実装の挙動一致)
# ---------------------------------------------------------------------------


def _load_research_gated_evolve():
    """research/verified_evolution/gated_evolve.py を import (sys.path をテスト内で限定)."""
    research_dir = Path(__file__).resolve().parents[2] / "research" / "verified_evolution"
    if str(research_dir) not in sys.path:
        sys.path.insert(0, str(research_dir))
    from gated_evolve import gated_evolve  # noqa: E402

    return gated_evolve


@pytest.mark.parametrize("mode", ["none", "state_norm", "contraction"])
@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_src_evolve_matches_research_gated_evolve(mode: str, seed: int) -> None:
    """[iv] src 配線 evolve(gate_mode=...) と research gated_evolve(...) が byte-identical.

    旧 research 実装 (検証コア) と新 src 配線が RNG ストリーム・結果・集計まで一致する
    ことで、移植が忠実 (re-implementation バグなし) であることを証明する。
    """
    gated_evolve = _load_research_gated_evolve()
    rs = evolve(_ff, rng=np.random.default_rng(seed), gate_mode=mode, **_KW)
    rg = gated_evolve(_ff, gate_mode=mode, rng=np.random.default_rng(seed), **_KW)

    assert rs.best_fitness_curve == rg.result.best_fitness_curve
    assert rs.diversity_curve == rg.result.diversity_curve

    if mode == "none":
        assert rs.gate_stats is None
    else:
        assert rs.gate_stats is not None
        assert rs.gate_stats.n_rejections == rg.n_rejections
        assert rs.gate_stats.n_resamples == rg.n_resamples
        assert rs.gate_stats.fallback_count == rg.fallback_count
        assert rs.gate_stats.n_children_generated == rg.n_children_generated


# ---------------------------------------------------------------------------
# (v) misuse guards
# ---------------------------------------------------------------------------


def test_gated_with_codec_raises() -> None:
    """[v] codec (coupled gene) + gated mode は未対応 → ValueError で fail-loud."""
    from llcore.kernel.rwkv_codec import RWKVCodec  # type: ignore

    with pytest.raises(ValueError, match="scalar StateUpdateGene"):
        evolve(
            _ff,
            rng=np.random.default_rng(1000),
            gate_mode="contraction",
            codec=RWKVCodec(),
            **_KW,
        )


def test_unknown_gate_mode_raises() -> None:
    """[v] 未知 gate_mode は _gate_admits が fail-loud (ValueError)."""
    with pytest.raises(ValueError, match="unknown gate_mode"):
        evolve(_ff, rng=np.random.default_rng(1000), gate_mode="bogus", **_KW)
