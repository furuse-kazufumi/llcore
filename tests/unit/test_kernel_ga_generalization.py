# SPDX-License-Identifier: Apache-2.0
"""S2: minimal_ga の GeneCodec 一般化 test.

検証命題:
- **後方互換 (byte-identity)**: ``evolve(codec=RWKVCodec())`` は ``evolve(codec=None)`` と
  best_fitness_curve / diversity_curve / final gene まで完全一致 (RNG ストリーム保存)。
- ``*_g`` operator は RWKV codec で RWKV 専用版と byte-identical。
- **gene 型非依存性**: 合成 4-dim toy gene + codec で GA が完走・決定論的・bounds 内、
  diversity が codec.to_array で計算される (gene が as_array を持たなくてよい)。
- ``Individual`` / ``Population`` の Generic 化が既存呼び出しを壊さない。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from llcore.evolution import (
    Individual,
    Population,
    crossover_uniform,
    crossover_uniform_g,
    evolve,
    initialize_random_population,
    initialize_random_population_g,
    uniform_mutate,
    uniform_mutate_g,
)
from llcore.kernel import RWKVCodec
from llcore.state_update import StateUpdateGene


# ---------------------------------------------------------------------------
# 決定論的 fitness (rng 非依存 → operator のみが RNG を消費)
# ---------------------------------------------------------------------------


def _rwkv_fitness(gene: StateUpdateGene, rng: np.random.Generator) -> float:
    """target (0.5, 0.0, 0.0) への近さ (決定論的, rng 不使用)."""
    a = gene.as_array()
    target = np.array([0.5, 0.0, 0.0])
    return float(-np.sum((a - target) ** 2))


# ---------------------------------------------------------------------------
# (1) 後方互換: codec=RWKVCodec() ≡ codec=None (byte-identity)
# ---------------------------------------------------------------------------


def test_evolve_rwkv_codec_equals_legacy_path() -> None:
    """evolve(codec=RWKVCodec()) と evolve(codec=None) が完全一致 (RNG ストリーム保存)."""
    kw = dict(pop_size=12, n_generations=15, elitism=1, mutation_sigma=0.2)
    r_legacy = evolve(_rwkv_fitness, rng=np.random.default_rng(7777), codec=None, **kw)
    r_codec = evolve(
        _rwkv_fitness, rng=np.random.default_rng(7777), codec=RWKVCodec(), **kw
    )
    assert r_legacy.best_fitness_curve == r_codec.best_fitness_curve
    assert r_legacy.diversity_curve == r_codec.diversity_curve
    assert r_legacy.final_best.gene == r_codec.final_best.gene
    assert r_legacy.final_best.fitness == r_codec.final_best.fitness


def test_evolve_rwkv_codec_full_generation_equality() -> None:
    """全世代の全個体 gene が一致 (構造的にも byte-identical)."""
    kw = dict(pop_size=8, n_generations=8)
    r_legacy = evolve(_rwkv_fitness, rng=np.random.default_rng(11), codec=None, **kw)
    r_codec = evolve(
        _rwkv_fitness, rng=np.random.default_rng(11), codec=RWKVCodec(), **kw
    )
    assert len(r_legacy.generations) == len(r_codec.generations)
    for gen_l, gen_c in zip(r_legacy.generations, r_codec.generations):
        genes_l = [ind.gene for ind in gen_l.individuals]
        genes_c = [ind.gene for ind in gen_c.individuals]
        assert genes_l == genes_c


# ---------------------------------------------------------------------------
# (2) *_g operator が RWKV で byte-identical
# ---------------------------------------------------------------------------


def test_uniform_mutate_g_matches_legacy() -> None:
    g = StateUpdateGene(decay=0.4, mix=0.1, gate_str=-0.3)
    codec = RWKVCodec()
    for _ in range(20):
        old = uniform_mutate(g, 0.3, np.random.default_rng(99))
        new = uniform_mutate_g(g, codec, 0.3, np.random.default_rng(99))
        assert old == new


def test_crossover_uniform_g_matches_legacy() -> None:
    a = StateUpdateGene(decay=0.1, mix=-0.5, gate_str=1.0)
    b = StateUpdateGene(decay=0.9, mix=0.5, gate_str=-1.0)
    codec = RWKVCodec()
    for seed in range(20):
        old = crossover_uniform(a, b, np.random.default_rng(seed))
        new = crossover_uniform_g(a, b, codec, np.random.default_rng(seed))
        assert old == new


def test_initialize_random_population_g_matches_legacy() -> None:
    codec = RWKVCodec()
    old = initialize_random_population(10, np.random.default_rng(123))
    new = initialize_random_population_g(10, codec, np.random.default_rng(123))
    assert old == new


# ---------------------------------------------------------------------------
# (3) gene 型非依存性: 合成 4-dim toy gene + codec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ToyGene:
    """4-dim toy gene (as_array を持たない = codec 経由のみ)."""

    p0: float
    p1: float
    p2: float
    p3: float


class _ToyCodec:
    """_ToyGene 用 GeneCodec (box [0,1]^4)."""

    @property
    def dim(self) -> int:
        return 4

    @property
    def lower(self) -> np.ndarray:
        return np.zeros(4)

    @property
    def upper(self) -> np.ndarray:
        return np.ones(4)

    def to_array(self, gene: _ToyGene) -> np.ndarray:
        return np.array([gene.p0, gene.p1, gene.p2, gene.p3], dtype=np.float64)

    def from_array(self, arr: np.ndarray) -> _ToyGene:
        return _ToyGene(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def clip(self, gene: _ToyGene) -> _ToyGene:
        arr = np.clip(self.to_array(gene), 0.0, 1.0)
        return self.from_array(arr)


def _toy_fitness(gene: _ToyGene, rng: np.random.Generator) -> float:
    target = np.array([0.2, 0.8, 0.5, 0.1])
    arr = _ToyCodec().to_array(gene)
    return float(-np.sum((arr - target) ** 2))


def test_evolve_with_toy_codec_completes() -> None:
    """非 RWKV gene 型 (4-dim, as_array なし) で GA が完走、bounds 内、finite."""
    codec = _ToyCodec()
    result = evolve(
        _toy_fitness,
        pop_size=12,
        n_generations=20,
        rng=np.random.default_rng(2024),
        codec=codec,
    )
    assert len(result.generations) == 21
    assert all(np.isfinite(f) for f in result.best_fitness_curve)
    # 全個体 bounds 内
    for gen in result.generations:
        for ind in gen.individuals:
            arr = codec.to_array(ind.gene)
            assert np.all(arr >= 0.0) and np.all(arr <= 1.0)
    # 進化が improve (elitism 単調)
    curve = result.best_fitness_curve
    for i in range(len(curve) - 1):
        assert curve[i + 1] >= curve[i] - 1e-9


def test_evolve_with_toy_codec_deterministic() -> None:
    """toy codec でも同 seed で完全一致 (G6 相当)."""
    codec = _ToyCodec()
    r1 = evolve(_toy_fitness, pop_size=10, n_generations=10,
                rng=np.random.default_rng(55), codec=codec)
    r2 = evolve(_toy_fitness, pop_size=10, n_generations=10,
                rng=np.random.default_rng(55), codec=codec)
    assert r1.best_fitness_curve == r2.best_fitness_curve
    assert r1.diversity_curve == r2.diversity_curve


def test_toy_codec_diversity_uses_codec() -> None:
    """diversity が codec.to_array 経由で計算される (gene に as_array 不要)."""
    codec = _ToyCodec()
    result = evolve(_toy_fitness, pop_size=10, n_generations=5,
                    rng=np.random.default_rng(1), codec=codec)
    # _ToyGene は as_array を持たない → gene_matrix を呼べば AttributeError になるはず
    with pytest.raises(AttributeError):
        _ = result.generations[0].gene_matrix
    # それでも diversity_curve は計算されている (codec 経由)
    assert all(np.isfinite(d) for d in result.diversity_curve)
    assert len(result.diversity_curve) == 6


# ---------------------------------------------------------------------------
# (4) Generic 化が既存呼び出しを壊さない
# ---------------------------------------------------------------------------


def test_individual_generic_backward_compat() -> None:
    ind = Individual(gene=StateUpdateGene(0.1, 0.1, 0.1), fitness=0.3)
    assert ind.fitness == 0.3
    assert ind.gene.decay == 0.1


def test_population_generic_backward_compat() -> None:
    pop = Population(
        individuals=(
            Individual(gene=StateUpdateGene(0.1, 0.1, 0.1), fitness=0.3),
            Individual(gene=StateUpdateGene(0.5, 0.5, 0.5), fitness=0.7),
        )
    )
    assert pop.size == 2
    assert pop.best.fitness == 0.7
    assert pop.gene_matrix.shape == (2, 3)  # RWKV gene は as_array を持つ
