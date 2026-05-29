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
from llcore.kernel import GeneCodec, RWKVCodec
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


# ===========================================================================
# PoC 強化 (workflow 4-lens + Codex review の TRUE findings 対応, re-PoC)
# ===========================================================================


# ---------------------------------------------------------------------------
# 核心契約: codec.clip が box では表せない依存制約を吸収する (S2 設計の中核命題)
# 設計 doc §2.1: 「LIF は V_reset<V_th、Izhikevich は c<V_PEAK という box clip では
# 表せない依存制約があるため、単純 bounds clip では不足。clipped() の委譲で吸収」。
# init_g は各 param を box から独立 draw するため、依存制約 gene では clip が repair
# しないと制約破りが残る (lens=正当性 Medium / lens=test網羅 Medium)。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DepGene:
    """依存制約 lo < hi を持つ 2-dim gene (box では表せない制約)."""

    lo: float
    hi: float


class _DepCodec:
    """_DepGene 用 codec — clip が **依存制約 lo<hi を repair** する (正しい実装)."""

    @property
    def dim(self) -> int:
        return 2

    @property
    def lower(self) -> np.ndarray:
        return np.zeros(2)

    @property
    def upper(self) -> np.ndarray:
        return np.ones(2)

    def to_array(self, gene: _DepGene) -> np.ndarray:
        return np.array([gene.lo, gene.hi], dtype=np.float64)

    def from_array(self, arr: np.ndarray) -> _DepGene:
        return _DepGene(float(arr[0]), float(arr[1]))

    def clip(self, gene: _DepGene) -> _DepGene:
        arr = np.clip(self.to_array(gene), 0.0, 1.0)
        # 依存制約 repair (box clip では不可能な部分): sort で lo<=hi を保証し、
        # 一致時は [0,1] 内で 0.05 分離 (端でも破綻しない堅牢版)。
        lo, hi = sorted((float(arr[0]), float(arr[1])))
        if lo >= hi:
            if hi <= 0.95:
                hi = hi + 0.05
            else:
                lo = lo - 0.05
        return _DepGene(lo, hi)


class _DepCodecBoxOnly(_DepCodec):
    """teeth 用: clip が box のみで依存制約を repair しない (誤った実装)."""

    def clip(self, gene: _DepGene) -> _DepGene:
        arr = np.clip(self.to_array(gene), 0.0, 1.0)
        return _DepGene(float(arr[0]), float(arr[1]))


def _dep_fitness(gene: _DepGene, rng: np.random.Generator) -> float:
    return float(gene.hi - gene.lo)  # 制約満たす個体ほど高 (lo<hi)


def test_clip_repairs_dependent_constraint_after_init() -> None:
    """init_g 後、依存制約 lo<hi が codec.clip で全個体満たされる (核心契約)."""
    codec = _DepCodec()
    genes = initialize_random_population_g(200, codec, np.random.default_rng(7))
    assert all(g.lo < g.hi for g in genes), "依存制約 repair に失敗した gene がある"


def test_box_only_clip_leaves_violations_teeth() -> None:
    """teeth: box-only clip だと init 後に依存制約破りが残る (= 核心契約 test が空でない証明)."""
    codec = _DepCodecBoxOnly()
    genes = initialize_random_population_g(200, codec, np.random.default_rng(7))
    violations = sum(1 for g in genes if g.lo >= g.hi)
    assert violations > 0, "box-only clip で violation が出ないなら test に teeth がない"


def test_clip_repairs_dependent_constraint_after_mutate() -> None:
    """mutate_g 後も依存制約が codec.clip で維持される."""
    codec = _DepCodec()
    g = _DepGene(0.2, 0.8)
    for seed in range(50):
        m = uniform_mutate_g(g, codec, 1.0, np.random.default_rng(seed))
        assert m.lo < m.hi


def test_evolve_with_dependent_constraint_codec_maintains_constraint() -> None:
    """evolve 全体 (init+mutate+crossover) を通して依存制約が維持される."""
    codec = _DepCodec()
    result = evolve(
        _dep_fitness, pop_size=12, n_generations=15,
        rng=np.random.default_rng(2026), codec=codec,
    )
    for gen in result.generations:
        for ind in gen.individuals:
            assert ind.gene.lo < ind.gene.hi


# ---------------------------------------------------------------------------
# byte-identity の config sweep (lens=test網羅 Medium): crossover_rate / elitism / initial_pop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crossover_rate,elitism",
    [(0.0, 1), (1.0, 1), (0.5, 3), (0.0, 3), (1.0, 2)],
)
def test_evolve_codec_equals_legacy_across_configs(
    crossover_rate: float, elitism: int
) -> None:
    """多様な config で codec=RWKVCodec() ≡ codec=None (byte-identity が config 非依存)."""
    kw = dict(
        pop_size=10, n_generations=12,
        crossover_rate=crossover_rate, elitism=elitism, mutation_sigma=0.25,
    )
    r_l = evolve(_rwkv_fitness, rng=np.random.default_rng(42), codec=None, **kw)
    r_c = evolve(_rwkv_fitness, rng=np.random.default_rng(42), codec=RWKVCodec(), **kw)
    assert r_l.best_fitness_curve == r_c.best_fitness_curve
    assert r_l.diversity_curve == r_c.diversity_curve
    for gen_l, gen_c in zip(r_l.generations, r_c.generations):
        assert [i.gene for i in gen_l.individuals] == [i.gene for i in gen_c.individuals]


def test_evolve_initial_pop_with_codec_equals_legacy() -> None:
    """initial_pop 明示 + codec でも legacy と byte-identical (init bypass path)."""
    init = [
        StateUpdateGene(decay=0.1 * i, mix=0.05 * i - 0.2, gate_str=0.1 * i - 0.3)
        for i in range(6)
    ]
    kw = dict(pop_size=6, n_generations=8, mutation_sigma=0.2)
    r_l = evolve(_rwkv_fitness, rng=np.random.default_rng(9), codec=None,
                 initial_pop=list(init), **kw)
    r_c = evolve(_rwkv_fitness, rng=np.random.default_rng(9), codec=RWKVCodec(),
                 initial_pop=list(init), **kw)
    assert r_l.best_fitness_curve == r_c.best_fitness_curve
    assert r_l.final_best.gene == r_c.final_best.gene


# ---------------------------------------------------------------------------
# fail-closed 検証分岐 (lens=test網羅 Low): codec path でも ValueError gate が効く
# ---------------------------------------------------------------------------


def test_evolve_validation_errors_with_codec() -> None:
    codec = RWKVCodec()
    with pytest.raises(ValueError):
        evolve(_rwkv_fitness, pop_size=5, elitism=6, codec=codec)
    with pytest.raises(ValueError):
        evolve(_rwkv_fitness, pop_size=5, tournament_k=6, codec=codec)
    with pytest.raises(ValueError):
        evolve(
            _rwkv_fitness, pop_size=5,
            initial_pop=[StateUpdateGene(0.1, 0.1, 0.1)] * 3,  # size != pop_size
            codec=codec,
        )


# ---------------------------------------------------------------------------
# RNG ストリーム直接比較 (Codex Low / lens=test網羅 Low): 消費後の state まで一致
# ---------------------------------------------------------------------------


def test_operator_rng_state_identical_to_legacy() -> None:
    """*_g operator が legacy と同一本数・同一順で RNG を消費 (bit_generator state 一致)."""
    codec = RWKVCodec()
    g = StateUpdateGene(decay=0.4, mix=0.1, gate_str=-0.3)
    # mutate
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    uniform_mutate(g, 0.3, r1)
    uniform_mutate_g(g, codec, 0.3, r2)
    assert r1.bit_generator.state == r2.bit_generator.state
    # crossover
    a, b = StateUpdateGene(0.1, -0.5, 1.0), StateUpdateGene(0.9, 0.5, -1.0)
    r3, r4 = np.random.default_rng(13), np.random.default_rng(13)
    crossover_uniform(a, b, r3)
    crossover_uniform_g(a, b, codec, r4)
    assert r3.bit_generator.state == r4.bit_generator.state
    # init
    r5, r6 = np.random.default_rng(21), np.random.default_rng(21)
    initialize_random_population(8, r5)
    initialize_random_population_g(8, codec, r6)
    assert r5.bit_generator.state == r6.bit_generator.state


# ---------------------------------------------------------------------------
# clip 発動 (lens=test網羅 Medium): bounds 外 mutation が境界に切り詰められる
# ---------------------------------------------------------------------------


def test_uniform_mutate_g_clips_out_of_bounds() -> None:
    """大 sigma で bounds 超過しても codec.clip で範囲内に収まる (clip 発動の直接確認)."""
    codec = RWKVCodec()
    g = StateUpdateGene(decay=0.99, mix=0.99, gate_str=1.99)
    for seed in range(30):
        m = uniform_mutate_g(g, codec, 5.0, np.random.default_rng(seed))
        a = m.as_array()
        assert 0.0 <= a[0] <= 1.0
        assert -1.0 <= a[1] <= 1.0
        assert -2.0 <= a[2] <= 2.0


# ---------------------------------------------------------------------------
# diversity 数値正しさ (lens=test網羅 Low): codec.to_array stack の pooled var
# ---------------------------------------------------------------------------


def test_toy_codec_diversity_numeric_correctness() -> None:
    """diversity_curve[0] が codec.to_array stack の pooled variance と数値一致."""
    codec = _ToyCodec()
    genes = [_ToyGene(0.1, 0.2, 0.3, 0.4), _ToyGene(0.5, 0.6, 0.7, 0.8)]
    result = evolve(
        _toy_fitness, pop_size=2, n_generations=0,
        initial_pop=list(genes), rng=np.random.default_rng(0), codec=codec,
    )
    expected = float(np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).var())
    assert abs(result.diversity_curve[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# 任意 codec の Protocol 準拠 (lens=test網羅 Low): isinstance structural
# ---------------------------------------------------------------------------


def test_custom_codecs_satisfy_gene_codec_protocol() -> None:
    assert isinstance(_ToyCodec(), GeneCodec)
    assert isinstance(_DepCodec(), GeneCodec)
    assert isinstance(_DepCodecBoxOnly(), GeneCodec)
