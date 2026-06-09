# SPDX-License-Identifier: Apache-2.0
"""F8 framework 性 — 3 plug-point swap が **1 オブジェクト差替で機能する**ことの単体テスト。

North Star #4 (b): 新 base / 新 changeop / 新 certifier を 1 オブジェクト差替で載せ替えられる
拡張性 (GeneCodec / Objective / VerifierBackend をテスト化)。

検証方針 (src 無改変, additive):
- 3 plug-point を src ``llcore.evolution.minimal_ga.evolve`` に各々差し替えて回し、fitness が
  finite かつ best curve が劣化しない (elitism で単調非減少) ことを assert。
- GeneCodec を差し替えると別次元 substrate (dim) に載ること。
- VerifierBackend (none/inf_norm/two_norm/sdp) が各々 certifies を返し、admit 個数が
  none ≥ certifier であること (gate が篩として機能 = soundness ladder)。
- swap が決定論的 (同 seed → 同 fitness) であること。

実行: ``py -3.11 -m pytest research/rllm_pivot/test_framework_f8.py -q``
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

# path: src (進化ループ) + coupled_nd (3 plug-point) + 本 dir (phase2_framework_f8) を additive 挿入。
_HERE = os.path.dirname(__file__)
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
_COUPLED = os.path.abspath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
for _p in (_SRC, _COUPLED, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coupled_nd as C  # noqa: E402
from llcore.evolution.minimal_ga import evolve  # noqa: E402
from phase2_framework_f8 import SrcCodecAdapter, _wrap_fitness  # noqa: E402

SEED = 20260609
POP = 12
GENS = 8


def _run(adapter, objective, verifier, seed=SEED):
    rng = np.random.default_rng(seed)
    return evolve(
        _wrap_fitness(adapter, objective, verifier),
        pop_size=POP, n_generations=GENS, mutation_sigma=0.18, rng=rng, codec=adapter,
    )


# --------------------------------------------------------------------------- #
# (i) GeneCodec plug-point: n / 基質を 1 オブジェクト差替で載せ替え
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 4])
def test_genecodec_swap_runs(n):
    """CoupledNDGeneCodec(n) を差し替えるだけで別次元 substrate に evolve が載る。"""
    codec = C.CoupledNDGeneCodec(n)
    adapter = SrcCodecAdapter(codec)
    obj = C.RotationNDObjective(n=n)
    res = _run(adapter, obj, None)
    # dim が n に応じて変わる (別基質に載った証拠)
    assert adapter.dim == n + n * n
    # fitness が finite かつ best curve が劣化しない (elitism 単調非減少)
    fb = res.final_best.fitness
    assert math.isfinite(fb)
    assert res.best_fitness_curve[-1] >= res.best_fitness_curve[0] - 1e-9
    # gene が CoupledNDGene に復元できる
    gene = adapter.to_gene(res.final_best.gene)
    assert isinstance(gene, C.CoupledNDGene)
    assert gene.n == n


def test_genecodec_swap_changes_dim():
    """別 GeneCodec オブジェクトに差し替えると dim が変わる (基質載せ替えの本質)。"""
    dims = {n: SrcCodecAdapter(C.CoupledNDGeneCodec(n)).dim for n in (2, 3, 4)}
    assert dims == {2: 6, 3: 12, 4: 20}
    assert len(set(dims.values())) == 3  # 全て異なる = 真に別次元


# --------------------------------------------------------------------------- #
# (ii) Objective plug-point: task を 1 オブジェクト差替で載せ替え
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("period,radius,amp", [(10.0, 0.93, 0.40), (8.0, 0.88, 0.30), (13.0, 0.95, 0.45)])
def test_objective_swap_runs(period, radius, amp):
    """RotationNDObjective を別 task params で差し替えるだけで evolve が回る。"""
    codec = C.CoupledNDGeneCodec(3)
    adapter = SrcCodecAdapter(codec)
    obj = C.RotationNDObjective(n=3, period=period, radius=radius, amp=amp)
    res = _run(adapter, obj, None)
    assert math.isfinite(res.final_best.fitness)
    assert res.best_fitness_curve[-1] >= res.best_fitness_curve[0] - 1e-9


def test_objective_swap_distinct_tasks():
    """異なる Objective は異なる fitness 地形を与える (task 載せ替えが効いている)。"""
    codec = C.CoupledNDGeneCodec(3)
    adapter = SrcCodecAdapter(codec)
    a = _run(adapter, C.RotationNDObjective(n=3, period=8.0, radius=0.88, amp=0.30), None)
    b = _run(adapter, C.RotationNDObjective(n=3, period=13.0, radius=0.95, amp=0.45), None)
    # 別 task なので best fitness は (ほぼ確実に) 異なる
    assert a.final_best.fitness != b.final_best.fitness


# --------------------------------------------------------------------------- #
# (iii) VerifierBackend plug-point: none/inf_norm/two_norm/sdp を 1 オブジェクト差替
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("vname", ["none", "inf_norm", "two_norm", "sdp"])
def test_verifier_swap_runs(vname):
    """make_nd_verifier(vname) を差し替えるだけで evolve が回り fitness が出る。"""
    codec = C.CoupledNDGeneCodec(3)
    adapter = SrcCodecAdapter(codec)
    obj = C.RotationNDObjective(n=3)
    verifier = C.make_nd_verifier(vname)
    assert verifier.name == vname
    res = _run(adapter, obj, verifier)
    assert math.isfinite(res.final_best.fitness)


def test_verifier_admit_ladder():
    """soundness ladder: none は全 admit、certifier は部分 admit (gate が篩として機能)。

    none(無条件 admit) ⊇ {inf_norm, two_norm, sdp}。かつ two_norm ⊇ inf_norm
    (two_norm は inf_norm の緩和 = inf_norm が admit する gene は two_norm も admit)。
    """
    codec = C.CoupledNDGeneCodec(3)
    adapter = SrcCodecAdapter(codec)
    # 固定 gene 集合で各 verifier の admit を測る (evolve に依存しない直接判定)
    rng = np.random.default_rng(SEED)
    genes = [adapter.to_gene(codec.clip(codec.random(rng))) for _ in range(200)]
    admit = {}
    for vname in ("none", "inf_norm", "two_norm", "sdp"):
        v = C.make_nd_verifier(vname)
        admit[vname] = sum(1 for g in genes if v.certifies(g))
    # none は全 admit
    assert admit["none"] == len(genes)
    # certifier は none 以下 (篩として機能)
    for vname in ("inf_norm", "two_norm", "sdp"):
        assert admit[vname] <= admit["none"]
    # two_norm は inf_norm の緩和 (inf_norm ⊆ two_norm ⊆ sdp の admit 包含)
    assert admit["inf_norm"] <= admit["two_norm"] <= admit["sdp"]


def test_verifier_subset_relation_per_gene():
    """per-gene: inf_norm が admit する gene は two_norm / sdp も admit する (sound 緩和)。"""
    codec = C.CoupledNDGeneCodec(3)
    adapter = SrcCodecAdapter(codec)
    rng = np.random.default_rng(SEED + 7)
    v_inf = C.make_nd_verifier("inf_norm")
    v_two = C.make_nd_verifier("two_norm")
    v_sdp = C.make_nd_verifier("sdp")
    for _ in range(150):
        g = adapter.to_gene(codec.clip(codec.random(rng)))
        if v_inf.certifies(g):
            assert v_two.certifies(g)
            assert v_sdp.certifies(g)
        if v_two.certifies(g):
            assert v_sdp.certifies(g)


# --------------------------------------------------------------------------- #
# 決定論性 (seed 固定 → 同結果)
# --------------------------------------------------------------------------- #
def test_swap_deterministic():
    """同 seed で 2 回回すと best fitness が byte-identical (再現性)。"""
    codec = C.CoupledNDGeneCodec(3)
    adapter = SrcCodecAdapter(codec)
    obj = C.RotationNDObjective(n=3)
    a = _run(adapter, obj, C.make_nd_verifier("two_norm"))
    b = _run(adapter, obj, C.make_nd_verifier("two_norm"))
    assert a.final_best.fitness == b.final_best.fitness
    assert a.best_fitness_curve == b.best_fitness_curve


# --------------------------------------------------------------------------- #
# adapter が src GeneCodec protocol を満たす (構造的タイピング)
# --------------------------------------------------------------------------- #
def test_adapter_satisfies_genecodec_protocol():
    """SrcCodecAdapter が src の GeneCodec runtime_checkable protocol を満たす。"""
    from llcore.kernel.protocol import GeneCodec
    adapter = SrcCodecAdapter(C.CoupledNDGeneCodec(3))
    assert isinstance(adapter, GeneCodec)
    # 往復: from_array(to_array(x)) == x
    rng = np.random.default_rng(SEED)
    arr = adapter.from_array(rng.random(adapter.dim))
    assert np.allclose(adapter.to_array(arr), arr)
    # bounds shape
    assert adapter.lower.shape == (adapter.dim,)
    assert adapter.upper.shape == (adapter.dim,)


def test_make_nd_verifier_rejects_unknown():
    """未知 verifier 名は fail-loud (ValueError)。"""
    with pytest.raises(ValueError):
        C.make_nd_verifier("bogus")
