# SPDX-License-Identifier: Apache-2.0
"""kernel plugin Protocol + RWKV 準拠例の test (S1).

検証命題:
- RWKVCodec / RWKVKernel / RWKVStateNormBackend が各 Protocol を満たす (structural)
- RWKV simulate が既存 run_sequence と一致 (委譲の正しさ = 挙動不変)
- apply_change_op が既存 apply_changeop と一致
- VerifierBackend が本流 verify_gene_safe と一致
- codec の往復 (to_array → from_array) が同値、bounds が clipped 範囲と一致
"""
from __future__ import annotations

import numpy as np
import pytest

from llcore.kernel import (
    GeneCodec,
    Kernel,
    RWKVCodec,
    RWKVKernel,
    RWKVStateNormBackend,
    Trajectory,
    VerifierBackend,
)
from llcore.state_update import StateUpdateGene, run_sequence
from llcore.verifier import ChangeOp, apply_changeop
from llcore.verifier.invariants import verify_gene_safe as _verify_gene_safe


@pytest.fixture
def gene() -> StateUpdateGene:
    return StateUpdateGene(decay=0.6, mix=0.3, gate_str=0.4)


@pytest.fixture
def inputs() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(16, 4))


# ---------------------------------------------------------------------------
# Protocol 準拠 (structural typing)
# ---------------------------------------------------------------------------


def test_rwkv_codec_satisfies_protocol() -> None:
    assert isinstance(RWKVCodec(), GeneCodec)


def test_rwkv_kernel_satisfies_protocol() -> None:
    assert isinstance(RWKVKernel(), Kernel)


def test_rwkv_backend_satisfies_protocol() -> None:
    assert isinstance(RWKVStateNormBackend(), VerifierBackend)


# ---------------------------------------------------------------------------
# GeneCodec — 往復 + bounds
# ---------------------------------------------------------------------------


def test_codec_dim_and_bounds() -> None:
    c = RWKVCodec()
    assert c.dim == 3
    np.testing.assert_array_equal(c.lower, [0.0, -1.0, -2.0])
    np.testing.assert_array_equal(c.upper, [1.0, 1.0, 2.0])


def test_codec_roundtrip(gene: StateUpdateGene) -> None:
    c = RWKVCodec()
    arr = c.to_array(gene)
    assert arr.shape == (3,)
    restored = c.from_array(arr)
    assert restored == gene


def test_codec_lower_upper_are_copies() -> None:
    """lower/upper は内部 array の copy を返す (mutate 防止)."""
    c = RWKVCodec()
    lo = c.lower
    lo[0] = 999.0
    assert c.lower[0] == 0.0  # 内部状態は不変


def test_codec_clip_delegates_to_clipped() -> None:
    c = RWKVCodec()
    out_of_range = StateUpdateGene(decay=5.0, mix=-9.0, gate_str=9.0)
    assert c.clip(out_of_range) == out_of_range.clipped()


# ---------------------------------------------------------------------------
# Kernel.simulate — 既存 run_sequence と一致 (挙動不変)
# ---------------------------------------------------------------------------


def test_simulate_matches_run_sequence(
    gene: StateUpdateGene, inputs: np.ndarray
) -> None:
    k = RWKVKernel()
    traj = k.simulate(inputs, gene)
    assert isinstance(traj, Trajectory)
    assert traj.kind == "state"
    assert traj.events == ()
    expected = run_sequence(inputs, gene)
    np.testing.assert_array_equal(traj.primary, expected)


def test_simulate_respects_initial_state(
    gene: StateUpdateGene, inputs: np.ndarray
) -> None:
    k = RWKVKernel()
    init = np.full(4, 0.2, dtype=np.float64)
    traj = k.simulate(inputs, gene, initial_state=init)
    expected = run_sequence(inputs, gene, initial_state=init)
    np.testing.assert_array_equal(traj.primary, expected)


# ---------------------------------------------------------------------------
# Kernel.apply_change_op — 既存 apply_changeop と一致 + op_type 宣言
# ---------------------------------------------------------------------------


def test_change_op_types_match_existing() -> None:
    assert RWKVKernel().change_op_types == (
        "decay_shift",
        "mix_shift",
        "gate_shift",
        "kernel_swap_mock",
    )


@pytest.mark.parametrize(
    "op",
    [
        ChangeOp(op_type="decay_shift", delta=-0.1),
        ChangeOp(op_type="mix_shift", delta=0.2),
        ChangeOp(op_type="gate_shift", delta=0.5),
        ChangeOp(op_type="kernel_swap_mock", delta=1.0),
    ],
)
def test_apply_change_op_matches_existing(gene: StateUpdateGene, op: ChangeOp) -> None:
    k = RWKVKernel()
    assert k.apply_change_op(gene, op) == apply_changeop(gene, op)


# ---------------------------------------------------------------------------
# VerifierBackend — 本流 verify_gene_safe と一致
# ---------------------------------------------------------------------------


def test_backend_matches_verify_gene_safe(gene: StateUpdateGene) -> None:
    backend = RWKVStateNormBackend()
    got = backend.verify_gene_safe(gene)
    expected = _verify_gene_safe(gene, max_input_abs=1.0, state_bound=1.0)
    assert got.ok == expected.ok
    assert got.used_z3 == expected.used_z3


def test_backend_is_available_matches_z3() -> None:
    from llcore.verifier import is_z3_available

    assert RWKVStateNormBackend().is_available() == is_z3_available()


def test_backend_admits_safe_gene(gene: StateUpdateGene) -> None:
    """clip 範囲内の素直な gene は admit される (z3 有無に関わらず ok=True)."""
    assert RWKVStateNormBackend().verify_gene_safe(gene).ok is True


def test_backend_custom_bounds_passthrough(gene: StateUpdateGene) -> None:
    backend = RWKVStateNormBackend(max_input_abs=0.5, state_bound=2.0)
    got = backend.verify_gene_safe(gene)
    expected = _verify_gene_safe(gene, max_input_abs=0.5, state_bound=2.0)
    assert got.ok == expected.ok
