# SPDX-License-Identifier: Apache-2.0
"""Stage 3b — verifier backend plugin tests.

Covers the pluggable contraction backends: scalar closed-form (default), coupled induced-norm
(inf/two), and SDP-Lyapunov (cvxpy optional, fail-closed). The headline arc result must show
up in the production API: the ∞-norm backend REJECTS a rotational contraction that the 2-norm
and SDP backends ADMIT.
"""
from __future__ import annotations

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene
from llcore.verifier import (
    available_backends,
    cvxpy_available,
    get_verifier_backend,
)
from llcore.verifier.backends import SdpLyapunovBackend


class _Coupled:
    def __init__(self, decay, W):
        self.decay = np.asarray(decay, dtype=float)
        self.W = np.asarray(W, dtype=float)


# rotational contraction: ∞-norm rejects (row abs-sums > 1) but ρ<1 and σ_max<1 (2-norm/SDP ok)
ROT = _Coupled([0.3, 0.3], [[0.646, -0.781], [0.781, 0.646]])
SAFE = _Coupled([0.9, 0.9], [[0.1, 0.05], [0.05, 0.1]])
EXPANSIVE = _Coupled([0.1, 0.1], [[1.5, 0.9], [0.9, 1.5]])


def test_registry_and_unknown():
    for name in ("closed_form_scalar", "inf_norm", "two_norm", "sdp_lyapunov"):
        assert get_verifier_backend(name).name == name
    with pytest.raises(KeyError):
        get_verifier_backend("nope")
    assert get_verifier_backend().name == "closed_form_scalar"  # default


def test_scalar_closed_form_backend():
    b = get_verifier_backend("closed_form_scalar")
    assert b.available is True
    assert b.certifies(StateUpdateGene(decay=0.5, mix=0.0, gate_str=0.0)) is True
    # gate_str=2 with decay=0 => |decay + (1-decay)*gate_str| = 2 > 1 (not a contraction)
    assert b.certifies(StateUpdateGene(decay=0.0, mix=0.0, gate_str=2.0)) is False


def test_coupled_norm_backends_distinguish_rotation():
    inf = get_verifier_backend("inf_norm")
    two = get_verifier_backend("two_norm")
    # the arc payoff, in the production API:
    assert inf.certifies(ROT) is False        # ∞-norm OVER-rejects the rotation
    assert two.certifies(ROT) is True         # 2-norm admits it (sound)
    for b in (inf, two):
        assert b.certifies(SAFE) is True
        assert b.certifies(EXPANSIVE) is False


def test_sdp_backend_admits_rotation_and_rejects_expansive():
    sdp = get_verifier_backend("sdp_lyapunov")
    assert sdp.available is cvxpy_available()
    if not sdp.available:
        pytest.skip("cvxpy not installed")
    assert sdp.certifies(ROT) is True
    assert sdp.certifies(SAFE) is True
    assert sdp.certifies(EXPANSIVE) is False


def test_sdp_fail_closed_without_cvxpy(monkeypatch):
    """If cvxpy is unavailable, the SDP backend reports unavailable and a non-2-norm gene is
    rejected (fail-closed) rather than raising."""
    monkeypatch.setattr("llcore.verifier.backends._CVXPY_AVAILABLE", False)
    b = SdpLyapunovBackend()
    assert b.available is False
    # ROT is 2-norm-certifiable, so the numpy fast-path still admits it (no solver needed):
    assert b.certifies(ROT) is True
    # a gene needing the actual solver (2-norm fails) must be rejected fail-closed, not raise.
    nonnormal = _Coupled([0.5, 0.5], [[0.0, 0.0], [1.6, 0.0]])
    assert b.certifies(nonnormal) is False


def test_malformed_gene_is_fail_closed():
    for name in ("inf_norm", "two_norm", "sdp_lyapunov"):
        assert get_verifier_backend(name).certifies(object()) is False


def test_available_backends_map():
    m = available_backends()
    assert m["closed_form_scalar"] is True and m["inf_norm"] is True and m["two_norm"] is True
    assert m["sdp_lyapunov"] is cvxpy_available()
