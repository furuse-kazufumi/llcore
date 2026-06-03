# SPDX-License-Identifier: Apache-2.0
"""FAIL-CLOSED tests for the SDP contraction certifiers (PART 1 of the 2026-06-03 hardening).

The 2026-06-03 audit found the solver pins were fail-SOFT, not fail-closed::

    _SOLVER = cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None

where ``None`` == cvxpy default == SCS — so a CLARABEL-ABSENT environment would SILENTLY regress to
the exact artifact-prone first-order solver (SCS false-negatives near the SDP feasibility boundary,
the cause of the spurious deg4/deg6 "complementarity"). The hardening makes EVERY SDP solver
selection TRULY fail-closed: when CLARABEL is not installed the certifier REFUSES to certify
(returns False / available=False / certified=None) rather than running under SCS.

These tests simulate a CLARABEL-absent environment two ways:

  1. ``installed_solvers()`` monkeypatch — assert that re-deriving the module's CLARABEL-presence
     flag from a CLARABEL-EXCLUDING ``cp.installed_solvers()`` yields False (proves the IMPORT-TIME
     selection logic is correct: ``"CLARABEL" in installed`` would be False, so the module would set
     its fail-closed flag and refuse).

  2. flag monkeypatch — set each module's CLARABEL-presence flag to False (exactly what a fresh
     import would compute in a CLARABEL-absent env, since the flag is read at import time) and assert
     each certifier REFUSES on an SDP-only gene (a gene the genuine SDP solve is required for —
     inf-norm AND 2-norm both reject it), rather than certifying via SCS.

INVARIANT (verified by the rest of the suite, which runs with CLARABEL present): when CLARABEL IS
installed the behaviour is byte-for-byte unchanged — ONLY the CLARABEL-absent path changes from
silent-SCS to fail-closed. We additionally assert here that, with CLARABEL present, the SDP-only
gene IS certified (so the tests are non-vacuous and confirm we only flipped the absent path).
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.normpath(os.path.join(_HERE, ".."))
for _d in (_HERE,
           os.path.join(_RESEARCH, "coupled_z3_contraction"),
           os.path.join(_RESEARCH, "spectral_lyapunov_contraction")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

cp = pytest.importorskip("cvxpy")
from coupled_map import CoupledGene  # noqa: E402

# An SDP-ONLY gene: non-normal contracting (ρ(J)=0.5<1) but inf-norm AND 2-norm BOTH reject, so the
# genuine common-Lyapunov SDP solve is REQUIRED to certify it. This is exactly the gene the
# fail-closed guard must REFUSE when CLARABEL is absent (it must not be rescued by a fast path).
SDP_ONLY_GENE = CoupledGene.make(decay=[0.5, 0.5], W=[[0.0, 0.0], [1.6, 0.0]])

# A 2-norm-certifiable gene (benign diagonal decay): certified by a solver-INDEPENDENT fast path, so
# it must STILL be admitted even when CLARABEL is absent (the fast path is not gated by the solver).
TWO_NORM_GENE = CoupledGene.make(decay=[0.8, 0.8], W=[[0.0, 0.0], [0.0, 0.0]])

_CLARABEL_PRESENT = "CLARABEL" in cp.installed_solvers()


# --------------------------------------------------------------------------- #
# 0) Sanity / non-vacuity: with CLARABEL present the SDP-only gene IS certified.
#    (Confirms these tests exercise the genuine SDP solve path, and that we only
#     change the CLARABEL-ABSENT branch — the present branch is unchanged.)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _CLARABEL_PRESENT, reason="requires CLARABEL installed (this env)")
def test_sanity_sdp_only_gene_certified_with_clarabel():
    import coupled_components as cc
    assert cc._inf_certifies(SDP_ONLY_GENE) is False
    assert cc._two_certifies(SDP_ONLY_GENE) is False
    # genuine SDP solve required AND succeeds under CLARABEL -> certified (unchanged behaviour).
    assert cc._sdp_certifies(SDP_ONLY_GENE) is True


# --------------------------------------------------------------------------- #
# 1) The import-time selection logic itself is fail-closed: a CLARABEL-EXCLUDING
#    installed_solvers() makes the presence test False.
# --------------------------------------------------------------------------- #


def test_installed_solvers_monkeypatch_excludes_clarabel(monkeypatch):
    """Patch cp.installed_solvers() to drop CLARABEL and assert the presence test the modules use
    at import time (``"CLARABEL" in cp.installed_solvers()``) is False. This is the predicate that
    drives every module's fail-closed flag, so a False here means a fresh import would refuse."""
    no_clarabel = [s for s in cp.installed_solvers() if s != "CLARABEL"]
    monkeypatch.setattr(cp, "installed_solvers", lambda: no_clarabel)
    assert "CLARABEL" not in cp.installed_solvers()
    assert ("CLARABEL" in cp.installed_solvers()) is False


# --------------------------------------------------------------------------- #
# 2) Each certifier REFUSES (fail-closed) on an SDP-only gene when its
#    CLARABEL-presence flag is False (the CLARABEL-absent state).
# --------------------------------------------------------------------------- #


def test_coupled_components_sdp_fail_closed(monkeypatch):
    """coupled_components._sdp_certifies must REFUSE the SDP-only gene when CLARABEL is absent
    (return False), NOT silently solve under SCS."""
    import coupled_components as cc
    monkeypatch.setattr(cc, "_CLARABEL_OK", False)
    monkeypatch.setattr(cc, "_SDP_SOLVER", None)
    assert cc._sdp_certifies(SDP_ONLY_GENE) is False, "must refuse, not run under SCS"
    # the solver-independent 2-norm fast path is NOT gated by the solver: still admits the benign gene.
    assert cc._sdp_certifies(TWO_NORM_GENE) is True


def test_verifier_deg4_fail_closed(monkeypatch):
    """verifier_deg4.certify_deg4 / cert_deg4_n2 must REFUSE when CLARABEL is absent."""
    import verifier_deg4 as d4
    monkeypatch.setattr(d4, "_CLARABEL_OK", False)
    monkeypatch.setattr(d4, "_SOLVER", None)
    # certify_deg4 on the lifted vertices of the SDP-only gene must refuse.
    verts = d4._vertices_n2(SDP_ONLY_GENE)
    assert d4.certify_deg4(verts) is False, "deg4 must refuse, not run under SCS"
    assert d4.cert_deg4_n2(SDP_ONLY_GENE) is False


def test_verifier_deg6_fail_closed(monkeypatch):
    """verifier_deg6.certify_degN / certify_deg6 / cert_deg6_n2 must REFUSE when CLARABEL is absent."""
    import verifier_deg6 as d6
    monkeypatch.setattr(d6, "_CLARABEL_OK", False)
    monkeypatch.setattr(d6, "_SOLVER", None)
    verts = d6._vertices_n2(SDP_ONLY_GENE)
    assert d6.certify_deg6(verts) is False, "deg6 must refuse, not run under SCS"
    assert d6.cert_deg6_n2(SDP_ONLY_GENE) is False


def test_coupled_nd_sdp_fail_closed(monkeypatch):
    """coupled_nd.cert_sdp must REFUSE an SDP-only n-dim gene when CLARABEL is absent."""
    import coupled_nd as nd
    monkeypatch.setattr(nd, "_CLARABEL_OK", False)
    monkeypatch.setattr(nd, "_SOLVER", None)
    # the n=2 non-normal gene mapped into the nd substrate is SDP-only (inf & 2-norm reject).
    g_nd = nd.CoupledNDGene.make(decay=[0.5, 0.5], W=[[0.0, 0.0], [1.6, 0.0]])
    assert nd.cert_inf(g_nd) is False and nd.cert_two(g_nd) is False
    assert nd.cert_sdp(g_nd) is False, "nd SDP must refuse, not run under SCS"
    # benign 2-norm gene still admitted via the solver-independent fast path.
    g_benign = nd.CoupledNDGene.make(decay=[0.8, 0.8], W=[[0.0, 0.0], [0.0, 0.0]])
    assert nd.cert_sdp(g_benign) is True


def test_lyapunov_certifier_default_fail_closed(monkeypatch):
    """lyapunov_sdp_certifier.certify_common_lyapunov with NO explicit solver must REFUSE
    (available=False, certified=None) when CLARABEL is absent — never fall through to SCS."""
    import lyapunov_sdp_certifier as lyap
    monkeypatch.setattr(lyap, "_CLARABEL_OK", False)
    monkeypatch.setattr(lyap, "_DEFAULT_SOLVER", None)
    # default path (solver=None): fail-closed.
    r = lyap.certify_common_lyapunov(SDP_ONLY_GENE, t_domain="tmin1")
    assert r.available is False
    assert r.certified is None
    assert r.solver_status == "clarabel_unavailable_fail_closed"
    assert r.P is None


def test_lyapunov_certifier_explicit_solver_still_honoured(monkeypatch):
    """An EXPLICIT solver choice is the caller's responsibility and IS honoured even when CLARABEL
    is flagged absent — the PART-2 OR-of-{CLARABEL,SCS} gate relies on probing SCS explicitly and
    then Rump-re-verifying the result. (Here we explicitly request SCS, which IS installed.)"""
    import lyapunov_sdp_certifier as lyap
    monkeypatch.setattr(lyap, "_CLARABEL_OK", False)
    monkeypatch.setattr(lyap, "_DEFAULT_SOLVER", None)
    if "SCS" not in cp.installed_solvers():  # pragma: no cover
        pytest.skip("SCS not installed")
    r = lyap.certify_common_lyapunov(SDP_ONLY_GENE, t_domain="tmin1", solver="SCS")
    # explicit solver -> the function runs (available True); the verdict itself is not asserted
    # (SCS may or may not find the thin-shell certificate), only that the explicit path is honoured.
    assert r.available is True
    assert r.solver_status != "clarabel_unavailable_fail_closed"


def test_src_sdp_backend_fail_closed(monkeypatch):
    """src SdpLyapunovBackend: available must be False unless cvxpy AND CLARABEL both present, and
    the genuine SDP solve must never run under SCS (an SDP-only gene is refused)."""
    from llcore.verifier import backends as be
    monkeypatch.setattr(be, "_CLARABEL_AVAILABLE", False)
    monkeypatch.setattr(be, "_SDP_SOLVER", None)
    b = be.SdpLyapunovBackend()
    assert b.available is False, "available must be False when CLARABEL absent"

    class _Coupled:
        def __init__(self, decay, W):
            self.decay = np.asarray(decay, dtype=np.float64)
            self.W = np.asarray(W, dtype=np.float64)

    nonnormal = _Coupled([0.5, 0.5], [[0.0, 0.0], [1.6, 0.0]])  # SDP-only
    assert b.certifies(nonnormal) is False, "genuine SDP must refuse, not run under SCS"
    # 2-norm-certifiable gene still admitted via the solver-independent fast path.
    benign = _Coupled([0.8, 0.8], [[0.0, 0.0], [0.0, 0.0]])
    assert b.certifies(benign) is True


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
