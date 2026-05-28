# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 7a — VNN-COMP `online-arch-evo` reference implementation.

Mirrors G1-G7 of `scripts/poc_7a_vnn_comp_reference_impl.py` plus deeper
white-box assertions on the StepVerdict / NetworkState dataclasses and on
the unsat-witness `.smt2` format.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from llcore.state_update import StateUpdateGene

# Import the reference impl as a module so we can test individual pieces.
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _PROJ_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import poc_7a_vnn_comp_reference_impl as ref  # noqa: E402


# ---------------------------------------------------------------------------
# G1-G7 reproducers (mirror the script's self-test)
# ---------------------------------------------------------------------------


def test_g1_handshake_emits_version_and_kernel() -> None:
    ok, line = ref.gate_g1_handshake()
    assert ok
    assert "0.1" in line
    assert "RwkvTimeMix" in line


def test_g2_init_state_holds(tmp_path: Path) -> None:
    ok, detail = ref.gate_g2_init()
    assert ok
    assert "1 block" in detail


def test_g3_five_step_sequence(tmp_path: Path) -> None:
    """Five-step canonical reparam_inplace sequence yields the expected verdicts."""
    ok, detail = ref.gate_g3_five_step_seq(tmp_path)
    assert ok, detail


def test_g4_per_step_budget(tmp_path: Path) -> None:
    """Each step finishes within the 500 ms budget; typical is sub-50 ms."""
    ok, detail = ref.gate_g4_per_step_budget(tmp_path)
    assert ok, detail


def test_g5_determinism(tmp_path: Path) -> None:
    """Same sequence run twice yields identical verdicts."""
    ok, detail = ref.gate_g5_determinism(tmp_path)
    assert ok, detail


def test_g6_family_soundness_decay_only() -> None:
    """family_soundness on f_decay_only returns 'sound'."""
    ok, detail = ref.gate_g6_family_soundness()
    assert ok, detail


def test_g7_unsupported_kernel_handled() -> None:
    """insert_subblock with non-RwkvTimeMix kernel returns 'unsupported'."""
    ok, detail = ref.gate_g7_unsupported_kernel()
    assert ok, detail


# ---------------------------------------------------------------------------
# Additional white-box tests
# ---------------------------------------------------------------------------


def test_apply_reparam_inplace_clip_boundary() -> None:
    """reparam_inplace at the clip box edge (decay=1.0) is admitted."""
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    co = {
        "step": 1,
        "op": "reparam_inplace",
        "family_id": "f",
        "target": {"layer": "L0"},
        "args": {"decay": 1.0},
    }
    new_net, status, err = ref.apply_changeop(net, co)
    assert status == "continue"
    assert new_net is not None
    assert new_net.blocks[0].decay == 1.0
    assert err is None


def test_apply_reparam_out_of_clip_errors() -> None:
    """mix=1.1 (out of [-1,1]) returns error with explanation."""
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    co = {
        "step": 1,
        "op": "reparam_inplace",
        "family_id": "f",
        "target": {"layer": "L0"},
        "args": {"mix": 1.1},
    }
    new_net, status, err = ref.apply_changeop(net, co)
    assert status == "error"
    assert new_net is None
    assert err is not None
    assert "mix=1.1" in err
    assert "clip" in err


def test_apply_unknown_op_errors() -> None:
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    co = {"step": 1, "op": "no_such_op", "family_id": "f", "target": {}, "args": {}}
    _, status, err = ref.apply_changeop(net, co)
    assert status == "error"
    assert err is not None
    assert "unknown op" in err


def test_step_once_unsat_emits_witness(tmp_path: Path) -> None:
    """An unsat step writes a parseable .smt2 witness with (check-sat)."""
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    co = {
        "step": 1,
        "op": "reparam_inplace",
        "family_id": "f",
        "target": {"layer": "L0"},
        "args": {"decay": 0.7},
    }
    new_net, verdict = ref.step_once(net, co, tmp_path, step_id=1)
    assert verdict.verdict == "unsat"
    assert verdict.witness_path is not None
    wp = Path(verdict.witness_path)
    assert wp.exists()
    text = wp.read_text(encoding="utf-8")
    assert "(set-logic QF_NRA)" in text
    assert "(check-sat)" in text
    assert "expected: unsat" in text


def test_unsat_witness_is_independently_verifiable(tmp_path: Path) -> None:
    """Run stand-alone z3 on the witness file; it should reply 'unsat'.

    This is the soundness audit path the judge uses.
    """
    z3 = pytest.importorskip("z3")
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.7, mix=0.3, gate_str=0.4)])
    wp = ref.write_unsat_witness(net, tmp_path, step=1)
    text = wp.read_text(encoding="utf-8")
    # Parse as SMT-LIB string and check via Z3.
    solver = z3.Solver()
    solver.from_string(text)
    result = solver.check()
    assert str(result) == "unsat", f"witness re-check failed: {result}"


def test_family_soundness_unknown_family_unsupported() -> None:
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    verdict, detail = ref.query_family_soundness(net, "f_never_declared")
    assert verdict == "unsupported"
    assert "unknown family_id" in detail


def test_insert_subblock_rwkv_extends_chain() -> None:
    """insert_subblock with RwkvTimeMix extends the chain length."""
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    co = {
        "step": 1,
        "op": "insert_subblock",
        "family_id": "f",
        "target": {"after_layer": "L0"},
        "args": {"kernel": "RwkvTimeMix", "init_params": {"decay": 0.4, "mix": 0.2, "gate_str": 0.3}},
    }
    new_net, status, err = ref.apply_changeop(net, co)
    assert status == "continue"
    assert new_net is not None
    assert len(new_net.blocks) == 2
    assert new_net.blocks[1].decay == 0.4


def test_chain_verification_admits_two_safe_blocks(tmp_path: Path) -> None:
    """Two safe RwkvTimeMix blocks in a chain are both admitted."""
    net = ref.NetworkState(blocks=[
        StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4),
        StateUpdateGene(decay=0.4, mix=0.2, gate_str=0.3),
    ])
    r = ref.verify_net(net)
    assert r.ok


def test_self_test_main_returns_zero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The script's main entry point (no args = self-test) exits 0."""
    monkeypatch.setattr(sys, "argv", ["poc_7a_vnn_comp_reference_impl.py", "--self-test"])
    rc = ref.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "verdict: PASS" in out


def test_serve_protocol_smoke(tmp_path: Path) -> None:
    """Run the --serve path end-to-end via subprocess for a 2-step session.

    Creates a JSON model + invariant dummy, feeds READY/INIT/STEP/STEP/END via
    stdin, and checks the verdicts.
    """
    model_path = tmp_path / "model.json"
    inv_path = tmp_path / "invariant.json"
    model_path.write_text(json.dumps({"blocks": [{"decay": 0.5, "mix": 0.3, "gate_str": 0.4}]}), encoding="utf-8")
    inv_path.write_text(json.dumps({"state_bound": 1.0, "max_input_abs": 1.0}), encoding="utf-8")

    script = _PROJ_ROOT / "scripts" / "poc_7a_vnn_comp_reference_impl.py"
    stdin_text = (
        "READY\n"
        f"INIT {model_path} {inv_path}\n"
        'STEP {"step":1,"op":"reparam_inplace","family_id":"f","target":{"layer":"L0"},"args":{"decay":0.7}}\n'
        'STEP {"step":2,"op":"reparam_inplace","family_id":"f","target":{"layer":"L0"},"args":{"mix":1.1}}\n'
        "END\n"
    )
    proc = subprocess.run(
        [sys.executable, str(script), "--serve"],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    # Expected: OK <ver> <kernels>, OK <init_ms>, unsat <t> <w>, error <t>..., OK <total> 2
    assert lines[0].startswith("OK 0.1")
    assert lines[1].startswith("OK ")
    assert lines[2].startswith("unsat ")
    assert lines[3].startswith("error ")
    assert lines[-1].startswith("OK ") and lines[-1].endswith(" 2")
