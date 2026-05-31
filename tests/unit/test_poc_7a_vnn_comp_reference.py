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


# ---------------------------------------------------------------------------
# Real .onnx / .vnnlib parser + writer round-trips (§9 item 10)
# ---------------------------------------------------------------------------


def test_vnnlib_roundtrip(tmp_path: Path) -> None:
    """write_invariant_vnnlib -> parse_invariant_vnnlib recovers (state_bound, max_input_abs)."""
    p = tmp_path / "inv.vnnlib"
    ref.write_invariant_vnnlib(state_bound=2.0, max_input_abs=1.5, path=p)
    sb, ia = ref.parse_invariant_vnnlib(p)
    assert sb == pytest.approx(2.0)
    assert ia == pytest.approx(1.5)


def test_vnnlib_parse_handwritten(tmp_path: Path) -> None:
    """A hand-written .vnnlib (comments + (- c) negation) parses to the right bounds."""
    p = tmp_path / "inv.vnnlib"
    p.write_text(
        "; comment line\n"
        "(declare-const X_0 Real)\n"
        "(declare-const Y_0 Real)\n"
        "(assert (<= X_0 0.8))\n"
        "(assert (>= X_0 (- 0.8)))\n"
        "(assert (or (>= Y_0 1.25) (<= Y_0 (- 1.25))))\n",
        encoding="utf-8",
    )
    sb, ia = ref.parse_invariant_vnnlib(p)
    assert sb == pytest.approx(1.25)
    assert ia == pytest.approx(0.8)


def test_vnnlib_missing_bounds_raises(tmp_path: Path) -> None:
    """A .vnnlib without the Y_0 state bound is rejected (no silent default)."""
    p = tmp_path / "bad.vnnlib"
    p.write_text("(declare-const X_0 Real)\n(assert (<= X_0 1.0))\n", encoding="utf-8")
    with pytest.raises(ValueError, match="X_0 / Y_0"):
        ref.parse_invariant_vnnlib(p)


def test_vnnlib_matches_json_dummy(tmp_path: Path) -> None:
    """Real .vnnlib and the JSON dummy yield the same (state_bound, max_input_abs)."""
    jp = tmp_path / "inv.json"
    jp.write_text(json.dumps({"state_bound": 1.0, "max_input_abs": 1.0}), encoding="utf-8")
    vp = tmp_path / "inv.vnnlib"
    ref.write_invariant_vnnlib(state_bound=1.0, max_input_abs=1.0, path=vp)
    assert ref.parse_invariant_vnnlib(jp) == ref.parse_invariant_vnnlib(vp)


def test_onnx_roundtrip(tmp_path: Path) -> None:
    """write_model_onnx -> parse_model_onnx recovers the RwkvTimeMix block chain."""
    pytest.importorskip("onnx")
    net = ref.NetworkState(
        blocks=[
            StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4),
            StateUpdateGene(decay=0.9, mix=-0.2, gate_str=1.5),
        ]
    )
    p = tmp_path / "model.onnx"
    ref.write_model_onnx(net, p)
    parsed = ref.parse_model_onnx(p)
    assert len(parsed.blocks) == 2
    for a, b in zip(net.blocks, parsed.blocks):
        assert a.decay == pytest.approx(b.decay)
        assert a.mix == pytest.approx(b.mix)
        assert a.gate_str == pytest.approx(b.gate_str)


def test_onnx_matches_json_dummy(tmp_path: Path) -> None:
    """A network from the JSON dummy, written to .onnx and parsed back, is identical."""
    pytest.importorskip("onnx")
    jp = tmp_path / "model.json"
    jp.write_text(
        json.dumps(
            {"blocks": [
                {"decay": 0.5, "mix": 0.3, "gate_str": 0.4},
                {"decay": 0.7, "mix": -0.1, "gate_str": 1.0},
            ]}
        ),
        encoding="utf-8",
    )
    from_json = ref.parse_model_onnx(jp)
    op = tmp_path / "model.onnx"
    ref.write_model_onnx(from_json, op)
    from_onnx = ref.parse_model_onnx(op)
    assert len(from_json.blocks) == len(from_onnx.blocks) == 2
    for a, b in zip(from_json.blocks, from_onnx.blocks):
        assert (a.decay, a.mix, a.gate_str) == pytest.approx((b.decay, b.mix, b.gate_str))


def test_onnx_parsed_net_verifies(tmp_path: Path) -> None:
    """End-to-end: a network parsed from .onnx feeds the verifier and admits."""
    pytest.importorskip("onnx")
    net = ref.NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    p = tmp_path / "model.onnx"
    ref.write_model_onnx(net, p)
    parsed = ref.parse_model_onnx(p)
    parsed.invariant_state_bound = 1.0
    parsed.invariant_max_input_abs = 1.0
    r = ref.verify_net(parsed)
    assert r.ok


def test_generate_properties_produces_parseable_instances(tmp_path: Path) -> None:
    """The seed-based benchmark generator emits .onnx/.vnnlib that round-trip + verify."""
    pytest.importorskip("onnx")
    gen_dir = _PROJ_ROOT / "scripts" / "vnncomp_benchmark"
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))
    import generate_properties as gen  # noqa: PLC0415

    csv_path = gen.generate(seed=7, out_dir=tmp_path, n_instances=3)
    rows = [r for r in csv_path.read_text(encoding="utf-8").splitlines() if r.strip()]
    assert len(rows) == 3
    for row in rows:
        cols = row.split(",")
        onnx_rel, vnnlib_rel = cols[0], cols[1]
        net = ref.parse_model_onnx(tmp_path / onnx_rel)
        sb, ia = ref.parse_invariant_vnnlib(tmp_path / vnnlib_rel)
        assert len(net.blocks) >= 1
        assert sb > 0.0 and ia > 0.0
        net.invariant_state_bound = sb
        net.invariant_max_input_abs = ia
        # verifier runs to a definite verdict (ok True/False both valid for a benchmark mix)
        assert ref.verify_net(net).used_z3 in (True, False)


def test_sat_genuine_emits_witness(tmp_path: Path) -> None:
    """A genuine invariant violation yields verdict 'sat' + a self-checkable witness."""
    if not ref.is_z3_available():
        pytest.skip("z3 not available")
    # decay=0, mix=1, gate_str=0 -> s_next = tanh(x); max real |s_next| = tanh(1) ~= 0.762
    # state_bound=0.4 < 0.762 so a real violation exists (grid scan confirms it).
    net = ref.NetworkState(
        blocks=[StateUpdateGene(decay=0.0, mix=1.0, gate_str=0.0)],
        invariant_state_bound=0.4,
        invariant_max_input_abs=1.0,
    )
    co = {"step": 1, "op": "reparam_inplace", "family_id": "f", "target": {"layer": "L0"}, "args": {"decay": 0.0}}
    _, v = ref.step_once(net, co, tmp_path, 1)
    assert v.verdict == "sat"
    assert v.witness_path is not None
    w = json.loads(Path(v.witness_path).read_text(encoding="utf-8"))
    assert w["confirmed_real_violation"] is True
    assert w["max_abs_s_next_on_grid"] > w["state_bound"]


def test_sat_spurious_becomes_unknown(tmp_path: Path) -> None:
    """A z3 sat from the tanh over-approximation with no real violation -> 'unknown' (sound)."""
    if not ref.is_z3_available():
        pytest.skip("z3 not available")
    # max real |s_next| = tanh(1) ~= 0.762 < 0.85, so NO real violation exists, but z3's
    # tanh over-approx can pick tanh_val up to 1.0 > 0.85 -> spurious sat. Must report unknown.
    net = ref.NetworkState(
        blocks=[StateUpdateGene(decay=0.0, mix=1.0, gate_str=0.0)],
        invariant_state_bound=0.85,
        invariant_max_input_abs=1.0,
    )
    co = {"step": 1, "op": "reparam_inplace", "family_id": "f", "target": {"layer": "L0"}, "args": {"decay": 0.0}}
    _, v = ref.step_once(net, co, tmp_path, 1)
    assert v.verdict == "unknown"
    assert v.witness_path is not None
    w = json.loads(Path(v.witness_path).read_text(encoding="utf-8"))
    assert w["genuine_violation"] is False


def test_write_sat_witness_recomputes_genuine(tmp_path: Path) -> None:
    """write_sat_witness confirms a genuine z3 candidate by recomputation (unit)."""
    from llcore.verifier import InvariantResult

    net = ref.NetworkState(
        blocks=[StateUpdateGene(decay=0.0, mix=1.0, gate_str=0.0)],
        invariant_state_bound=0.5,
        invariant_max_input_abs=1.0,
    )
    ce = {"decay": 0.0, "mix": 1.0, "gate_str": 0.0, "s": 0.0, "x": 1.0, "tanh_val": 1.0}
    res = InvariantResult(ok=False, used_z3=True, reason="sat", counterexample=ce)
    _path, genuine = ref.write_sat_witness(net, res, tmp_path, 1)
    assert genuine is True  # real tanh(1.0)=0.762 > 0.5


def test_generate_properties_deterministic(tmp_path: Path) -> None:
    """Same seed -> identical instances.csv + .vnnlib content (reproducibility)."""
    pytest.importorskip("onnx")
    gen_dir = _PROJ_ROOT / "scripts" / "vnncomp_benchmark"
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))
    import generate_properties as gen  # noqa: PLC0415

    a, b = tmp_path / "a", tmp_path / "b"
    gen.generate(seed=99, out_dir=a, n_instances=2)
    gen.generate(seed=99, out_dir=b, n_instances=2)
    assert (a / "instances.csv").read_text() == (b / "instances.csv").read_text()
    assert (a / "vnnlib" / "invariant_0.vnnlib").read_text() == (b / "vnnlib" / "invariant_0.vnnlib").read_text()
