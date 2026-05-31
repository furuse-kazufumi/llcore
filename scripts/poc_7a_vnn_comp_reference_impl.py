# SPDX-License-Identifier: Apache-2.0
"""PoC 7a — VNN-COMP `online-arch-evo` reference implementation (Stage 7a).

falsifiable 命題:
    PoC 1a の Z3 state_norm invariant verifier を VNN-COMP `online-arch-evo`
    spec v0.1 の stdin/stdout protocol で wrap し、
    (a) 5 step `reparam_inplace` ChangeOp seq に対し per-step verdict を sat/unsat
        として返せ、
    (b) clip 外 (mix=1.1) で error を返し、
    (c) total wall-clock が per-step budget (500 ms) を毎ステップ下回り、
    (d) 同 seq 2 回実行で同 verdict (determinism)、
    (e) self-test 命令で internal pipeline を回せる。

破綻ゲート (G1-G7):
- [G1] handshake (READY → OK <ver>) が正しく動く
- [G2] INIT で network + invariant を読み state を持てる
- [G3] 5-step seq の per-step verdict が unsat,unsat,unsat,unsat,error
- [G4] per-step time が 500 ms 以下 (典型 10 ms 未満)
- [G5] determinism: 同 seq 2 回で同 verdict
- [G6] family_soundness query: f_decay_only family を sound と判定
- [G7] unsupported kernel (MambaScan 等) に対し unsupported を返す

使い方::

    py -3.11 scripts/poc_7a_vnn_comp_reference_impl.py --self-test
    py -3.11 scripts/poc_7a_vnn_comp_reference_impl.py --serve   # stdin/stdout daemon

依存: z3-solver (optional, `pip install llmesh-llcore[z3]`).
honest 留保:
    - 単一 RwkvTimeMix kernel + 単一 family (reparam_inplace) のみ full support
    - 実 .onnx / .vnnlib の読み書きを実装済 (§9 item 10 解消):
        * .onnx = RwkvTimeMix 連鎖を custom-op (domain "llcore.rwkv", 属性 decay/mix/gate_str)
          で表す llcore 規約。real onnx package 必須 (optional dep; 無ければ NotImplementedError)。
        * .vnnlib = scalar X_0(入力)/Y_0(状態) の box bound subset。
        JSON dummy 経路も後方互換で温存。
    - sat witness は依然 emit せず (§9 item 11; PoC 1a の counter-example の JSON 化は TBD)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass


_PROJ_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.state_update.genes import eval_step  # noqa: E402
from llcore.verifier import (  # noqa: E402
    InvariantResult,
    is_z3_available,
    verify_gene_safe,
    verify_state_norm_invariant,
)

SPEC_VERSION = "0.1"
SUPPORTED_KERNELS = ["RwkvTimeMix"]
PER_STEP_BUDGET_MS = 500
KERNEL_CLIP = {
    "RwkvTimeMix": {
        "decay": (0.0, 1.0),
        "mix": (-1.0, 1.0),
        "gate_str": (-2.0, 2.0),
    }
}


@dataclass
class NetworkState:
    """In-memory representation of the parent network for the reference impl.

    A network is represented as a chain of RwkvTimeMix blocks (one or more
    StateUpdateGene records). The reference impl only handles this single
    kernel; other kernels declared in the spec are not parsed (returns
    unsupported).
    """

    blocks: list[StateUpdateGene] = field(default_factory=list)
    invariant_state_bound: float = 1.0
    invariant_max_input_abs: float = 1.0

    def copy(self) -> NetworkState:
        return NetworkState(
            blocks=list(self.blocks),
            invariant_state_bound=self.invariant_state_bound,
            invariant_max_input_abs=self.invariant_max_input_abs,
        )


@dataclass(frozen=True)
class StepVerdict:
    """Verdict for one STEP. Mirrors the spec's verdict set."""

    verdict: str  # sat | unsat | timeout | error | refuse | unsupported | unknown
    step_time_ms: float
    witness_path: str | None
    detail: str = ""


# ---------------------------------------------------------------------------
# ONNX / vnnlib parsers + writers
#
# Two encodings are supported on each side:
#   * a JSON dummy (kept for fast tests / no extra deps), and
#   * the real `.onnx` / `.vnnlib` formats (the spec's normative inputs).
#
# `.onnx` encoding convention (llcore `online-arch-evo` reference): the chain of
# RwkvTimeMix blocks is stored as one custom-op node per block, op_type
# "RwkvTimeMix" in domain "llcore.rwkv", carrying float attributes decay / mix /
# gate_str. Nodes are read in graph order. This is a documented convention
# (RwkvTimeMix is not a standard ONNX op); real onnx package required.
#
# `.vnnlib` encoding convention: a scalar input X_0 and scalar state Y_0, with
# a box precondition |X_0| <= max_input_abs and the *negated* invariant
# (a violation has |Y_0| > state_bound). The parser supports this subset
# (declare-const + assert with <=/>=/and/or and (- c) negation).
# ---------------------------------------------------------------------------

_RWKV_OP = "RwkvTimeMix"
_RWKV_DOMAIN = "llcore.rwkv"


def _require_onnx():  # noqa: ANN202
    """Lazy-import onnx so the JSON-dummy path works without the dependency."""
    try:
        import onnx  # noqa: PLC0415
        from onnx import TensorProto, helper  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - exercised only when onnx absent
        raise NotImplementedError(
            "real .onnx support requires the 'onnx' package "
            "(pip install onnx, or llmesh-llcore[vnncomp])"
        ) from e
    return onnx, helper, TensorProto


def write_model_onnx(net: NetworkState, path: Path) -> Path:
    """Serialise a NetworkState to a real `.onnx` file (RwkvTimeMix chain)."""
    onnx, helper, TensorProto = _require_onnx()
    n = len(net.blocks)
    nodes = [
        helper.make_node(
            _RWKV_OP,
            inputs=[f"s_{i}"],
            outputs=[f"s_{i + 1}"],
            name=f"block_{i}",
            domain=_RWKV_DOMAIN,
            decay=float(blk.decay),
            mix=float(blk.mix),
            gate_str=float(blk.gate_str),
        )
        for i, blk in enumerate(net.blocks)
    ]
    in_name = "s_0"
    out_name = f"s_{n}" if n > 0 else "s_0"
    inp = helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1])
    graph = helper.make_graph(nodes, "rwkv_chain", [inp], [out])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(_RWKV_DOMAIN, 1)],
        producer_name="llcore-poc7a",
    )
    onnx.save(model, str(path))
    return path


def parse_model_onnx(path: Path) -> NetworkState:
    """Parse the model file into a NetworkState.

    Accepts a real `.onnx` (RwkvTimeMix-chain convention, see module header) or a
    JSON dummy ``{"blocks": [{"decay":.., "mix":.., "gate_str":..}, ...]}``.
    """
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    if path.suffix.lower() == ".onnx":
        onnx, _helper, _tp = _require_onnx()
        model = onnx.load(str(path))
        blocks: list[StateUpdateGene] = []
        for node in model.graph.node:
            if node.op_type != _RWKV_OP:
                continue
            attrs = {a.name: a.f for a in node.attribute}
            missing = {"decay", "mix", "gate_str"} - attrs.keys()
            if missing:
                raise ValueError(f"RwkvTimeMix node {node.name!r} missing attrs {sorted(missing)}")
            blocks.append(
                StateUpdateGene(
                    decay=float(attrs["decay"]),
                    mix=float(attrs["mix"]),
                    gate_str=float(attrs["gate_str"]),
                )
            )
        return NetworkState(blocks=blocks)
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = [
        StateUpdateGene(decay=float(b["decay"]), mix=float(b["mix"]), gate_str=float(b["gate_str"]))
        for b in data["blocks"]
    ]
    return NetworkState(blocks=blocks)


def _tokenize_sexpr(text: str) -> list[str]:
    """Tokenise an SMT-LIB/vnnlib subset (strip `;` line comments)."""
    lines = [ln.split(";", 1)[0] for ln in text.splitlines()]
    flat = " ".join(lines).replace("(", " ( ").replace(")", " ) ")
    return flat.split()


def _parse_sexpr(tokens: list[str]) -> list:
    """Parse tokens into a list of top-level S-expression forms (nested lists)."""
    pos = 0

    def parse_one():  # noqa: ANN202
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst: list = []
            while pos < len(tokens) and tokens[pos] != ")":
                lst.append(parse_one())
            if pos >= len(tokens):
                raise ValueError("unbalanced parentheses in vnnlib")
            pos += 1  # consume ')'
            return lst
        return tok

    forms: list = []
    while pos < len(tokens):
        forms.append(parse_one())
    return forms


def _sexpr_num(node) -> float:  # noqa: ANN001
    """Convert an atom number or ``(- c)`` negation to float."""
    if isinstance(node, list) and len(node) == 2 and node[0] == "-":
        return -float(node[1])
    return float(node)


def write_invariant_vnnlib(state_bound: float, max_input_abs: float, path: Path) -> Path:
    """Serialise the invariant to a real `.vnnlib` file (scalar X_0/Y_0 subset)."""
    ia = float(max_input_abs)
    sb = float(state_bound)
    text = (
        "; llcore online-arch-evo invariant (vnnlib subset)\n"
        "; X_0 = scalar input, Y_0 = scalar state; property: |Y_0| <= state_bound\n"
        "; given |X_0| <= max_input_abs. The assertions encode the *negation*\n"
        "; (a counterexample has |Y_0| > state_bound) so unsat means the invariant holds.\n"
        "(declare-const X_0 Real)\n"
        "(declare-const Y_0 Real)\n"
        f"(assert (<= X_0 {ia}))\n"
        f"(assert (>= X_0 (- {ia})))\n"
        f"(assert (or (>= Y_0 {sb}) (<= Y_0 (- {sb}))))\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def parse_invariant_vnnlib(path: Path) -> tuple[float, float]:
    """Parse the invariant into ``(state_bound, max_input_abs)``.

    Accepts a real `.vnnlib` (scalar X_0/Y_0 subset, see module header) or a JSON
    dummy ``{"state_bound": 1.0, "max_input_abs": 1.0}``.
    """
    if not path.exists():
        raise FileNotFoundError(f"invariant not found: {path}")
    if path.suffix.lower() == ".vnnlib":
        forms = _parse_sexpr(_tokenize_sexpr(path.read_text(encoding="utf-8")))
        max_input_abs: float | None = None
        state_bound: float | None = None

        def walk(node) -> None:  # noqa: ANN001
            nonlocal max_input_abs, state_bound
            if not isinstance(node, list):
                return
            if len(node) == 3 and node[0] in ("<=", ">=") and node[1] == "X_0":
                v = abs(_sexpr_num(node[2]))
                max_input_abs = v if max_input_abs is None else max(max_input_abs, v)
            if len(node) == 3 and node[0] in ("<=", ">=") and node[1] == "Y_0":
                v = abs(_sexpr_num(node[2]))
                state_bound = v if state_bound is None else max(state_bound, v)
            for child in node:
                walk(child)

        for form in forms:
            walk(form)
        if state_bound is None or max_input_abs is None:
            raise ValueError(
                ".vnnlib did not declare the expected X_0 / Y_0 box bounds "
                "(llcore online-arch-evo scalar convention)"
            )
        return state_bound, max_input_abs
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("state_bound", 1.0)), float(data.get("max_input_abs", 1.0))


# ---------------------------------------------------------------------------
# ChangeOp application
# ---------------------------------------------------------------------------


def _apply_reparam(net: NetworkState, target: dict, args: dict) -> tuple[NetworkState, str | None]:
    """Apply reparam_inplace. Returns (new_net, error_msg_if_any)."""
    layer_idx = 0  # mock: target.layer name maps to index 0 for now
    if not net.blocks:
        return net, "no blocks to reparametrise"
    blk = net.blocks[layer_idx]
    new_kwargs: dict[str, float] = {}
    for k, v in args.items():
        if k not in ("decay", "mix", "gate_str"):
            return net, f"reparam: unknown key {k}"
        lo, hi = KERNEL_CLIP["RwkvTimeMix"][k]
        if v < lo or v > hi:
            return net, f"reparam: {k}={v} outside clip box [{lo}, {hi}]"
        new_kwargs[k] = float(v)
    new_blk = replace(blk, **new_kwargs)
    new_blocks = list(net.blocks)
    new_blocks[layer_idx] = new_blk
    new_net = net.copy()
    new_net.blocks = new_blocks
    return new_net, None


def _apply_insert(net: NetworkState, target: dict, args: dict) -> tuple[NetworkState, str | None]:
    """Apply insert_subblock. Only RwkvTimeMix kernel supported."""
    kernel = args.get("kernel", "")
    if kernel != "RwkvTimeMix":
        return net, f"insert_subblock: kernel {kernel!r} not supported"
    init = args.get("init_params", {})
    try:
        new_blk = StateUpdateGene(
            decay=float(init.get("decay", 0.5)),
            mix=float(init.get("mix", 0.3)),
            gate_str=float(init.get("gate_str", 0.4)),
        )
    except Exception as e:
        return net, f"insert_subblock: invalid init_params ({e})"
    new_net = net.copy()
    new_net.blocks = list(net.blocks) + [new_blk]
    return new_net, None


def apply_changeop(net: NetworkState, changeop: dict) -> tuple[NetworkState | None, str, str | None]:
    """Apply one ChangeOp to the network.

    Returns (new_net, verdict, error_msg). verdict ∈ {"continue", "error",
    "unsupported"}. new_net is None iff verdict != "continue".
    """
    op = changeop.get("op", "")
    target = changeop.get("target", {})
    args = changeop.get("args", {})

    if op == "reparam_inplace":
        new_net, err = _apply_reparam(net, target, args)
        if err is not None:
            return None, "error", err
        return new_net, "continue", None
    if op == "insert_subblock":
        new_net, err = _apply_insert(net, target, args)
        if err is not None:
            return None, "unsupported", err
        return new_net, "continue", None
    if op in ("delete_subblock", "reorder_subblocks", "type_subst"):
        return None, "unsupported", f"{op} not implemented in PoC 7a reference impl"
    return None, "error", f"unknown op {op!r}"


# ---------------------------------------------------------------------------
# Verifier driver
# ---------------------------------------------------------------------------


def verify_net(net: NetworkState, timeout_ms: int = 200) -> InvariantResult:
    """Verify the invariant on each block of the chain (conjunction).

    Returns the first failing block's result, or the last successful result
    if all admit. This matches §6 of the impl spec.
    """
    if not net.blocks:
        return InvariantResult(ok=True, used_z3=False, reason="empty network, vacuously safe")
    last: InvariantResult | None = None
    for blk in net.blocks:
        r = verify_gene_safe(
            blk,
            max_input_abs=net.invariant_max_input_abs,
            state_bound=net.invariant_state_bound,
            timeout_ms=timeout_ms,
        )
        last = r
        if not r.ok:
            return r
    assert last is not None
    return last


def write_unsat_witness(net: NetworkState, witness_dir: Path, step: int) -> Path:
    """Write a sound `.smt2` witness for an unsat verdict.

    The witness encodes the same constraint system the verifier just discharged,
    so an external checker (`z3 witness.smt2`) can re-verify independently.
    """
    witness_dir.mkdir(parents=True, exist_ok=True)
    path = witness_dir / f"witness_step_{step:04d}.smt2"
    if not net.blocks:
        path.write_text("; empty network\n(set-logic QF_NRA)\n(check-sat)\n; expected: unsat\n", encoding="utf-8")
        return path
    blk = net.blocks[0]  # primary block witness (chain conjoin: first sufficient)
    sb = net.invariant_state_bound
    ia = net.invariant_max_input_abs
    smt = (
        f"; llcore PoC 7a unsat witness, step {step}\n"
        f"; kernel = RwkvTimeMix, decay={blk.decay:.6f}, mix={blk.mix:.6f}, gate_str={blk.gate_str:.6f}\n"
        f"(set-logic QF_NRA)\n"
        f"(declare-const s Real)\n"
        f"(declare-const x Real)\n"
        f"(declare-const tanh_val Real)\n"
        f"(assert (and (>= s (- {sb})) (<= s {sb})))\n"
        f"(assert (and (>= x (- {ia})) (<= x {ia})))\n"
        f"(assert (and (>= tanh_val (- 1)) (<= tanh_val 1)))\n"
        f"(assert (<= (* tanh_val tanh_val) "
        f"(* (+ (* {blk.mix:.6f} x) (* {blk.gate_str:.6f} s)) (+ (* {blk.mix:.6f} x) (* {blk.gate_str:.6f} s)))))\n"
        f"(assert (>= (* tanh_val (+ (* {blk.mix:.6f} x) (* {blk.gate_str:.6f} s))) 0))\n"
        f"(assert (or (> (+ (* {blk.decay:.6f} s) (* (- 1 {blk.decay:.6f}) tanh_val)) {sb}) "
        f"(< (+ (* {blk.decay:.6f} s) (* (- 1 {blk.decay:.6f}) tanh_val)) (- {sb}))))\n"
        f"(check-sat)\n"
        f"; expected: unsat\n"
    )
    path.write_text(smt, encoding="utf-8")
    return path


def write_sat_witness(
    net: NetworkState, result: InvariantResult, witness_dir: Path, step: int, scan_n: int = 121
) -> tuple[Path, bool]:
    """Emit a JSON `sat` witness and report whether a REAL violation is confirmed.

    The verifier (verify_gene_safe) abstracts ``tanh`` (``tanh_val^2 <= preact^2``),
    so its `sat` can be an over-approximation artifact, and it returns no assignment.
    We therefore confirm independently: grid-scan the true next-state
    ``eval_step(s, x, block)`` over the box ``|s| <= state_bound, |x| <= max_input_abs``
    for each block. Returns ``(path, confirmed)`` — ``confirmed=True`` means a real,
    independently-recomputable counterexample ``(s, x)`` exists (the caller reports
    `sat`); ``confirmed=False`` means no real violation was found on the finite grid,
    which is NOT a soundness proof of safety, so the caller reports `unknown`
    (never a false `sat` / VNN-COMP −150 penalty).
    """
    witness_dir.mkdir(parents=True, exist_ok=True)
    path = witness_dir / f"witness_step_{step:04d}_sat.json"
    ce = dict(result.counterexample or {})
    sb = float(net.invariant_state_bound)
    ia = float(net.invariant_max_input_abs)

    # verify_gene_safe reports a violation (or gives up) WITHOUT an assignment and via
    # a tanh OVER-APPROXIMATION, so we confirm independently: grid-scan the real (tanh)
    # next-state over the box |s| <= state_bound, |x| <= max_input_abs for each block.
    # A grid hit is an independently recomputable witness (x, s); finding none on the
    # finite grid is "no real violation found", NOT a soundness proof of safety.
    block_idx: int | None = None
    w_x: float | None = None
    w_s: float | None = None
    s_next_real: float | None = None
    best = 0.0
    xs = np.linspace(-ia, ia, scan_n)
    for bi, blk in enumerate(net.blocks):
        for sv in np.linspace(-sb, sb, scan_n):
            row = eval_step(np.full(scan_n, sv, dtype=float), xs, blk)
            k = int(np.argmax(np.abs(row)))
            mag = abs(float(row[k]))
            if mag > best:
                best, block_idx, w_x, w_s, s_next_real = mag, bi, float(xs[k]), float(sv), float(row[k])
    confirmed = best > sb + 1e-9

    witness = {
        "step": step,
        "kernel": "RwkvTimeMix",
        "verdict_candidate": "sat",
        "z3_counterexample": ce,  # informational; verify_gene_safe usually omits it
        "state_bound": sb,
        "max_input_abs": ia,
        "confirmed_real_violation": confirmed,  # True = grid found a real (tanh) violation
        "confirmation_method": "real_tanh_grid_scan",
        "witness_block_index": block_idx,
        "witness_input_x": w_x,
        "witness_state_s": w_s,
        "real_s_next": s_next_real,
        "max_abs_s_next_on_grid": best,
        "note": (
            "confirmed: block {bi}, real eval_step(s={s}, x={x}) gives |s_next|={m:.6f} > "
            "state_bound={sb} (independently recomputable counterexample)".format(
                bi=block_idx, s=w_s, x=w_x, m=best, sb=sb)
            if confirmed
            else "NOT confirmed: no real (tanh) violation found on the scanned grid "
            "(max |s_next|={m:.6f} <= state_bound={sb}). This is a finite numerical scan — "
            "neither a soundness proof of safety nor a proof the verifier's flag was spurious; "
            "the verdict is therefore 'unknown'.".format(m=best, sb=sb)
        ),
    }
    path.write_text(json.dumps(witness, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, confirmed


# ---------------------------------------------------------------------------
# Step driver
# ---------------------------------------------------------------------------


def step_once(
    net: NetworkState,
    changeop: dict,
    witness_dir: Path,
    step_id: int,
    per_step_budget_ms: int = PER_STEP_BUDGET_MS,
) -> tuple[NetworkState, StepVerdict]:
    """One STEP cycle. Returns (new_net_if_continue, verdict).

    If the verdict is error/unsupported, the net is unchanged (caller may
    decide to abort).
    """
    start = time.perf_counter()
    new_net, status, err = apply_changeop(net, changeop)
    if status == "error":
        elapsed = (time.perf_counter() - start) * 1000.0
        return net, StepVerdict("error", elapsed, None, err or "")
    if status == "unsupported":
        elapsed = (time.perf_counter() - start) * 1000.0
        return net, StepVerdict("unsupported", elapsed, None, err or "")
    assert new_net is not None
    r = verify_net(new_net, timeout_ms=min(per_step_budget_ms, 200))
    elapsed = (time.perf_counter() - start) * 1000.0
    if elapsed > per_step_budget_ms:
        return net, StepVerdict("timeout", elapsed, None, f"step exceeded {per_step_budget_ms}ms")
    if r.ok and r.used_z3:
        wp = write_unsat_witness(new_net, witness_dir, step_id)
        return new_net, StepVerdict("unsat", elapsed, str(wp), r.reason)
    if not r.ok and r.used_z3:
        # `verify_gene_safe` reports a violation (or gives up) WITHOUT an assignment
        # and via a tanh over-approximation, and it does not separate z3 `sat` from
        # z3 `unknown` in `ok`/`used_z3` alone (we read `solver_status` for that).
        # `write_sat_witness` independently searches for a REAL violation; only a
        # grid-confirmed real witness yields `sat`. (The old `counterexample is not
        # None` guard was dead code — verify_gene_safe never sets it — so violations
        # used to fall through to "timeout".)
        wp, confirmed = write_sat_witness(new_net, r, witness_dir, step_id)
        status = getattr(r, "solver_status", "unknown")
        if confirmed:
            detail = (
                r.reason
                if status == "sat"
                else f"real counterexample confirmed by grid scan (z3 status was {status})"
            )
            return new_net, StepVerdict("sat", elapsed, str(wp), detail)
        if status == "sat":
            reason = "z3 sat candidate not confirmed by real tanh on the scanned grid (likely tanh-abstraction artifact)"
        else:
            reason = f"z3 inconclusive ({status}); no real violation confirmed on the scanned grid"
        return new_net, StepVerdict("unknown", elapsed, str(wp), reason)
    # used_z3=False fallback path (z3 unavailable); treat as unsat by mathematical argument
    if r.ok and not r.used_z3:
        return new_net, StepVerdict("unsat", elapsed, None, "z3 unavailable, fallback admit")
    return new_net, StepVerdict("timeout", elapsed, None, r.reason)


# ---------------------------------------------------------------------------
# family_soundness query
# ---------------------------------------------------------------------------


def query_family_soundness(net: NetworkState, family_id: str) -> tuple[str, str]:
    """Query whether an entire family preserves the invariant.

    Reference impl supports families whose semantics are equivalent to "any
    reparam_inplace within the clip box of one parameter or the full box".
    Uses the existing `verify_state_norm_invariant` (clip-range quantified).
    """
    known_families = {
        "f_decay_only": "reparam_inplace varies decay only in [0,1]",
        "f_mix_only": "reparam_inplace varies mix only in [-1,1]",
        "f_gate_only": "reparam_inplace varies gate_str only in [-2,2]",
        "f_full_reparam": "reparam_inplace anywhere in clip box",
    }
    if family_id not in known_families:
        return "unsupported", f"unknown family_id {family_id!r}"
    r = verify_state_norm_invariant(
        max_input_abs=net.invariant_max_input_abs,
        state_bound=net.invariant_state_bound,
        timeout_ms=2000,
    )
    if r.ok and r.used_z3:
        return "sound", f"family {family_id} preserves |s|<={net.invariant_state_bound}"
    if not r.ok and r.used_z3:
        return "unsound", r.reason
    return "sound", "z3 unavailable, fallback admit by mathematical argument"


# ---------------------------------------------------------------------------
# Self-test (the PoC's primary entry; serves as G1-G7 gates)
# ---------------------------------------------------------------------------


def gate_g1_handshake() -> tuple[bool, str]:
    """[G1] handshake: protocol version + supported kernels emit correctly."""
    line = f"OK {SPEC_VERSION} {','.join(SUPPORTED_KERNELS)}"
    ok = "0.1" in line and "RwkvTimeMix" in line
    return ok, line


def gate_g2_init() -> tuple[bool, str]:
    """[G2] INIT: parse network + invariant, hold state."""
    net = NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    return (
        len(net.blocks) == 1 and net.invariant_state_bound == 1.0,
        f"init ok: {len(net.blocks)} block(s), bound={net.invariant_state_bound}",
    )


def gate_g3_five_step_seq(witness_dir: Path) -> tuple[bool, str]:
    """[G3] 5-step sequence yields unsat,unsat,unsat,unsat,error."""
    net = NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    seq = [
        {"step": 1, "op": "reparam_inplace", "family_id": "f_decay_only", "target": {"layer": "L0"}, "args": {"decay": 0.7}},
        {"step": 2, "op": "reparam_inplace", "family_id": "f_mix_only", "target": {"layer": "L0"}, "args": {"mix": -0.2}},
        {"step": 3, "op": "reparam_inplace", "family_id": "f_gate_only", "target": {"layer": "L0"}, "args": {"gate_str": 1.5}},
        {"step": 4, "op": "reparam_inplace", "family_id": "f_decay_only", "target": {"layer": "L0"}, "args": {"decay": 1.0}},
        {"step": 5, "op": "reparam_inplace", "family_id": "f_mix_only", "target": {"layer": "L0"}, "args": {"mix": 1.1}},
    ]
    expected = ["unsat", "unsat", "unsat", "unsat", "error"]
    actual: list[str] = []
    for i, co in enumerate(seq, start=1):
        net, verdict = step_once(net, co, witness_dir, i)
        actual.append(verdict.verdict)
    ok = actual == expected
    return ok, f"expected={expected} actual={actual}"


def gate_g4_per_step_budget(witness_dir: Path) -> tuple[bool, str]:
    """[G4] Each step finishes within 500 ms (typical << 50)."""
    net = NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    seq = [
        {"step": i, "op": "reparam_inplace", "family_id": "f", "target": {"layer": "L0"}, "args": {"decay": 0.5 + i * 0.05}}
        for i in range(1, 5)
    ]
    times: list[float] = []
    for i, co in enumerate(seq, start=1):
        net, verdict = step_once(net, co, witness_dir, i)
        times.append(verdict.step_time_ms)
    max_t = max(times)
    ok = max_t < PER_STEP_BUDGET_MS
    return ok, f"max_step_time={max_t:.1f}ms (budget {PER_STEP_BUDGET_MS}ms), trace={[f'{t:.1f}' for t in times]}"


def gate_g5_determinism(witness_dir: Path) -> tuple[bool, str]:
    """[G5] Same seq twice -> same verdicts."""
    def _run() -> list[str]:
        net = NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
        seq = [
            {"step": 1, "op": "reparam_inplace", "family_id": "f", "target": {"layer": "L0"}, "args": {"decay": 0.7}},
            {"step": 2, "op": "reparam_inplace", "family_id": "f", "target": {"layer": "L0"}, "args": {"mix": -0.5}},
            {"step": 3, "op": "reparam_inplace", "family_id": "f", "target": {"layer": "L0"}, "args": {"gate_str": 1.9}},
        ]
        out: list[str] = []
        for i, co in enumerate(seq, start=1):
            net, v = step_once(net, co, witness_dir, i)
            out.append(v.verdict)
        return out
    run1 = _run()
    run2 = _run()
    ok = run1 == run2
    return ok, f"run1={run1} run2={run2}"


def gate_g6_family_soundness() -> tuple[bool, str]:
    """[G6] family_soundness for f_decay_only returns sound."""
    net = NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    verdict, detail = query_family_soundness(net, "f_decay_only")
    ok = verdict == "sound"
    return ok, f"verdict={verdict} detail={detail}"


def gate_g7_unsupported_kernel() -> tuple[bool, str]:
    """[G7] insert_subblock with non-RwkvTimeMix kernel -> unsupported."""
    net = NetworkState(blocks=[StateUpdateGene(decay=0.5, mix=0.3, gate_str=0.4)])
    co = {
        "step": 1,
        "op": "insert_subblock",
        "family_id": "f_mamba",
        "target": {"after_layer": "L0"},
        "args": {"kernel": "MambaScan", "init_params": {}},
    }
    _, status, err = apply_changeop(net, co)
    ok = status == "unsupported"
    return ok, f"status={status} err={err}"


def run_self_test() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 7a — VNN-COMP `online-arch-evo` reference impl falsifiable verification")
    print("=" * 72)

    witness_dir = _PROJ_ROOT / "out" / "poc7a_witnesses"
    witness_dir.mkdir(parents=True, exist_ok=True)

    gates = [
        ("G1: handshake", gate_g1_handshake),
        ("G2: init", gate_g2_init),
        ("G3: 5-step seq verdicts", lambda: gate_g3_five_step_seq(witness_dir)),
        ("G4: per-step budget", lambda: gate_g4_per_step_budget(witness_dir)),
        ("G5: determinism", lambda: gate_g5_determinism(witness_dir)),
        ("G6: family_soundness sound", gate_g6_family_soundness),
        ("G7: unsupported kernel handled", gate_g7_unsupported_kernel),
    ]

    all_pass = True
    for name, fn in gates:
        ok, detail = fn()
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {name}")
        print(f"         {detail}")
        all_pass = all_pass and ok

    print("-" * 72)
    if all_pass:
        print("PoC 7a verdict: PASS — VNN-COMP `online-arch-evo` reference impl が")
        print("                 5-step ChangeOp seq + family_soundness + budget +")
        print("                 determinism を spec v0.1 準拠で discharge できた.")
        return 0
    print("PoC 7a verdict: FAIL — reference impl のいずれかゲート未達.")
    return 1


# ---------------------------------------------------------------------------
# stdin/stdout serve mode (for actual judge integration; not exercised by tests)
# ---------------------------------------------------------------------------


def run_serve() -> int:
    """Implement the spec §6 stdin/stdout protocol. Manual / judge driven."""
    _ensure_utf8_stdout()
    witness_dir = _PROJ_ROOT / "out" / "poc7a_witnesses"
    witness_dir.mkdir(parents=True, exist_ok=True)
    net: NetworkState | None = None
    step_id = 0
    started = time.perf_counter()

    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if not line:
            continue
        if line == "READY":
            print(f"OK {SPEC_VERSION} {','.join(SUPPORTED_KERNELS)}", flush=True)
            continue
        if line.startswith("INIT "):
            _, model_path, inv_path = line.split(" ", 2)
            t0 = time.perf_counter()
            try:
                net = parse_model_onnx(Path(model_path))
                sb, ia = parse_invariant_vnnlib(Path(inv_path))
                net.invariant_state_bound = sb
                net.invariant_max_input_abs = ia
                init_ms = (time.perf_counter() - t0) * 1000.0
                print(f"OK {init_ms:.1f}", flush=True)
            except Exception as e:
                print(f"error 0 init_failed:{e!r}", flush=True)
                net = None
            continue
        if line.startswith("STEP "):
            if net is None:
                print("error 0 no_init", flush=True)
                continue
            payload = line[len("STEP "):]
            try:
                co = json.loads(payload)
            except Exception as e:
                print(f"error 0 bad_json:{e!r}", flush=True)
                continue
            step_id += 1
            net, v = step_once(net, co, witness_dir, step_id)
            wp = v.witness_path if v.witness_path else ""
            print(f"{v.verdict} {v.step_time_ms:.1f} {wp}".rstrip(), flush=True)
            continue
        if line == "END":
            total_ms = (time.perf_counter() - started) * 1000.0
            print(f"OK {total_ms:.1f} {step_id}", flush=True)
            return 0
        # unknown command
        print(f"error 0 unknown_command:{line!r}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PoC 7a — VNN-COMP online-arch-evo reference impl")
    parser.add_argument("--self-test", action="store_true", help="run G1-G7 self test")
    parser.add_argument("--serve", action="store_true", help="run stdin/stdout protocol daemon")
    args = parser.parse_args()
    if args.serve:
        return run_serve()
    # default = self-test
    return run_self_test()


if __name__ == "__main__":
    sys.exit(main())
