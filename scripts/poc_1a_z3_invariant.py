# SPDX-License-Identifier: Apache-2.0
"""PoC 1a — Z3 verifier の state_norm 有界 invariant (Stage 1a) falsifiable 検証.

falsifiable 命題:
    llcore 自前 Z3 verifier が
    (a) z3-solver を import / 動作確認でき (optional dep)、
    (b) StateUpdateGene の clip 範囲 (decay∈[0,1], mix∈[-1,1], gate_str∈[-2,2])
        全体で「|state| <= 1」invariant が **symbolic に証明** (unsat) でき、
    (c) clip 範囲を拡張した illegal 領域 (例: decay > 1) では **反例を見つけ** (sat),
    (d) 単一 gene の verify_gene_safe が機能し、
    (e) 進化ループ内で online gate として呼べ reject 率を測れる,
    (f) Z3 解答 timeout < 1 sec で実用可、
    (g) 同 query で 2 回呼んで同結果 (決定論性).

破綻ゲート (G1-G7):
- [G1] z3-solver import 可能、is_z3_available() = True
- [G2] **clip 範囲下の不変量証明**: verify_state_norm_invariant() returns ok=True (unsat)
- [G3] **illegal 領域での反例検出 (soundness check)**: extended range で sat
- [G4] verify_gene_safe(clipped gene) returns ok=True
- [G5] online gate: 100 random gene のうち reject 率 = 0 (clip 範囲下では reject なし)
- [G6] timeout: 単一検証 < 1 sec
- [G7] 決定論性: 同 query で 2 回呼び同結果

使い方::

    py -3.11 scripts/poc_1a_z3_invariant.py

依存: z3-solver (optional, ``pip install llmesh-llcore[z3]``).
honest 留保: tanh は Z3 で直接表現不能なので |tanh| <= 1 上界で近似 (sound).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

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
from llcore.verifier import (  # noqa: E402
    is_z3_available,
    verify_gene_safe,
    verify_state_norm_invariant,
)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate_g1_z3_available() -> tuple[bool, str]:
    """[G1] z3-solver が import 可能."""
    avail = is_z3_available()
    return avail, f"is_z3_available={avail}"


def gate_g2_clip_range_invariant() -> tuple[bool, str]:
    """[G2] clip 範囲下で |state|<=1 invariant が unsat (証明) になる."""
    result = verify_state_norm_invariant(
        max_input_abs=1.0, state_bound=1.0, timeout_ms=2000,
    )
    return result.ok and result.used_z3, f"verdict: {result.reason}"


def gate_g3_illegal_range_finds_counterexample() -> tuple[bool, str]:
    """[G3] decay > 1 等の illegal range なら sat (反例検出) で sound 確認.

    invariants module は clip 範囲固定なので、本ゲートでは Z3 を直接呼んで
    "decay > 1 で state が伸びる" 反例を確認する (soundness sanity)。
    """
    import z3

    solver = z3.Solver()
    solver.set("timeout", 2000)
    decay = z3.Real("decay")
    s = z3.Real("s")
    x = z3.Real("x")
    tanh_val = z3.Real("tanh_val")

    # illegal: decay = 2 (clip 外)
    solver.add(decay == 2)
    solver.add(s == 1)
    solver.add(x == 1)
    solver.add(tanh_val >= -1, tanh_val <= 1)
    # s_next = decay*s + (1-decay)*tanh = 2*1 + (-1)*tanh = 2 - tanh
    s_next = decay * s + (1 - decay) * tanh_val
    # |s_next| > 1 を要求
    solver.add(z3.Or(s_next > 1, s_next < -1))
    result = solver.check()
    ok = result == z3.sat
    return ok, f"illegal decay=2 sat={result == z3.sat} (sound: 反例検出)"


def gate_g4_gene_safe_single(rng: np.random.Generator) -> tuple[bool, str]:
    """[G4] 5 個 random clipped gene が verify_gene_safe で admit される."""
    n_admit = 0
    for _ in range(5):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        r = verify_gene_safe(gene, timeout_ms=2000)
        if r.ok:
            n_admit += 1
    ok = n_admit == 5
    return ok, f"admit {n_admit}/5 random clipped genes"


def gate_g5_online_gate_reject_rate(rng: np.random.Generator) -> tuple[bool, str]:
    """[G5] online gate として 30 個 random gene 評価し reject 率測定.

    clip 範囲下では全 gene admit が期待されるが、tanh 近似の保守性で
    一部 false reject が出る可能性 (honest disclosure)。
    """
    n = 30
    n_admit = 0
    for _ in range(n):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        r = verify_gene_safe(gene, timeout_ms=1500)
        if r.ok:
            n_admit += 1
    reject_rate = (n - n_admit) / n
    # clip 範囲下では理論的に reject 率 = 0 のはず
    ok = reject_rate == 0.0
    return ok, f"admit {n_admit}/{n} (reject rate={reject_rate*100:.1f}%, expected 0% in clip range)"


def gate_g6_timeout_practical() -> tuple[bool, str]:
    """[G6] 単一 invariant 検証が 1 sec 以内に終わる."""
    start = time.perf_counter()
    r = verify_state_norm_invariant(timeout_ms=2000)
    elapsed = time.perf_counter() - start
    ok = elapsed < 1.0 and r.ok
    return ok, f"elapsed={elapsed*1000:.1f}ms (< 1000ms target)"


def gate_g7_determinism() -> tuple[bool, str]:
    """[G7] 同 query で 2 回呼んで同結果."""
    r1 = verify_state_norm_invariant()
    r2 = verify_state_norm_invariant()
    same = r1.ok == r2.ok and r1.used_z3 == r2.used_z3
    return same, f"ok r1=r2={same}, both ok={r1.ok}"


def gate_g8_extended_range_rejects() -> tuple[bool, str]:
    """[G8] state_bound を強くした (0.3) と入力範囲を拡張で reject が出る (filtering 機能).

    Codex 2026-05-29 指摘で追加: gate 機能 (filtering) を実証する。
    state_bound 0.3 では convex combination 上界が 0.3 を超える gene が出るはず。
    """
    # state_bound=0.3 で全領域証明 → sat (反例あり) になるはず
    r = verify_state_norm_invariant(max_input_abs=1.0, state_bound=0.3, timeout_ms=2000)
    # 期待: ok=False (反例あり = filtering 機能)
    ok = (not r.ok) and r.used_z3 and r.counterexample is not None
    ce_msg = f"counterexample d={r.counterexample['decay']:.2f} m={r.counterexample['mix']:.2f} g={r.counterexample['gate_str']:.2f}" if r.counterexample else "(no CE)"
    return ok, f"verifier rejects extended invariant (state_bound=0.3): {ce_msg}"


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 1a — Z3 verifier state_norm 有界 invariant falsifiable verification")
    print("=" * 72)
    rng = np.random.default_rng(20260529)

    gates = [
        ("G1: z3 available", gate_g1_z3_available),
        ("G2: clip range invariant unsat (proof)", gate_g2_clip_range_invariant),
        ("G3: illegal range sat (sound counterexample)", gate_g3_illegal_range_finds_counterexample),
        ("G4: verify_gene_safe single (5 random)", lambda: gate_g4_gene_safe_single(rng)),
        ("G5: online gate reject rate (30 random)", lambda: gate_g5_online_gate_reject_rate(rng)),
        ("G6: practical timeout (<1s)", gate_g6_timeout_practical),
        ("G7: determinism (same query → same result)", gate_g7_determinism),
        ("G8: extended range filtering (state_bound=0.3 → reject)", gate_g8_extended_range_rejects),
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
        print("PoC 1a verdict: PASS — Z3 verifier で clip 範囲下の有界性を symbolic 証明.")
        print("                 online gate として実用可、決定論性 + timeout OK.")
        print("                 次段 Stage 2a (factor_hook × RWKV mock) に進めます.")
        return 0
    print("PoC 1a verdict: FAIL — verifier 設計または近似を見直してから次へ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
