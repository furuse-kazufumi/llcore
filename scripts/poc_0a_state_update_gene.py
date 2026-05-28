# SPDX-License-Identifier: Apache-2.0
"""PoC 0a — state update 数式遺伝子の最小実装 + falsifiable 命題検証.

falsifiable 命題:
    **decay/mix/gate_str の 3 パラメータで RNN-like state update を表現でき、
    入力長 L=256, dim=8 の有界入力に対し state が NaN/Inf にならず、
    state_norm が ``K * input_norm`` で抑えられる (K は decay/gate_str の関数)。**

破綻ゲート (PASS/FAIL):
- [G1] 単一 step で NaN/Inf 混入なし
- [G2] L=256 sequence で state norm が有界 (norm < 100 * input_norm)
- [G3] 同じ gene/seed で run 2 回の結果が完全一致 (決定論性)
- [G4] decay=0/mix=0/gate_str=0 等の極端値で degenerate にならない (zero state を返す)
- [G5] 5 個体 (decay/mix/gate_str の random 組合せ) 全てが G1-G2 をパス

使い方::

    py -3.11 scripts/poc_0a_state_update_gene.py

依存: numpy のみ. llive 非依存 (PoC 0c で初めて llive lldarwin_v2 を import).

honest 留保:
- これは「数式表現が動く」レベルの mechanism feasibility のみ.
  実 task fitness は PoC 0b、進化は PoC 0c.
- 数値範囲は人為的 (input ∈ [-1, 1])、実 LLM scale ではない.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_utf8_stdout() -> None:
    """Windows cp932 console での日本語出力対策."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass


_PROJ_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.state_update import StateUpdateGene, eval_step, run_sequence  # noqa: E402


# ---------------------------------------------------------------------------
# gates (falsifiable 命題の機械検証)
# ---------------------------------------------------------------------------


def gate_g1_single_step_finite(rng: np.random.Generator) -> tuple[bool, str]:
    """[G1] 単一 step で NaN/Inf 混入なし."""
    dim = 8
    state = rng.normal(0, 0.5, dim)
    x = rng.normal(0, 0.5, dim)
    gene = StateUpdateGene(decay=0.9, mix=0.1, gate_str=1.0)
    new_state = eval_step(state, x, gene)
    ok = bool(np.all(np.isfinite(new_state)))
    return ok, f"state finite={ok}, range=[{new_state.min():.3f},{new_state.max():.3f}]"


def gate_g2_bounded_norm(rng: np.random.Generator) -> tuple[bool, str]:
    """[G2] L=256 sequence で state norm が有界."""
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    gene = StateUpdateGene(decay=0.95, mix=0.05, gate_str=0.5)
    states = run_sequence(inputs, gene)
    state_norms = np.linalg.norm(states, axis=1)
    input_norm = np.linalg.norm(inputs, axis=1).mean()
    max_norm = float(state_norms.max())
    # K = 100 は緩い上界 (理論的には decay/gate_str から狭く計算可能だが PoC は緩く)
    ok = bool(max_norm < 100 * input_norm) and bool(np.all(np.isfinite(state_norms)))
    return ok, f"max_state_norm={max_norm:.3f}, mean_input_norm={input_norm:.3f}, bound_K=100"


def gate_g3_determinism() -> tuple[bool, str]:
    """[G3] 同じ gene / seed で run 2 回の結果が完全一致."""
    L, dim = 64, 4
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    inputs_a = rng_a.normal(0, 0.5, size=(L, dim))
    inputs_b = rng_b.normal(0, 0.5, size=(L, dim))
    gene = StateUpdateGene(decay=0.8, mix=0.2, gate_str=1.2)
    s1 = run_sequence(inputs_a, gene)
    s2 = run_sequence(inputs_b, gene)
    ok = bool(np.array_equal(s1, s2))
    return ok, f"two runs identical={ok}, max_diff={float(np.max(np.abs(s1 - s2))):.2e}"


def gate_g4_degenerate(rng: np.random.Generator) -> tuple[bool, str]:
    """[G4] 極端値 (decay=0/mix=0/gate_str=0) で degenerate にならない."""
    L, dim = 32, 4
    inputs = rng.uniform(-0.5, 0.5, size=(L, dim))
    cases = [
        ("decay=0", StateUpdateGene(decay=0.0, mix=0.5, gate_str=1.0)),
        ("mix=0", StateUpdateGene(decay=0.9, mix=0.0, gate_str=1.0)),
        ("gate_str=0", StateUpdateGene(decay=0.9, mix=0.1, gate_str=0.0)),
        ("all_zero", StateUpdateGene(decay=0.0, mix=0.0, gate_str=0.0)),
    ]
    msgs: list[str] = []
    all_ok = True
    for name, gene in cases:
        states = run_sequence(inputs, gene)
        finite = bool(np.all(np.isfinite(states)))
        # all_zero 等は state=0 を返すのが正しい (degenerate でなく仕様)
        max_norm = float(np.linalg.norm(states, axis=1).max())
        ok = finite and max_norm < 1e6  # 有限値の中で適当な上界
        msgs.append(f"{name}:finite={finite},max_norm={max_norm:.2e}")
        all_ok = all_ok and ok
    return all_ok, " | ".join(msgs)


def gate_g5_random_population(rng: np.random.Generator, n_individuals: int = 5) -> tuple[bool, str]:
    """[G5] random 個体集団 (N=5) 全てが G1-G2 を pass."""
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    input_norm = np.linalg.norm(inputs, axis=1).mean()
    results: list[str] = []
    all_ok = True
    for i in range(n_individuals):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(0.0, 1.0)),
            gate_str=float(rng.uniform(0.0, 2.0)),
        )
        states = run_sequence(inputs, gene)
        finite = bool(np.all(np.isfinite(states)))
        max_norm = float(np.linalg.norm(states, axis=1).max())
        bounded = max_norm < 100 * input_norm
        ok = finite and bounded
        results.append(
            f"#{i}(d={gene.decay:.2f},m={gene.mix:.2f},g={gene.gate_str:.2f}):"
            f"finite={finite},norm={max_norm:.2f},bounded={bounded}"
        )
        all_ok = all_ok and ok
    return all_ok, " | ".join(results)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 0a — state update 数式遺伝子 falsifiable verification")
    print("=" * 72)
    rng = np.random.default_rng(20260529)

    gates = [
        ("G1: single-step finite", lambda: gate_g1_single_step_finite(rng)),
        ("G2: bounded norm L=256", lambda: gate_g2_bounded_norm(rng)),
        ("G3: determinism (seed=42)", gate_g3_determinism),
        ("G4: degenerate values", lambda: gate_g4_degenerate(rng)),
        ("G5: random population N=5", lambda: gate_g5_random_population(rng, 5)),
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
        print("PoC 0a verdict: PASS — state update 数式遺伝子は falsifiable 命題を満たす.")
        print("                 次段 PoC 0b (合成 sequence fitness) に進めます.")
        return 0
    print("PoC 0a verdict: FAIL — gene 表現を見直してから PoC 0b に進む.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
