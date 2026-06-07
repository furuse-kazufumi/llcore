# SPDX-License-Identifier: Apache-2.0
"""PoC 0a v2 — state update 数式遺伝子 (RWKV-style leak integrator) + falsifiable 命題検証.

履歴:
- v1: ``decay*s + mix*x*tanh(gate_str*s)``
      → state=0 fixed point で zero attractor、G1-G5 形式 PASS だが情報伝達ゼロ。
- v2 (2026-05-29): 2 reviewer (gem-critic + gpt-5.4 codex) 独立 verdict で再設計。
      RWKV-style leak integrator ``decay*s + (1-decay)*tanh(mix*x + gate_str*s)`` 採用。

falsifiable 命題 (v2):
    decay/mix/gate_str の 3 パラメータで RNN-like leak integrator + recurrent
    nonlinear coupling を表現でき、入力長 L=256, dim=8 の有界入力に対し
    (a) state が NaN/Inf にならず、
    (b) state_norm が K * input_norm 以下で抑えられ (K=10 緩い上界),
    (c) 非ゼロ入力で state が非自明 (variance > 0) に動き、
    (d) 異なる入力列で state 軌跡が区別できる。

破綻ゲート (G1-G10, v2 拡張):
- [G1] 単一 step で NaN/Inf 混入なし
- [G2] L=256 sequence で state norm 有界 (norm < 10 * input_norm)
- [G3] 同じ gene/seed で run 2 回の結果が完全一致 (決定論性)
- [G4] 極端値 (decay=0/mix=0/gate_str=0 等) で finite かつ仕様通り
- [G5] random 個体集団 (N=20, v2 で coverage 強化) が全員 G1-G2 をパス
- [G6] **非自明性 (non-trivial activation)**: 非ゼロ入力で state variance > 0
- [G7] **入力区別性 (distinguishability)**: 異なる入力列で state 軌跡が異なる
- [G8] **記憶持続性 (memory persistence)**: decay=0.95 + zero-input で state が即落ちしない
- [G9] **zero-state escape**: state=0 初期、非ゼロ入力で N step 内に norm > eps
- [G10] **parameter sensitivity**: decay/mix/gate_str を動かすと出力軌跡が変わる

使い方::

    py -3.11 scripts/poc_0a_state_update_gene.py

依存: numpy のみ. llive 非依存.
"""
from __future__ import annotations

import sys
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

from llcore.state_update import StateUpdateGene, eval_step, run_sequence  # noqa: E402


# ---------------------------------------------------------------------------
# gates G1-G5 (v1 から継続、ただし K を緩い 10 へ、N を 20 へ拡張)
# ---------------------------------------------------------------------------


def gate_g1_single_step_finite(rng: np.random.Generator) -> tuple[bool, str]:
    """[G1] 単一 step で NaN/Inf 混入なし."""
    dim = 8
    state = rng.normal(0, 0.5, dim)
    x = rng.normal(0, 0.5, dim)
    gene = StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.5)
    new_state = eval_step(state, x, gene)
    ok = bool(np.all(np.isfinite(new_state)))
    return ok, f"state finite={ok}, range=[{new_state.min():.3f},{new_state.max():.3f}]"


def gate_g2_bounded_norm(rng: np.random.Generator) -> tuple[bool, str]:
    """[G2] L=256 sequence で state norm が有界 (緩い上界 K=10)."""
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    gene = StateUpdateGene(decay=0.95, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene)
    state_norms = np.linalg.norm(states, axis=1)
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    max_norm = float(state_norms.max())
    # v2: K=10 (v1 の 100 は緩すぎた)。RWKV-style は convex なので構造的に <= sqrt(dim)
    ok = bool(max_norm < 10 * input_norm) and bool(np.all(np.isfinite(state_norms)))
    return ok, f"max_state_norm={max_norm:.3f}, mean_input_norm={input_norm:.3f}, bound_K=10"


def gate_g3_determinism() -> tuple[bool, str]:
    """[G3] 同じ gene / seed で run 2 回の結果が完全一致."""
    L, dim = 64, 4
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    inputs_a = rng_a.normal(0, 0.5, size=(L, dim))
    inputs_b = rng_b.normal(0, 0.5, size=(L, dim))
    gene = StateUpdateGene(decay=0.8, mix=0.5, gate_str=0.5)
    s1 = run_sequence(inputs_a, gene)
    s2 = run_sequence(inputs_b, gene)
    ok = bool(np.array_equal(s1, s2))
    return ok, f"two runs identical={ok}, max_diff={float(np.max(np.abs(s1 - s2))):.2e}"


def gate_g4_degenerate(rng: np.random.Generator) -> tuple[bool, str]:
    """[G4] 極端値で NaN/Inf 出ず、finite かつ仕様通り.

    v2 redesign: v1 では all_zero ケースで state=0 を「仕様」としていたが、
    これが zero attractor を見逃した直接原因。v2 では各極端値で
    「finite かつ意味のある動作」を要求する (詳細は G6 以降で補完)。
    """
    L, dim = 32, 4
    inputs = rng.uniform(-0.5, 0.5, size=(L, dim))
    cases = [
        ("decay=0", StateUpdateGene(decay=0.0, mix=0.5, gate_str=0.5)),
        ("decay=1", StateUpdateGene(decay=1.0, mix=0.5, gate_str=0.5)),  # 完全記憶
        ("mix=0", StateUpdateGene(decay=0.9, mix=0.0, gate_str=0.5)),
        ("mix_neg", StateUpdateGene(decay=0.9, mix=-0.5, gate_str=0.5)),  # v2 で負許容
        ("gate_str=0", StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0)),
        ("gate_str_neg", StateUpdateGene(decay=0.9, mix=0.5, gate_str=-1.0)),  # 抑制性
    ]
    msgs: list[str] = []
    all_ok = True
    for name, gene in cases:
        states = run_sequence(inputs, gene)
        finite = bool(np.all(np.isfinite(states)))
        max_norm = float(np.linalg.norm(states, axis=1).max())
        ok = finite and max_norm < 1e6
        msgs.append(f"{name}:finite={finite},max_norm={max_norm:.2e}")
        all_ok = all_ok and ok
    return all_ok, " | ".join(msgs)


def gate_g5_random_population(rng: np.random.Generator, n_individuals: int = 20) -> tuple[bool, str]:
    """[G5] random 個体集団 (N=20, v2 で coverage 強化) が全員 G1-G2 を pass."""
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    all_ok = True
    n_finite = 0
    n_bounded = 0
    for _ in range(n_individuals):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        states = run_sequence(inputs, gene)
        finite = bool(np.all(np.isfinite(states)))
        max_norm = float(np.linalg.norm(states, axis=1).max())
        bounded = max_norm < 10 * input_norm
        n_finite += int(finite)
        n_bounded += int(bounded)
        all_ok = all_ok and finite and bounded
    return all_ok, f"N={n_individuals}: finite={n_finite}/{n_individuals}, bounded={n_bounded}/{n_individuals}"


# ---------------------------------------------------------------------------
# v2 追加ゲート G6-G10 (2 reviewer 指摘で必須)
# ---------------------------------------------------------------------------


def gate_g6_nontrivial_activation(rng: np.random.Generator) -> tuple[bool, str]:
    """[G6] 非自明性: 非ゼロ入力で state が非自明 (variance > 0) に動く.

    burn-in 10 step 後の state norm の variance が閾値以上であることを要求。
    zero attractor を機械的に弾く。
    """
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    # decay 中庸 / mix 非ゼロ / gate_str 中庸 = 一般的な「動く」 gene
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene)
    state_norms = np.linalg.norm(states[10:], axis=1)  # burn-in 除外
    mean_norm = float(state_norms.mean())
    var_norm = float(state_norms.var())
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    ok = (mean_norm > 0.01 * input_norm) and (var_norm > 1e-6)
    return ok, f"mean_norm={mean_norm:.4f}, var_norm={var_norm:.4e}, threshold={0.01 * input_norm:.4f}"


def gate_g7_input_distinguishability(rng: np.random.Generator) -> tuple[bool, str]:
    """[G7] 入力区別性: 異なる入力列で state 軌跡が区別可能.

    2 つの独立入力列で最終 state の相対距離 > 0.1 を要求。
    情報伝達能力の最低ライン (PoC 0b fitness の前提)。
    """
    L, dim = 256, 8
    inputs_a = rng.uniform(-1, 1, size=(L, dim))
    rng2 = np.random.default_rng(int(rng.integers(1, 1_000_000)))
    inputs_b = rng2.uniform(-1, 1, size=(L, dim))
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    s_a = run_sequence(inputs_a, gene)[-1]
    s_b = run_sequence(inputs_b, gene)[-1]
    eps = 1e-10
    rel_diff = float(np.linalg.norm(s_a - s_b) / (np.linalg.norm(s_a) + np.linalg.norm(s_b) + eps))
    ok = rel_diff > 0.1
    return ok, f"relative_distance={rel_diff:.3f} > 0.1 threshold"


def gate_g8_memory_persistence(rng: np.random.Generator) -> tuple[bool, str]:
    """[G8] 記憶持続性: decay=0.95 + 非ゼロ入力後に zero-input で state が即落ちしない.

    t=0..50 で非ゼロ入力、t=51..100 で zero 入力。
    ``||state[100]|| > 0.01 * ||state[50]||`` を要求。
    """
    L1, L2, dim = 50, 50, 8
    inputs = np.concatenate([
        rng.uniform(-1, 1, size=(L1, dim)),
        np.zeros((L2, dim)),
    ])
    gene = StateUpdateGene(decay=0.95, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene)
    norm_at_50 = float(np.linalg.norm(states[L1]))
    norm_at_100 = float(np.linalg.norm(states[L1 + L2]))
    ok = norm_at_100 > 0.01 * norm_at_50 if norm_at_50 > 1e-6 else False
    return ok, f"norm[50]={norm_at_50:.4f}, norm[100]={norm_at_100:.4f}, ratio={norm_at_100 / max(norm_at_50, 1e-9):.4f}"


def gate_g9_zero_state_escape(rng: np.random.Generator) -> tuple[bool, str]:
    """[G9] zero-state escape: state=0 初期 + 非ゼロ入力で N step 以内に norm > eps.

    v1 の degenerate を正面から弾くゲート (gpt-5.4 reviewer 指摘)。
    """
    L, dim = 16, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    gene = StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene, initial_state=np.zeros(dim))
    state_norms = np.linalg.norm(states, axis=1)
    # state[0] = 0 のはず、それ以降が escape するか
    escape_step = -1
    for t in range(1, L + 1):
        if state_norms[t] > 1e-3:
            escape_step = t
            break
    ok = escape_step >= 0 and escape_step <= 5
    return ok, f"escape at step {escape_step} (within 5 steps), final_norm={state_norms[-1]:.4f}"


def gate_g10_parameter_sensitivity(rng: np.random.Generator) -> tuple[bool, str]:
    """[G10] parameter sensitivity: decay/mix/gate_str を動かすと出力が変わる.

    base gene と 3 種類の摂動 gene で最終 state 距離 > 0.01 を要求。
    parameter が dead でないことの保証。
    """
    L, dim = 128, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    base = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    perturbed = [
        ("decay+0.2", StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.5)),
        ("mix+0.3", StateUpdateGene(decay=0.7, mix=0.8, gate_str=0.5)),
        ("gate+0.5", StateUpdateGene(decay=0.7, mix=0.5, gate_str=1.0)),
    ]
    s_base = run_sequence(inputs, base)[-1]
    msgs: list[str] = []
    all_ok = True
    for name, gene in perturbed:
        s_p = run_sequence(inputs, gene)[-1]
        dist = float(np.linalg.norm(s_p - s_base))
        ok = dist > 0.01
        msgs.append(f"{name}:dist={dist:.4f}")
        all_ok = all_ok and ok
    return all_ok, " | ".join(msgs)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 0a v2 — state update 数式遺伝子 (RWKV-style) falsifiable verification")
    print("=" * 72)
    rng = np.random.default_rng(20260529)

    gates = [
        ("G1: single-step finite", lambda: gate_g1_single_step_finite(rng)),
        ("G2: bounded norm L=256 (K=10)", lambda: gate_g2_bounded_norm(rng)),
        ("G3: determinism (seed=42)", gate_g3_determinism),
        ("G4: degenerate values (v2 extended)", lambda: gate_g4_degenerate(rng)),
        ("G5: random population N=20", lambda: gate_g5_random_population(rng, 20)),
        ("G6: non-trivial activation", lambda: gate_g6_nontrivial_activation(rng)),
        ("G7: input distinguishability", lambda: gate_g7_input_distinguishability(rng)),
        ("G8: memory persistence", lambda: gate_g8_memory_persistence(rng)),
        ("G9: zero-state escape", lambda: gate_g9_zero_state_escape(rng)),
        ("G10: parameter sensitivity", lambda: gate_g10_parameter_sensitivity(rng)),
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
        print("PoC 0a v2 verdict: PASS — RWKV-style state update gene は falsifiable")
        print("                   命題 (有界 ∧ 非自明 ∧ 情報伝達 ∧ 記憶) を全て満たす.")
        print("                   次段 PoC 0b (合成 sequence fitness) に進めます.")
        return 0
    print("PoC 0a v2 verdict: FAIL — gene 表現または範囲設計を見直してから次へ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
