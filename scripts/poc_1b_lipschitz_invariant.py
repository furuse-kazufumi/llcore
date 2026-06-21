# SPDX-License-Identifier: Apache-2.0
"""PoC 1b — 状態方向 Lipschitz contraction invariant (Stage 1b) falsifiable 検証.

falsifiable 命題:
    RWKV 更新写像 ``s' = decay·s + (1−decay)·tanh(mix·x + gate_str·s)`` は各座標独立
    (対角写像) で、座標ヤコビ ``∂s'/∂s = decay + (1−decay)·gate_str·t``
    (t = sech²(pre) = 1−tanh²(pre) ∈ (0,1]) を持つ。状態方向 Lipschitz 定数を
    ``L = sup_{|s|≤1,|x|≤1} |∂s'/∂s|`` と定義する。clip 済み gene について Z3 が
    ``∃ t∈[0,1]. |decay+(1−decay)·gate_str·t| ≥ 1`` を unsat と判定したとき、その gene は
    L<1 を満たす (state-direction contraction certified)。これは sup-norm の global
    contraction を含意し、Banach により一意固定点と |s|≤1 有界 (state_norm 整合) を保証する。

反証形:
    certified と判定された gene が numpy シミュレーション (L=200, dim=8, 2 本の初期状態を
    変えた軌跡) で軌跡差ノルムを縮小させない (gapN/gap0 が 1 に近いまま) ことが一度でも
    起これば、本命題は偽。

破綻ゲート (BG1-BG5):
- [BG1] Z3 per-gene 応答が代表 200 gene 全件で elapsed<1.0s (線形縮約=高速)
- [BG2] reject 率が代表 200 gene で 0% でも 100% でもない (ゲートが弁別する)
- [BG3] certified gene が numpy シミュレーションで実際に縮小 (ratio<1; 命題の直接反証回避)
        + 非 certified の対照 (d=0.9,g=2.0) は発散 (ratio>1) = 弁別力の証拠
- [BG4] 既存 verifier テスト (test_poc_1a_z3_invariant.py) が本追加で 1 件も fail しない
- [BG5] contraction certified gene は |s|≤1 有界 (state_norm) も満たす (整合性)

使い方::

    py -3.11 scripts/poc_1b_lipschitz_invariant.py

依存: z3-solver (optional, ``pip install llmesh-llcore[z3]``) + numpy。
honest 留保:
- sat は achievable でない t (例 t=0) を使った反例かもしれず保守的 (false reject 可)。
  soundness は「certified なら必ず真」だけを要求し completeness は要求しない。
- decay=1 (純記憶, gate=0) は L=1 ちょうどで strict L<1 では reject される
  (marginal 安定を contraction 厳格定義から外す副作用)。
"""
from __future__ import annotations

import subprocess
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

from llcore.state_update import StateUpdateGene, run_sequence  # noqa: E402
from llcore.verifier import (  # noqa: E402
    is_z3_available,
    verify_gene_safe,
    verify_lipschitz_contraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_genes(n: int, seed: int) -> list[StateUpdateGene]:
    """clip 範囲一様に n 個 gene を生成."""
    rng = np.random.default_rng(seed)
    out: list[StateUpdateGene] = []
    for _ in range(n):
        out.append(
            StateUpdateGene(
                decay=float(rng.uniform(0.0, 1.0)),
                mix=float(rng.uniform(-1.0, 1.0)),
                gate_str=float(rng.uniform(-2.0, 2.0)),
            )
        )
    return out


def _trajectory_ratio(
    gene: StateUpdateGene, *, L: int = 200, dim: int = 8, seed: int = 0
) -> float:
    """初期状態のみ異なる 2 軌跡の差ノルム比 gapN/gap0 を返す.

    L<1 contraction なら軌跡差は L^L で 0 に縮む (sup-norm contraction)。
    """
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-1.0, 1.0, size=(L, dim))
    s0a = rng.uniform(-1.0, 1.0, size=dim)
    s0b = rng.uniform(-1.0, 1.0, size=dim)
    traj_a = run_sequence(inputs, gene, initial_state=s0a)
    traj_b = run_sequence(inputs, gene, initial_state=s0b)
    gap0 = float(np.linalg.norm(traj_a[0] - traj_b[0]))
    gap_n = float(np.linalg.norm(traj_a[-1] - traj_b[-1]))
    if gap0 == 0.0:
        return 0.0
    return gap_n / gap0


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate_bg1_timeout(genes: list[StateUpdateGene]) -> tuple[bool, str]:
    """[BG1] Z3 per-gene 応答が全件 <1.0s (線形縮約=高速)."""
    if not is_z3_available():
        return False, "z3 unavailable — cannot measure Z3 timing (fail-closed)"
    elapsed_ms: list[float] = []
    for g in genes:
        t0 = time.perf_counter()
        verify_lipschitz_contraction(g, timeout_ms=1000)
        elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(elapsed_ms)
    max_ms = float(arr.max())
    med_ms = float(np.median(arr))
    ok = max_ms < 1000.0
    return ok, (
        f"{len(genes)} genes: median={med_ms:.2f}ms, max={max_ms:.2f}ms "
        f"(target <1000ms; total={arr.sum():.1f}ms)"
    )


def gate_bg2_reject_rate(genes: list[StateUpdateGene]) -> tuple[bool, str]:
    """[BG2] reject 率が 0% でも 100% でもない (ゲートが弁別する)."""
    rejects = 0
    for g in genes:
        r = verify_lipschitz_contraction(g)
        if r.contraction is False:
            rejects += 1
    rate = rejects / len(genes)
    ok = 0.0 < rate < 1.0
    return ok, (
        f"reject {rejects}/{len(genes)} = {rate*100:.1f}% "
        f"(require 0% < rate < 100%; admit boundary: decay+(1-decay)|gate|<1)"
    )


def gate_bg3_sim_contraction() -> tuple[bool, str]:
    """[BG3] certified gene が numpy sim で縮小 + 非 certified は発散 (弁別力)."""
    certified = [
        StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0),
        StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5),
        StateUpdateGene(decay=0.95, mix=0.3, gate_str=0.1),
        StateUpdateGene(decay=0.8, mix=0.2, gate_str=0.2),
    ]
    ratios: list[float] = []
    all_contract = True
    for g in certified:
        r = verify_lipschitz_contraction(g)
        if r.contraction is not True:
            return False, f"expected certified but got {r.contraction} for {g}"
        ratio = _trajectory_ratio(g)
        ratios.append(ratio)
        if ratio >= 1.0:
            all_contract = False

    # 対照: 非 certified の発散例 (弁別力の証拠)
    non_cert = StateUpdateGene(decay=0.9, mix=1.0, gate_str=2.0)
    nc_result = verify_lipschitz_contraction(non_cert)
    nc_ratio = _trajectory_ratio(non_cert)
    discriminates = (nc_result.contraction is False) and (nc_ratio > 1.0)

    ok = all_contract and discriminates
    ratios_str = ", ".join(f"{r:.4f}" for r in ratios)
    return ok, (
        f"certified ratios=[{ratios_str}] (all<1: {all_contract}); "
        f"non-certified d=0.9,g=2.0 ratio={nc_ratio:.4f} reject={nc_result.contraction is False} "
        f"(discriminates: {discriminates})"
    )


def gate_bg4_regression() -> tuple[bool, str]:
    """[BG4] 既存 verifier テスト (test_poc_1a_z3_invariant.py) 全 PASS 維持."""
    test_file = _PROJ_ROOT / "tests" / "unit" / "test_poc_1a_z3_invariant.py"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q"],
        cwd=str(_PROJ_ROOT),
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    tail = proc.stdout.strip().splitlines()
    summary = tail[-1] if tail else "(no output)"
    return ok, f"pytest test_poc_1a_z3_invariant.py: {summary}"


def gate_bg5_statenorm_consistency() -> tuple[bool, str]:
    """[BG5] contraction certified gene は state_norm (|s|≤1) も満たす (整合性).

    理論: L<1 ⟹ Banach 一意固定点 + |s'|≤decay·|s|+(1−decay)·1≤1 (convex comb)
    で state_norm は無条件成立。よって contraction⟹state_norm は構造的に保証。
    """
    certified = [
        StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.0),
        StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5),
        StateUpdateGene(decay=0.95, mix=0.3, gate_str=0.1),
        StateUpdateGene(decay=0.8, mix=0.2, gate_str=0.2),
    ]
    failures: list[str] = []
    max_state_overall = 0.0
    for g in certified:
        lr = verify_lipschitz_contraction(g)
        if lr.contraction is not True:
            failures.append(f"{g} not certified")
            continue
        # (a) Z3: state_norm も unsat (両立)
        sn = verify_gene_safe(g)
        if not sn.ok:
            failures.append(f"{g} certified contraction but state_norm rejected: {sn.reason}")
        # (b) long-sim max|state| <= 1+eps
        rng = np.random.default_rng(11)
        inputs = rng.uniform(-1.0, 1.0, size=(500, 8))
        states = run_sequence(inputs, g, initial_state=rng.uniform(-1.0, 1.0, size=8))
        max_state = float(np.max(np.abs(states)))
        max_state_overall = max(max_state_overall, max_state)
        if max_state > 1.0 + 1e-6:
            failures.append(f"{g} max|state|={max_state:.4f} > 1")
    ok = not failures
    return ok, (
        f"all {len(certified)} certified genes pass state_norm (Z3 ok) + "
        f"max|state|={max_state_overall:.4f}<=1"
        + ("" if ok else f"; FAILURES: {failures}")
    )


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 1b — 状態方向 Lipschitz contraction invariant falsifiable verification")
    print("=" * 72)
    print(f"z3 available: {is_z3_available()}")
    print()

    # 代表 200 gene サンプル (BG1/BG2 共有, seed=7)
    sample_genes = _random_genes(200, seed=7)

    gates = [
        ("BG1: per-gene Z3 timing (<1s, 200 genes)", lambda: gate_bg1_timeout(sample_genes)),
        ("BG2: reject rate non-degenerate (0%<r<100%)", lambda: gate_bg2_reject_rate(sample_genes)),
        ("BG3: sim contraction (certified shrink, non-cert diverge)", gate_bg3_sim_contraction),
        ("BG4: regression (test_poc_1a all PASS)", gate_bg4_regression),
        ("BG5: state_norm consistency (contraction⟹|s|≤1)", gate_bg5_statenorm_consistency),
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
        print("PoC 1b verdict: PASS — Z3 で状態方向 contraction (L<1) を sound に証明.")
        print("                 state_norm の真部分集合を能動的に弁別する質的に強い不変量.")
        print("                 Banach 一意固定点 + state_norm 整合を構造的に保証.")
        return 0
    print("PoC 1b verdict: FAIL — encoding/sound 性または gene 集合を見直してから次へ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
