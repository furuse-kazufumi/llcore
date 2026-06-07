# SPDX-License-Identifier: Apache-2.0
"""Stage 3b kernel 多様化 — tiny dynamic smoke (finite state + state_norm bounded).

`kernels.py` の最小 skeleton (KernelGenome + 4 kernel forward dynamics) に **有界入力
(|x|<=1) を数ステップ流し**、各 kernel について:

  (a) finite_state   — 軌跡が NaN/Inf を一切含まない (発散しない)
  (b) state_norm_ok  — 全 step の state ノルムが緩い上界 K (=MAX_DIM 起因 + margin) 以下

を numpy で確認する **動的 smoke** (Z3 gate 検証は `smoke_kernel_gates.py` が別途担当)。
これは BREAK_GATES.md の BG4 (kernel 軌跡 finite) の最小版に相当する。

honest 留保:
- mamba/hopfield/linear_attn は対角スカラ mock (full kernel ではない)。
- 各 kernel について clip 範囲から複数 gene をサンプルし、最悪値を集約する。
- rwkv は既存 run_sequence 経由 (後方互換)、他 3 は research mock 経由。
- 失敗 (発散/NaN) も握り潰さず JSON に記録する (honest disclosure)。

実行 (単独可・seed 固定・UTF-8):
    py -3.11 research/kernel_diversification/smoke_kernels.py
出力:
    research/kernel_diversification/smoke_results.json
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

# UTF-8 stdout (Windows cp932 console 対策, feedback_cli_utf8_stdout_pattern)
def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:  # pragma: no cover - already wrapped / non-buffered stream
        pass


# research skeleton を import (同ディレクトリ)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernels import (  # noqa: E402
    KERNEL_NAMES,
    KERNEL_THETA_LOWER,
    KERNEL_THETA_UPPER,
    KERNEL_DIMS,
    MAX_DIM,
    N_KERNELS,
    KernelGenome,
    run_sequence_kernel,
)

# smoke パラメータ (tiny)
SEED = 20260601
N_GENE = 32          # 各 kernel で試す gene 数 (tiny)
L = 64               # sequence 長
DIM = 8              # state 次元
MAX_INPUT = 1.0      # |x| <= 1 (有界入力)
# 緩い state_norm 上界。有界入力 + (rwkv/mamba/hopfield は convex/bounded 写像で |s|<=1) を
# 想定するが、linear_attn は softplus*x で 1 step 寄与が >1 になり得るので余裕を持たせる。
# K は「発散していない」を判定する緩い sanity bound (np.sqrt(DIM) * margin)。
STATE_NORM_K = float(np.sqrt(DIM) * 20.0)


def _sample_genome(name: str, rng: np.random.Generator) -> KernelGenome:
    """clip 範囲内で当該 kernel の random genome を作る (junk 次元含む)."""
    kid = float(KERNEL_NAMES.index(name)) + 0.5  # floor で当該 index に落ちる連続値
    dim = KERNEL_DIMS[name]
    lo, hi = KERNEL_THETA_LOWER[name], KERNEL_THETA_UPPER[name]
    theta = np.zeros(MAX_DIM, dtype=np.float64)
    theta[:dim] = lo + (hi - lo) * rng.random(dim)
    if dim < MAX_DIM:
        theta[dim:] = rng.random(MAX_DIM - dim)  # junk DNA
    return KernelGenome(kernel_id=kid, theta=theta).clipped()


def smoke_one_kernel(name: str, rng: np.random.Generator) -> dict:
    """1 kernel に N_GENE gene × 有界入力を流し finite/state_norm を集約."""
    finite_all = True
    max_norm_over_genes = 0.0
    n_nonfinite = 0
    n_norm_exceed = 0
    # 代表 1 gene の終端 norm (可視化用)
    sample_final_norm = None

    for i in range(N_GENE):
        g = _sample_genome(name, rng)
        inputs = rng.uniform(-MAX_INPUT, MAX_INPUT, size=(L, DIM))
        states = run_sequence_kernel(g, inputs)

        finite = bool(np.all(np.isfinite(states)))
        if not finite:
            finite_all = False
            n_nonfinite += 1
            # 非有限なら norm 計算をスキップ (inf*inf 回避)
            continue

        # 各 step の L2 ノルムの最大
        norms = np.linalg.norm(states, axis=1)
        gmax = float(np.max(norms))
        max_norm_over_genes = max(max_norm_over_genes, gmax)
        if gmax > STATE_NORM_K:
            n_norm_exceed += 1
        if sample_final_norm is None:
            sample_final_norm = float(norms[-1])

    state_norm_ok = (n_norm_exceed == 0) and finite_all
    note = "via existing run_sequence (backcompat)" if name == "rwkv" else "research diagonal mock"
    return {
        "kernel": name,
        "n_gene": N_GENE,
        "finite_state": finite_all,
        "state_norm_ok": bool(state_norm_ok),
        "max_state_norm": max_norm_over_genes,
        "state_norm_K": STATE_NORM_K,
        "n_nonfinite_genes": n_nonfinite,
        "n_norm_exceed_genes": n_norm_exceed,
        "sample_final_norm": sample_final_norm,
        "note": note,
    }


def run() -> dict:
    rng = np.random.default_rng(SEED)
    per_kernel = []
    for name in KERNEL_NAMES:
        per_kernel.append(smoke_one_kernel(name, rng))

    # BG5 後方互換 quick check: kernel_id=0 経路 == 既存 run_sequence(StateUpdateGene) bit 一致。
    # (kernels.py の rwkv 経路は内部で run_sequence を呼ぶので、ここでは同じ theta で外から
    #  直接 run_sequence を呼び bit 一致を確認 = src 不変の機械担保。)
    from llcore.state_update import StateUpdateGene, run_sequence  # noqa: E402

    bc_match = True
    for _ in range(20):
        th = np.array(
            [rng.random(), rng.uniform(-1, 1), rng.uniform(-2, 2), rng.random()]
        )
        g = KernelGenome(kernel_id=0.0, theta=th)
        inputs = rng.uniform(-1, 1, size=(L, DIM))
        got = run_sequence_kernel(g, inputs)
        gene = StateUpdateGene(decay=float(th[0]), mix=float(th[1]), gate_str=float(th[2]))
        ref = run_sequence(inputs, gene)
        if not np.array_equal(got, ref):
            bc_match = False
            break

    all_ok = all(k["finite_state"] and k["state_norm_ok"] for k in per_kernel) and bc_match

    return {
        "meta": {
            "seed": SEED,
            "n_gene_per_kernel": N_GENE,
            "L": L,
            "dim": DIM,
            "max_input_abs": MAX_INPUT,
            "state_norm_K": STATE_NORM_K,
            "n_kernels": N_KERNELS,
            "note": (
                "tiny dynamic smoke: bounded |x|<=1 through each kernel, "
                "check finite state + state_norm bounded. diagonal scalar mocks "
                "(except rwkv = existing run_sequence). NOT full kernel impl."
            ),
        },
        "per_kernel": per_kernel,
        "bg5_rwkv_backcompat_bit_match": bc_match,
        "all_ok": bool(all_ok),
    }


if __name__ == "__main__":
    _ensure_utf8_stdout()
    res = run()
    out = Path(__file__).resolve().parent / "smoke_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for k in res["per_kernel"]:
        print(
            f"{k['kernel']:18s} finite={k['finite_state']!s:5s} "
            f"norm_ok={k['state_norm_ok']!s:5s} "
            f"max_norm={k['max_state_norm']:.4f} "
            f"({k['note']})"
        )
    print(f"BG5 backcompat bit_match={res['bg5_rwkv_backcompat_bit_match']}")
    print(f"all_ok={res['all_ok']}")
    print(f"written: {out}")
