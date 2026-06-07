# SPDX-License-Identifier: Apache-2.0
"""実験7: 実 ESN landscape で ③ が立つか — 手法比較 (heuristic でなく実測).

実験6 の grid は滑らか broad ridge に見えたが、crude valley 検出器の誤検出を避けるため
**実際に MAP-Elites vs RR-hillclimb vs panmictic-GA vs random** を走らせ、③ が立つか判定する。

2 条件:
- (A) 3-param ESN gene (spectral_radius, leak, input_scale): behavior=(rho,leak)。
- (B) 高次元 gene (per-neuron leak ベクトル, N=40 dim): llcore thesis (dynamics を gene 化) に近い。
  behavior=(平均 leak, leak の分散) の 2D descriptor。high-dim で random が不利 → MAP-Elites の
  本領 (もし landscape が欺瞞的なら ③ が立つはず)。

exp5 の判定基準: MAP-Elites が 3 baseline 全てに有意勝利なら ③成立、そうでなければ landscape は
滑らか/低次元で ③ 不要 (= 実 proxy では MAP-Elites/GPU 投資の追加価値は限定的)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esn_landscape import ESN, _ensure_utf8_stdout, load_corpus  # noqa: E402
from selection_lab import compare, run_methods_over_seeds  # noqa: E402

_ensure_utf8_stdout()

_IDX, _V, _ = load_corpus(max_chars=24000)
_ESN = ESN(n_reservoir=40, vocab=_V, seed=0)
_N_TRAIN, _N_EVAL, _WASHOUT = 3000, 1500, 80


def _acc_3param(gene: np.ndarray) -> float:
    """3-param ESN gene の next-char 精度 (deterministic)."""
    rho = 0.1 + 1.4 * gene[0]      # [0,1]->[0.1,1.5]
    leak = 0.05 + 0.95 * gene[1]   # [0,1]->[0.05,1.0]
    in_s = 0.3 + 1.2 * gene[2]     # [0,1]->[0.3,1.5]
    return _ncacc(np.array([rho, leak, in_s]))


def _ncacc(esn_gene: np.ndarray) -> float:
    states = _ESN.run(_IDX[: _WASHOUT + _N_TRAIN + _N_EVAL], esn_gene)
    tgt = _IDX[1: _WASHOUT + _N_TRAIN + _N_EVAL + 1]
    S = states[_WASHOUT: _WASHOUT + _N_TRAIN]
    Y = np.eye(_V)[tgt[_WASHOUT: _WASHOUT + _N_TRAIN]]
    A = np.concatenate([S, np.ones((len(S), 1))], axis=1)
    W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Y)
    Se = states[_WASHOUT + _N_TRAIN: _WASHOUT + _N_TRAIN + _N_EVAL]
    Ae = np.concatenate([Se, np.ones((len(Se), 1))], axis=1)
    pred = (Ae @ W).argmax(axis=1)
    return float(np.mean(pred == tgt[_WASHOUT + _N_TRAIN: _WASHOUT + _N_TRAIN + _N_EVAL]))


# ---- 高次元 gene: per-neuron leak ベクトル (N dim), rho/in_scale 固定 ----
_N_RES = _ESN.N


def _acc_perneuron(gene: np.ndarray) -> float:
    """gene = per-neuron leak (N dim) ∈ [0,1]. rho=0.95, in_scale=1.0 固定. 力学を gene 化."""
    leak = np.clip(gene, 0.0, 1.0)
    W = _ESN.W0 * 0.95
    state = np.zeros(_N_RES)
    n = _WASHOUT + _N_TRAIN + _N_EVAL
    states = np.empty((n, _N_RES))
    for t in range(n):
        u = _ESN.W_in[:, _IDX[t]] * 1.0
        state = (1 - leak) * state + leak * np.tanh(u + W @ state)
        states[t] = state
    tgt = _IDX[1: n + 1]
    S = states[_WASHOUT: _WASHOUT + _N_TRAIN]
    Y = np.eye(_V)[tgt[_WASHOUT: _WASHOUT + _N_TRAIN]]
    A = np.concatenate([S, np.ones((len(S), 1))], axis=1)
    Wr = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Y)
    Se = states[_WASHOUT + _N_TRAIN: _WASHOUT + _N_TRAIN + _N_EVAL]
    Ae = np.concatenate([Se, np.ones((len(Se), 1))], axis=1)
    pred = (Ae @ Wr).argmax(axis=1)
    return float(np.mean(pred == tgt[_WASHOUT + _N_TRAIN: _WASHOUT + _N_TRAIN + _N_EVAL]))


def _run(name, eval_once, behavior, dim, behavior_bounds, grid_shape, n_evals, n_seeds):
    bounds = (np.zeros(dim), np.ones(dim))
    res = run_methods_over_seeds(
        lambda g, rng: eval_once(g), behavior,
        dim=dim, bounds=bounds, behavior_bounds=behavior_bounds, grid_shape=grid_shape,
        n_evals=n_evals, n_seeds=n_seeds, honest_n_trials=1, sigma=0.12,
    )
    print(f"\n=== 条件 {name} (dim={dim}, n_evals={n_evals}, n_seeds={n_seeds}) ===")
    for k, v in res.items():
        print(f"  {k:14s}: acc mean={v.mean():.4f} std={v.std():.4f} max={v.max():.4f}")
    passes = {}
    for base in ("rr_hillclimb", "panmictic_ga", "random"):
        c = compare(res["map_elites"], res[base], "me", base)
        passes[base] = c.passes
        print(f"  MAP-Elites vs {base:13s}: diff={c.diff:+.4f} p={c.wilcoxon_p:.4g} "
              f"δ={c.cliff_delta:+.2f} → {'③成立側' if c.passes else '有意差なし'}")
    return all(passes.values())


def main() -> int:
    print(f"実験7: 実 ESN×テキスト landscape で ③ が立つか (corpus={len(_IDX)} vocab={_V} N={_N_RES})")
    print("=" * 72)
    a = _run("(A) 3-param ESN", _acc_3param, lambda g: g[:2].copy(), 3,
             (np.zeros(2), np.ones(2)), (10, 10), n_evals=400, n_seeds=8)
    b = _run("(B) per-neuron leak (高次元)", _acc_perneuron,
             lambda g: np.array([g.mean(), g.std()]), _N_RES,
             (np.array([0.0, 0.0]), np.array([1.0, 0.5])), (12, 12), n_evals=600, n_seeds=6)
    print("\n" + "=" * 72)
    print(f"  (A) 3-param ESN: ③成立 = {a}")
    print(f"  (B) per-neuron 高次元: ③成立 = {b}")
    if not a and not b:
        print("  → 実 proxy landscape は ③ が立たない (滑らか/低次元) = MAP-Elites の追加価値は限定的。")
        print("    exp4 の欺瞞 corridor は実テキスト proxy には自然出現しない (honest)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
