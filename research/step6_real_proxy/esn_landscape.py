# SPDX-License-Identifier: Apache-2.0
"""手順 6: 実 substrate proxy の landscape 欺瞞性測定 — ③/MAP-Elites の実問題価値判定.

監査 §7b 手順 6 + STEP4_SELECTION_VERDICT.md の残課題:
exp4 で「③ が立つのは欺瞞的 corridor landscape 限定」と判明。残る問いは
**「実 substrate (実 task / 実 LLM fitness) の landscape が欺瞞的か滑らかか」**。

CPU 完結かつ llcore の thesis (dynamics gene を進化させる) に最も近い proxy =
**Echo State Network (ESN: 固定 reservoir 力学 + 学習 ridge readout)** を **実テキスト**
(llcore 自身の Python source, ASCII 124K chars) の **next-char 予測**に適用する。

- gene = ESN ハイパーパラメータ (spectral_radius, leak_rate, input_scale) — reservoir 力学を決める。
  = llcore の「state update 力学を gene 化」の reservoir 版 (固定重み + 力学パラメータ進化)。
- fitness = held-out next-char 予測精度 (ridge readout, closed-form, CPU 安価)。

測定: gene 空間の landscape を grid sample し **欺瞞性を判定**:
- 単峰 broad-basin (copy delay=0 型) → ③ 不要、hill-climbing で十分。
- 複数の分離 basin / 局所最適 + valley (exp4 型) → ③/MAP-Elites が load-bearing になりうる。

honest 留保: これは reservoir computing (固定力学 + 学習 readout) であり backprop で学習する
full LLM ではない。「dynamics gene を進化 + readout を学習」という llcore thesis の CPU 近似で
あって、deep network の学習 landscape へ一般化できる保証はない。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


def load_corpus(max_chars: int = 60000) -> tuple[np.ndarray, int, dict]:
    """llcore の Python source を ASCII char index 列として読む (実テキスト)."""
    texts = []
    for f in sorted(glob.glob(str(_REPO / "src" / "**" / "*.py"), recursive=True)):
        t = Path(f).read_text(encoding="utf-8", errors="ignore")
        texts.append("".join(c for c in t if ord(c) < 128))
    allc = "".join(texts)[:max_chars]
    vocab = sorted(set(allc))
    stoi = {c: i for i, c in enumerate(vocab)}
    idx = np.array([stoi[c] for c in allc], dtype=np.int64)
    return idx, len(vocab), stoi


class ESN:
    """Echo State Network: 固定 random reservoir + gene 制御の力学.

    state[t] = (1-leak)*state[t-1] + leak*tanh(input_scale*W_in@u[t] + W_res@state[t-1])
    W_res は spectral radius=1 に正規化済 → gene の spectral_radius を直接乗算。
    """

    def __init__(self, n_reservoir: int, vocab: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((n_reservoir, n_reservoir))
        # spectral radius 1 に正規化 (gene で再スケール)
        sr = max(abs(np.linalg.eigvals(w)))
        self.W0 = w / sr
        self.W_in = rng.standard_normal((n_reservoir, vocab))
        self.N = n_reservoir
        self.V = vocab

    def run(self, idx: np.ndarray, gene: np.ndarray) -> np.ndarray:
        """char index 列を回し state 軌跡 (T, N) を返す."""
        rho, leak, in_scale = float(gene[0]), float(gene[1]), float(gene[2])
        W = self.W0 * rho
        state = np.zeros(self.N)
        states = np.empty((len(idx), self.N))
        for t, c in enumerate(idx):
            u = self.W_in[:, c] * in_scale  # one-hot 入力 = W_in の c 列
            state = (1 - leak) * state + leak * np.tanh(u + W @ state)
            states[t] = state
        return states


def next_char_accuracy(
    esn: ESN, idx: np.ndarray, gene: np.ndarray, *,
    n_train: int, n_eval: int, washout: int = 100, ridge_lambda: float = 1.0,
) -> float:
    """ridge readout で next-char 予測精度 (held-out). gene の力学が予測に資するほど高い."""
    states = esn.run(idx[: washout + n_train + n_eval], gene)
    V = esn.V
    # target = next char の one-hot
    tgt = idx[1: washout + n_train + n_eval + 1]
    S = states[washout: washout + n_train]
    Y = np.eye(V)[tgt[washout: washout + n_train]]
    # ridge fit (bias 込み)
    A = np.concatenate([S, np.ones((len(S), 1))], axis=1)
    G = A.T @ A + ridge_lambda * np.eye(A.shape[1])
    W = np.linalg.solve(G, A.T @ Y)
    # held-out 精度
    Se = states[washout + n_train: washout + n_train + n_eval]
    Ae = np.concatenate([Se, np.ones((len(Se), 1))], axis=1)
    pred = (Ae @ W).argmax(axis=1)
    true = tgt[washout + n_train: washout + n_train + n_eval]
    return float(np.mean(pred == true))
