# SPDX-License-Identifier: Apache-2.0
"""正規化交絡検証 — clip 切替可能な held-out R² eval_once (src 非改変・research 隔離).

背景 (INVENTORY/DESIGN):
- ``research/step_c_memory_tasks/reservoir.py`` の :func:`make_eval_once` は
  最終行 (L177) で ``np.clip(r2, 0.0, 1.0)`` を **ハードコード**しており clip 引数を
  持たない。交絡 (A) = 「MSE>=baseline で R²=0 に飽和 → 勾配消失 → 平坦化 →
  谷が床に埋まり単峰に見える」を falsifiable に検証するには clip を OFF にした
  eval_once が必要。
- src/ は変更禁止のため、ここで **clip だけ切替可能**な同等 eval_once を再実装する。
  reservoir の ``res.run`` と src の :func:`fit_ridge_readout` を **read-only 流用**し、
  最後の clip 段のみ {True (現行 baseline) / False (raw R²) / soft (tanh)} を切替える。
  collect ロジックは reservoir.make_eval_once と bit-for-bit 同一手順
  (train→fit→held-out→R²) を踏襲する (clip 以外の差を作らない=交絡を clip に限定)。

非循環性 (G4): この eval_once は behavior 記述子も grid binning も③の勝敗も一切
参照しない。clip は eval_once の **出力変換** であり behavior 軸ではない。よって
clip=False で谷が出るなら「出力飽和が谷を床に潰していた」という純機構的因果であり、
③の定義 (選択圧/分離) に依存しない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- src/llcore を read-only 流用 (RAPTOR/llcore 規律: sys.path は src のみ追加) ---
_LLCORE_ROOT = Path(__file__).resolve().parents[2]  # research/normalization_confound -> llcore
sys.path.insert(0, str(_LLCORE_ROOT / "src"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402


def make_eval_once_clipswitch(
    res,
    task,
    *,
    n_train: int = 48,
    n_eval: int = 48,
    ridge_lambda: float = 1e-2,
    clip: str = "hard",
):
    """clip 段だけ切替可能な held-out R² eval_once を作る (reservoir.make_eval_once 同等).

    Parameters
    ----------
    res :
        ``run(gene, inputs) -> states`` を満たす reservoir (LeakyDelayLineReservoir)。
    task :
        ``generate(rng) -> (inputs, target)`` を満たすタスク。
    n_train, n_eval, ridge_lambda :
        reservoir.make_eval_once と同じ既定で揃える。
    clip : str
        - ``"hard"``  : ``np.clip(r2, 0, 1)`` = **現行 baseline** (reservoir.py L177 と同一)。
        - ``"none"``  : raw R² (負値・spread 許容)。交絡 A の OFF 条件。
        - ``"soft"``  : ``tanh(r2)`` (下限 -1, 飽和を緩和した滑らか変換)。

    Returns
    -------
    eval_once : Callable[[np.ndarray, np.random.Generator], float]
    """
    if clip not in ("hard", "none", "soft"):
        raise ValueError(f"clip must be hard/none/soft, got {clip!r}")

    def _collect(gene: np.ndarray, n: int, rng: np.random.Generator):
        states: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _ in range(n):
            inputs, target = task.generate(rng)
            states.append(res.run(gene, inputs)[-1])  # 最終時刻のみ (make_eval_once と同一)
            targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
        return np.array(states, dtype=np.float64), np.array(targets, dtype=np.float64)

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        s_tr, y_tr = _collect(gene, n_train, rng)
        readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)
        s_ev, y_ev = _collect(gene, n_eval, rng)
        pred = np.atleast_2d(readout(s_ev))
        mse = float(np.mean((pred - y_ev) ** 2))
        var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        if clip == "hard":
            return float(np.clip(r2, 0.0, 1.0))
        if clip == "soft":
            return float(np.tanh(r2))
        return float(r2)  # none = raw R²

    return eval_once


__all__ = ["make_eval_once_clipswitch"]
