# SPDX-License-Identifier: Apache-2.0
"""quadratic_readout の positive control — degree-2 readout は何次までの parity を解けるか.

梯子段1 VERDICT の決定的主張:

    「5-bit parity = degree-5 単項式 (b1*b2*b3*b4*b5)。明示的2次 (degree-2) readout は
     2-bit XOR (= b1*b2) のみ線形分離でき、window>=3 では原理的に表現できない。」

を **reservoir ダイナミクスから切り離して** 検証する (Codex pair-review Finding 2 対応:
verdict は window=2 の R²=1.0 を positive control と書いていたが、対応する実験が無く
raw R² も保存されていなかった。本スクリプトがその裏付け artifact を生成する)。

方法 (ideal per-bit, 完全記憶を仮定):
- DelayedParityTask の target は ``bits[:window]`` の積 = ``b1*...*bw`` (±1 の parity)。
- reservoir を使わず、**真の直近 window ビットそのもの** を per-bit 特徴 (= 完全記憶 reservoir
  の理想上限) とし、``quadratic_features`` で 2 次展開して ridge fit する。
- これにより「2 次特徴で parity を線形分離できるか」は **純粋に readout の多項式次数の問題**
  に帰着する。reservoir が状態を保持できるか否かの交絡を排除した上限テストである。

測定は held-out (train/eval 別 draw)・**raw (unclip) R²** と clip[0,1] R² の両方を報告。
予測: window=2 -> R²≈1.0, window=3,4,5 -> R²≈0 (or 負)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Windows cp932 console でも R²/絵文字を出せるよう stdout/stderr を UTF-8 に reconfigure
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# src/llcore (ridge readout) と step_c (task) を read-only 流用
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step_c_memory_tasks"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402
from memory_tasks import DelayedParityTask  # noqa: E402

# quadratic_features は機構実装と同一展開を使う (positive control の readout を本番と一致させる)
from mech_quadratic_readout import quadratic_features  # noqa: E402

_HERE = Path(__file__).resolve().parent

N_TRAIN = 256
N_EVAL = 256
RIDGE_LAMBDA = 1e-6  # positive control は overfit でなく可解性を見るため弱い正則化
SEEDS = list(range(8))
WINDOWS = [2, 3, 4, 5]
SEQ_LEN = 20


def _ideal_features(task: DelayedParityTask, n: int, rng: np.random.Generator):
    """真の直近 window ビット (完全記憶) を per-bit 特徴とし、target と一緒に収集する.

    reservoir を一切使わない。``inputs`` の先頭 window ビット = parity の入力そのものを
    特徴 (window,) とする = 完全記憶 reservoir の理想上限。"""
    feats: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(n):
        inputs, target = task.generate(rng)
        bits = inputs[: task.window, 0]  # 完全記憶: parity 対象ビットそのもの (window,)
        feats.append(bits.astype(np.float64))
        targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
    return np.array(feats, dtype=np.float64), np.array(targets, dtype=np.float64)


def _raw_r2(pred: np.ndarray, y: np.ndarray) -> float:
    """clip しない生の held-out R² (負値もそのまま返す)."""
    pred = np.atleast_2d(pred)
    mse = float(np.mean((pred - y) ** 2))
    var = float(np.mean((y - y.mean(axis=0)) ** 2))
    return 1.0 - mse / max(var, 1e-12)


def _eval_window(window: int, degree2: bool, seed: int) -> float:
    """1 seed・指定 window で ideal per-bit (+任意で 2 次展開) readout の raw R² を返す."""
    task = DelayedParityTask(seq_len=SEQ_LEN, window=window, in_dim=1)
    rng = np.random.default_rng(seed)
    x_tr, y_tr = _ideal_features(task, N_TRAIN, rng)
    x_ev, y_ev = _ideal_features(task, N_EVAL, rng)  # rng 続き = train と独立 (held-out)
    if degree2:
        x_tr = quadratic_features(x_tr)
        x_ev = quadratic_features(x_ev)
    readout = fit_ridge_readout(x_tr, y_tr, ridge_lambda=RIDGE_LAMBDA)
    return _raw_r2(readout(x_ev), y_ev)


def main() -> None:
    print("=== quadratic_readout positive control (ideal per-bit, raw R²) ===")
    print(f"n_train={N_TRAIN} n_eval={N_EVAL} ridge_lambda={RIDGE_LAMBDA} "
          f"seeds={len(SEEDS)} seq_len={SEQ_LEN}\n", flush=True)
    print(f"{'window':>6} | {'linear raw R²':>16} | {'degree-2 raw R²':>16} | "
          f"{'degree-2 解ける?':>14}")
    print("-" * 66)

    results: dict[str, dict] = {}
    for w in WINDOWS:
        lin = np.array([_eval_window(w, degree2=False, seed=s) for s in SEEDS])
        quad = np.array([_eval_window(w, degree2=True, seed=s) for s in SEEDS])
        solved = bool(quad.mean() > 0.99)
        results[f"window={w}"] = {
            "degree_of_target_monomial": w,
            "linear_raw_r2": {"mean": float(lin.mean()), "std": float(lin.std()),
                              "per_seed": lin.tolist()},
            "degree2_raw_r2": {"mean": float(quad.mean()), "std": float(quad.std()),
                               "per_seed": quad.tolist()},
            "degree2_solves": solved,
        }
        print(f"{w:>6} | {lin.mean():>+10.4f} ±{lin.std():.3f} | "
              f"{quad.mean():>+10.4f} ±{quad.std():.3f} | {str(solved):>14}", flush=True)

    out = {
        "experiment": "quadratic_readout positive control (ideal per-bit features)",
        "claim_under_test": ("degree-2 readout solves degree-2 monomial (window=2 parity) "
                             "but not degree>=3 (window>=3)"),
        "protocol": {"n_train": N_TRAIN, "n_eval": N_EVAL, "ridge_lambda": RIDGE_LAMBDA,
                     "seeds": SEEDS, "seq_len": SEQ_LEN, "metric": "raw held-out R² (unclipped)"},
        "results": results,
    }
    (_HERE / "exp_quad_positive_control_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 解釈 ===")
    w2 = results["window=2"]["degree2_raw_r2"]["mean"]
    w5 = results["window=5"]["degree2_raw_r2"]["mean"]
    print(f"  window=2 degree-2 raw R² = {w2:+.4f} (≈1.0 期待: 2-bit XOR = degree-2 単項式)")
    print(f"  window=5 degree-2 raw R² = {w5:+.4f} (≈0 期待: 5-bit parity = degree-5 単項式)")
    print("  → 2 次 readout の可解性は target 単項式の次数で決まる (完全記憶でも degree 不足は解けない)。")


if __name__ == "__main__":
    main()
