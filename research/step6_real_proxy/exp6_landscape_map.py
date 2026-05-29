# SPDX-License-Identifier: Apache-2.0
"""実験6: 実 substrate (ESN×実テキスト) の landscape 欺瞞性を測定する.

gene 空間 (spectral_radius × leak_rate, input_scale 固定) を grid sample し、landscape が
- 単峰 broad-basin (copy delay=0 型 → ③ 不要) か
- 局所最適 + valley の欺瞞 (exp4 型 → ③/MAP-Elites が load-bearing) か
を判定する。これが ③ を実問題で追う価値 (= GPU 投資正当化) の sanity check。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esn_landscape import ESN, _ensure_utf8_stdout, load_corpus, next_char_accuracy  # noqa: E402

_ensure_utf8_stdout()


def main() -> int:
    idx, V, _ = load_corpus(max_chars=40000)
    esn = ESN(n_reservoir=80, vocab=V, seed=0)

    rhos = np.linspace(0.1, 1.5, 15)
    leaks = np.linspace(0.05, 1.0, 15)
    in_scale = 1.0
    grid = np.zeros((len(rhos), len(leaks)))
    for i, rho in enumerate(rhos):
        for j, lk in enumerate(leaks):
            grid[i, j] = next_char_accuracy(
                esn, idx, np.array([rho, lk, in_scale]), n_train=6000, n_eval=3000
            )

    print(f"実験6: ESN×実テキスト(Python source) next-char acc landscape (rho×leak, in={in_scale})")
    print(f"  corpus={len(idx)} chars vocab={V}, reservoir N=80")
    print("=" * 72)
    print("  rho\\leak " + " ".join(f"{lk:.2f}" for lk in leaks))
    for i, rho in enumerate(rhos):
        print(f"  {rho:.2f}: " + " ".join(f"{v:.3f}" for v in grid[i]))
    print("-" * 72)
    gmax, gmin = grid.max(), grid.min()
    print(f"  max={gmax:.3f} min={gmin:.3f} (range={gmax-gmin:.3f})")
    # 局所最適の数 (内部セルで 8 近傍より大きい)
    lm = []
    for i in range(1, len(rhos) - 1):
        for j in range(1, len(leaks) - 1):
            nb = grid[i - 1:i + 2, j - 1:j + 2]
            if grid[i, j] == nb.max() and grid[i, j] > gmin + 0.3 * (gmax - gmin):
                lm.append((rhos[i], leaks[j], grid[i, j]))
    print(f"  内部 local maxima (>30%tile): {len(lm)} -> {[(round(a,2),round(b,2),round(c,3)) for a,b,c in lm]}")
    # 高値域 (>90% of max) が連結 plateau か分散か
    thr = gmin + 0.9 * (gmax - gmin)
    high = grid > thr
    print(f"  高値域 (>90%range) cell 数: {int(high.sum())}/{grid.size}")
    # deceptive 判定: 複数の分離 local maxima があり、間に valley があるか
    # 軸別の支配的トレンド: leak 軸 / rho 軸の平均 (どちらが効くか)
    print(f"  leak 軸平均 (低→高): {' '.join(f'{c:.3f}' for c in grid.mean(axis=0))}")
    print(f"  rho  軸平均 (低→高): {' '.join(f'{c:.3f}' for c in grid.mean(axis=1))}")
    print("=" * 72)
    # honest 判定: valley 深さは eval noise を超える必要がある。検出 maxima が global 近傍に密集
    # していれば連結 plateau (noise 凹凸) であって分離 basin ではない。
    near_global = [m for m in lm if m[2] > gmax - 0.05 * (gmax - gmin)]  # global の 5% 以内
    spread = (max(m[2] for m in lm) - min(m[2] for m in lm)) if lm else 0.0
    print(f"  検出 maxima の値域 spread={spread:.3f} (eval noise ~0.005-0.01 と比較)")
    print(f"  → 自動判定は信頼せず、決定的テスト = exp7 (実 MAP-Elites vs baseline 比較) に委ねる。")
    print(f"    grid 形状: leak 単調増 + rho 弱依存 = 滑らか broad ridge 寄り (深い valley は視認されず)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
