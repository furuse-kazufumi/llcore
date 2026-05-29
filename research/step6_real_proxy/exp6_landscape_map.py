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
    print("=" * 72)
    if len(lm) <= 1:
        print("  → 単峰 broad-basin 型 (copy delay=0 に類似): ③ 不要、hill-climbing で十分の見込み")
    else:
        # 2 local maxima 間の経路に valley があるか (簡易: 直線中点の値)
        valleys = 0
        for a in range(len(lm)):
            for b in range(a + 1, len(lm)):
                # grid 上の 2 点
                ia = int(np.argmin(abs(rhos - lm[a][0]))); ja = int(np.argmin(abs(leaks - lm[a][1])))
                ib = int(np.argmin(abs(rhos - lm[b][0]))); jb = int(np.argmin(abs(leaks - lm[b][1])))
                mid = grid[(ia + ib) // 2, (ja + jb) // 2]
                if mid < min(lm[a][2], lm[b][2]) - 0.1 * (gmax - gmin):
                    valleys += 1
        print(f"  複数 local maxima 間の valley ペア: {valleys}")
        if valleys > 0:
            print("  → 欺瞞的 (exp4 型) の兆候あり: ③/MAP-Elites が load-bearing になりうる")
        else:
            print("  → 複数 maxima だが valley 弱 = 連結 plateau 寄り: ③ の効果は限定的の見込み")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
