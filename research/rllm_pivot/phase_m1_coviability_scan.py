# SPDX-License-Identifier: Apache-2.0
"""Phase -1: 純数値 ε>0 両立帯 scan (make-or-break feasibility, $0/CPU)。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) の Phase -1 / red-team F1 の数理判定を実データで決着する。
実装投資ゼロ: SmolLM2/Net2Net 不要、backends.py の cert_inf (_infnorm_sup) / _t_min のみを使う。

問い (make-or-break):
  width_grow で recurrent adapter (CoupledNDGene, s' = decay⊙s + (1-decay)⊙tanh(Ws+Vx)) を n→n+1
  に成長させたとき、**「(a) 新 unit が既存ダイナミクスを *非自明に* 変える ∧ (b) 全既存 row が cert_inf
  (sup‖J‖∞<1) を sound に保つ」を両立する ε>0 帯 (新 unit の結合強度)** は存在するか。

red-team F1 の訂正を反映:
  脅威は「M=Σ|W| 増 → box (t_min) 拡大 → sup 越境」ではない (sup は ti=1 支配が大半=box 幅と無関係)。
  真の脅威は「新 column が既存 row i の off-sum Σ_{j≠i}|W_ij| を増やし、ti=1 の row-bound が 1 を越える」
  = per-row abs-sum 増。本 scan は per-row 効果と box 効果を分離して測る。

honest:
  - 「非自明な進化価値」は proxy (新 unit 追加で既存出力軌跡が相対 L2 > τ 変化) で測る。真の「n+1 次元
    でしか表現できない関数」の証明ではない。proxy の限界は VERDICT に明記。
  - ε=0 (死んだ unit, 出力結合ゼロ) は (b) を自明に満たすが (a) を満たさない=無進化。両立帯の核心は
    「(a) を満たす最小 ε_alive < (b) を満たす最大 ε_max」が成立するか。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # cp932 console で日本語/記号を出す

# llcore.src を path に (本 research script は自己完結, raptor path 規約は別 project ゆえ非適用)
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llcore.verifier.backends import _infnorm_sup, _t_min  # noqa: E402


# --------------------------------------------------------------------------- #
# coupled kernel (backends.py の Jacobian と整合する forward)
# --------------------------------------------------------------------------- #
def coupled_run(X: np.ndarray, decay: np.ndarray, W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """s_{t+1} = decay⊙s_t + (1-decay)⊙tanh(W s_t + V x_t). X:(L,n) -> states:(L,n)."""
    n = decay.shape[0]
    assert X.shape[1] == n, f"X dim {X.shape[1]} != n {n}"
    s = np.zeros(n)
    out = np.empty((X.shape[0], n))
    for t in range(X.shape[0]):
        s = decay * s + (1.0 - decay) * np.tanh(W @ s + V @ X[t])
        out[t] = s
    return out


def cert_inf_sup(decay, W, V, max_input_abs=1.0) -> float:
    return _infnorm_sup(decay, W, _t_min(decay, W, V, max_input_abs))


def row_bounds_at(decay, W, t_lo, ti_choice):
    """各 row i の bound を ti=ti_choice ('lo' or 'hi') で返す (box vs ti=1 分離用)。"""
    n = decay.shape[0]
    res = np.empty(n)
    for i in range(n):
        off = sum(abs(W[i, j]) for j in range(n) if j != i)
        ti = (t_lo[i] if ti_choice == "lo" else 1.0)
        diag = abs(decay[i] + (1.0 - decay[i]) * ti * W[i, i])
        res[i] = diag + (1.0 - decay[i]) * ti * off
    return res


# --------------------------------------------------------------------------- #
# base gene 生成 (admit 済 = cert_inf PASS) を headroom 帯ごとに
# --------------------------------------------------------------------------- #
def sample_admitted_base(rng, n, max_input_abs=1.0, max_tries=60):
    """W を admit するまで縮小して cert_inf PASS な (decay, W, V=I) を返す。

    高 n では一様サンプルの admit 率が指数的に低い (全 row sup<1 が必要) ため、
    W を 0.85 倍ずつ縮小して必ず admit に到達させる (任意 n で確実、初期スケールで headroom を散らす)。
    """
    V = np.eye(n)
    for _ in range(max_tries):
        decay = rng.uniform(0.0, 1.0, size=n)
        W = (rng.normal(0.0, 1.0, size=(n, n)) / np.sqrt(n)) * float(rng.uniform(0.2, 0.9))
        for _ in range(50):
            sup = cert_inf_sup(decay, W, V, max_input_abs)
            if sup < 1.0:
                return decay, W, V, float(sup)
            W = W * 0.85
    return None


# --------------------------------------------------------------------------- #
# width_grow: n -> n+1。新 unit の incoming(row n) / outgoing(col n) / self を ε でスケール
# --------------------------------------------------------------------------- #
def grow_one(decay, W, V, rng, eps, in_dir, out_dir, self_w, new_decay):
    n = decay.shape[0]
    decay2 = np.empty(n + 1)
    decay2[:n] = decay
    decay2[n] = new_decay
    W2 = np.zeros((n + 1, n + 1))
    W2[:n, :n] = W
    # outgoing: 既存 row i <- 新 unit (col n) = 既存ダイナミクスを変える主因 (off-sum 増の主因)
    W2[:n, n] = eps * out_dir
    # incoming: 新 row n <- 既存 j (row n) = 新 state を駆動
    W2[n, :n] = eps * in_dir
    W2[n, n] = self_w
    V2 = np.eye(n + 1)
    return decay2, W2, V2


def function_change(decay, W, V, decay2, W2, V2, Xs):
    """新 unit 追加で既存出力 (先頭 n 次元) 軌跡が相対 L2 でどれだけ変わるか (複数入力で平均)。"""
    n = decay.shape[0]
    rels = []
    for X in Xs:
        base = coupled_run(X, decay, W, V)             # (L, n)
        Xg = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)  # 新 unit は外部入力なし
        grown = coupled_run(Xg, decay2, W2, V2)[:, :n]  # 既存次元のみ比較
        num = np.linalg.norm(grown - base)
        den = np.linalg.norm(base) + 1e-12
        rels.append(num / den)
    return float(np.mean(rels))


# --------------------------------------------------------------------------- #
# 1 base × 1 方向 で ε を sweep し両立帯を測る
# --------------------------------------------------------------------------- #
def sweep(decay, W, V, rng, Xs, max_input_abs=1.0, tau=0.05, n_eps=60, eps_hi=3.0):
    n = decay.shape[0]
    in_dir = rng.normal(size=n); in_dir /= (np.linalg.norm(in_dir) + 1e-12)
    out_dir = rng.normal(size=n); out_dir /= (np.linalg.norm(out_dir) + 1e-12)
    self_w = float(rng.uniform(-0.5, 0.5))
    new_decay = float(rng.uniform(0.0, 1.0))

    eps_grid = np.linspace(0.0, eps_hi, n_eps)
    sound = np.zeros(n_eps, dtype=bool)
    change = np.zeros(n_eps)
    for k, eps in enumerate(eps_grid):
        d2, W2, V2 = grow_one(decay, W, V, rng, eps, in_dir, out_dir, self_w, new_decay)
        sound[k] = cert_inf_sup(d2, W2, V2, max_input_abs) < 1.0
        change[k] = function_change(decay, W, V, d2, W2, V2, Xs)

    # ε_max = 最大の連続 sound 区間 (ε=0 から) の上端
    eps_max = 0.0
    for k, eps in enumerate(eps_grid):
        if sound[k]:
            eps_max = eps
        else:
            break
    # ε_alive = change >= tau になる最小 ε
    alive_idx = np.where(change >= tau)[0]
    eps_alive = float(eps_grid[alive_idx[0]]) if alive_idx.size else float("inf")

    band_exists = eps_alive < eps_max  # (a)∧(b) 両立帯が存在
    # 両立帯内 (ε_alive..ε_max) での最大 function change (=どれだけ非自明に動けるか)
    if band_exists:
        in_band = (eps_grid >= eps_alive) & (eps_grid <= eps_max)
        change_in_band = float(change[in_band].max())
    else:
        change_in_band = 0.0
    return {
        "eps_max": float(eps_max),
        "eps_alive": eps_alive,
        "band_exists": bool(band_exists),
        "band_width": float(max(0.0, eps_max - eps_alive)),
        "max_change_in_band": change_in_band,
        "change_at_eps_max": float(np.interp(eps_max, eps_grid, change)),
    }


# --------------------------------------------------------------------------- #
# box vs ti=1 支配の分離計測 (red-team F1 の数値確認)
# --------------------------------------------------------------------------- #
def ti1_dominance(rng, n, n_genes=400, max_input_abs=1.0, w_scale=0.35):
    V = np.eye(n)
    rows_total = 0
    rows_ti1 = 0
    for _ in range(n_genes):
        decay = rng.uniform(0.0, 1.0, size=n)
        W = rng.normal(0.0, w_scale, size=(n, n)) / np.sqrt(n)
        t_lo = _t_min(decay, W, V, max_input_abs)
        lo = row_bounds_at(decay, W, t_lo, "lo")
        hi = row_bounds_at(decay, W, t_lo, "hi")
        rows_total += n
        rows_ti1 += int(np.sum(hi >= lo))  # ti=1 が sup を与える row 数
    return rows_ti1 / rows_total if rows_total else float("nan")


def main():
    rng = np.random.default_rng(20260609)
    L, n_inputs = 64, 4
    results = {"meta": {"seed": 20260609, "tau": 0.05, "kernel": "coupled RWKV s'=decay*s+(1-decay)*tanh(Ws+Vx)"},
               "per_n": {}}

    for n in (4, 8, 16):
        Xs = [rng.normal(size=(L, n)) for _ in range(n_inputs)]  # |x|~N(0,1) (max_input_abs=1.0 と整合は別途留保)
        dom = ti1_dominance(rng, n)
        sweeps = []
        n_bases, n_dirs = 60, 4
        got = 0
        while got < n_bases:
            base = sample_admitted_base(rng, n)
            if base is None:
                continue
            decay, W, V, sup0 = base
            headroom = 1.0 - sup0
            for _ in range(n_dirs):
                s = sweep(decay, W, V, rng, Xs)
                s["base_headroom"] = headroom
                s["base_sup"] = sup0
                sweeps.append(s)
            got += 1

        band = [s for s in sweeps if s["band_exists"]]
        frac_band = len(band) / len(sweeps)
        # headroom と band_width の関係 (3 分位)
        hr = np.array([s["base_headroom"] for s in sweeps])
        bw = np.array([s["band_width"] for s in sweeps])
        be = np.array([1.0 if s["band_exists"] else 0.0 for s in sweeps])
        order = np.argsort(hr)
        thirds = np.array_split(order, 3)
        by_headroom = []
        for t in thirds:
            by_headroom.append({
                "headroom_mean": float(hr[t].mean()),
                "frac_band": float(be[t].mean()),
                "band_width_mean": float(bw[t].mean()),
            })
        max_change_band = np.array([s["max_change_in_band"] for s in sweeps if s["band_exists"]])
        results["per_n"][str(n)] = {
            "ti1_dominance_frac": float(dom),
            "n_sweeps": len(sweeps),
            "frac_coviability_band": float(frac_band),
            "eps_max_median": float(np.median([s["eps_max"] for s in sweeps])),
            "eps_alive_median_finite": float(np.median([s["eps_alive"] for s in sweeps if np.isfinite(s["eps_alive"])])) if any(np.isfinite(s["eps_alive"]) for s in sweeps) else None,
            "band_width_median": float(np.median(bw)),
            "max_change_in_band_median": float(np.median(max_change_band)) if max_change_band.size else 0.0,
            "by_headroom_tercile": by_headroom,
        }
        print(f"[n={n}] ti=1 支配={dom:.3f}  両立帯あり={frac_band:.3f}  "
              f"ε_max中央={np.median([s['eps_max'] for s in sweeps]):.3f}  "
              f"band幅中央={np.median(bw):.3f}  帯内max変化中央={(np.median(max_change_band) if max_change_band.size else 0):.3f}")

    out = os.path.join(os.path.dirname(__file__), "phase_m1_coviability_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}")
    return results


if __name__ == "__main__":
    main()
