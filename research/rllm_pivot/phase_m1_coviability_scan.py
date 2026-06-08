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

from llcore.verifier.backends import _infnorm_sup, _t_min, _jac_at_t, _box_vertices  # noqa: E402


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


def cert_two_admits(decay, W, V, max_input_abs=1.0) -> bool:
    """cert_two: achievable-t box 全頂点で σ_max(J)<1 (n<=8 で feasible; 2^n 頂点 SVD)。"""
    t_lo = _t_min(decay, W, V, max_input_abs)
    return all(float(np.linalg.svd(_jac_at_t(decay, W, v), compute_uv=False)[0]) < 1.0
               for v in _box_vertices(t_lo))


def cert_b2_admits(decay, W, V, max_input_abs=1.0) -> bool:
    """vertex-free B2 = σ_max(|M| + R) < 1 (paper §8.3, 単一 SVD, 2^n 列挙なし=高次元 feasible)。

    M = J(t_mid)(box 中点), R_ij = (1-decay_i)·((1-t_lo_i)/2)·|W_ij| (entrywise 半幅)。
    box 上 sup σ_max(J) の sound 上界 (|J| ≤ |M|+R entrywise ∧ σ_max は非負支配で単調)。
    cert_two の admit set ⊆ B2 (sound, 0 false-admit)。体系化: n=16 で cert_inf に収束(navigability 喪失)。
    """
    t_lo = _t_min(decay, W, V, max_input_abs)
    t_mid = 0.5 * (t_lo + 1.0)
    M = _jac_at_t(decay, W, t_mid)
    R = ((1.0 - decay) * ((1.0 - t_lo) / 2.0))[:, None] * np.abs(W)
    return bool(float(np.linalg.svd(np.abs(M) + R, compute_uv=False)[0]) < 1.0)


def _band_metrics(sound, change, eps_grid, tau):
    """sound(bool array) と change(float array) から両立帯メトリクスを計算。"""
    eps_max = 0.0
    for k, eps in enumerate(eps_grid):
        if sound[k]:
            eps_max = eps
        else:
            break
    alive_idx = np.where(change >= tau)[0]
    eps_alive = float(eps_grid[alive_idx[0]]) if alive_idx.size else float("inf")
    band_exists = eps_alive < eps_max
    if band_exists:
        in_band = (eps_grid >= eps_alive) & (eps_grid <= eps_max)
        max_change_in_band = float(change[in_band].max())
    else:
        max_change_in_band = 0.0
    return {
        "eps_max": float(eps_max),
        "eps_alive": eps_alive,
        "band_exists": bool(band_exists),
        "band_width": float(max(0.0, eps_max - eps_alive)),
        "max_change_in_band": max_change_in_band,
        "change_at_eps_max": float(np.interp(eps_max, eps_grid, change)),
    }


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
def grow_one(decay, W, V, rng, eps, in_dir, out_dir, self_w, new_decay, mode="fresh", k=0):
    """width_grow: n -> n+1。

    mode="fresh"   : 新 unit の incoming/outgoing を両方 ε でスケール (新 state が O(ε)
                     → 既存出力への影響 O(ε^2)。素朴な「無から新 unit 追加」)。
    mode="net2net" : 計画の実スキーム近似。新 unit が活性既存 unit k の incoming を copy
                     (新 state s_n が O(1) に駆動される)→ outgoing のみ ε で摂動
                     (関数変化 O(ε)。Net2Net function-preserving の精神: 既存活性 unit を複製)。
    """
    n = decay.shape[0]
    decay2 = np.empty(n + 1)
    decay2[:n] = decay
    W2 = np.zeros((n + 1, n + 1))
    W2[:n, :n] = W
    if mode == "fresh":
        decay2[n] = new_decay
        W2[:n, n] = eps * out_dir          # outgoing (既存 row の off 増)
        W2[n, :n] = eps * in_dir           # incoming (新 state は ε で駆動 → O(ε))
        W2[n, n] = self_w
    elif mode == "net2net":
        decay2[n] = decay[k]               # 活性 unit k の timescale を継承
        W2[n, :n] = W[k, :n].copy()        # incoming を copy → s_n は s_k と同オーダー O(1)
        W2[n, n] = 0.0                      # 新 self は 0 から
        W2[:n, n] = eps * out_dir          # outgoing のみ ε 摂動 (既存への影響 O(ε))
    else:
        raise ValueError(mode)
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
def sweep(decay, W, V, rng, Xs, max_input_abs=1.0, tau=0.05, n_eps=40, eps_hi=3.0, mode="fresh", do_two=True):
    n = decay.shape[0]
    in_dir = rng.normal(size=n); in_dir /= (np.linalg.norm(in_dir) + 1e-12)
    out_dir = rng.normal(size=n); out_dir /= (np.linalg.norm(out_dir) + 1e-12)
    self_w = float(rng.uniform(-0.5, 0.5))
    new_decay = float(rng.uniform(0.0, 1.0))
    kunit = int(np.argmax(np.abs(W).sum(axis=1)))  # net2net で copy する活性既存 unit

    eps_grid = np.linspace(0.0, eps_hi, n_eps)
    sound_inf = np.zeros(n_eps, dtype=bool)
    sound_b2 = np.zeros(n_eps, dtype=bool)
    sound_two = np.zeros(n_eps, dtype=bool)
    change = np.zeros(n_eps)
    for ke, eps in enumerate(eps_grid):
        d2, W2, V2 = grow_one(decay, W, V, rng, eps, in_dir, out_dir, self_w, new_decay, mode=mode, k=kunit)
        sound_inf[ke] = cert_inf_sup(d2, W2, V2, max_input_abs) < 1.0
        sound_b2[ke] = cert_b2_admits(d2, W2, V2, max_input_abs)   # vertex-free, 高次元 feasible
        if do_two:
            sound_two[ke] = cert_two_admits(d2, W2, V2, max_input_abs)
        change[ke] = function_change(decay, W, V, d2, W2, V2, Xs)

    out = {"inf": _band_metrics(sound_inf, change, eps_grid, tau),
           "b2": _band_metrics(sound_b2, change, eps_grid, tau)}
    out["two"] = _band_metrics(sound_two, change, eps_grid, tau) if do_two else None
    return out


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
    L, n_inputs = 48, 3
    results = {"meta": {"seed": 20260609, "tau": 0.05, "kernel": "coupled RWKV s'=decay*s+(1-decay)*tanh(Ws+Vx)"},
               "per_n": {}}

    for n in (4, 8, 16):
        print(f"[n={n}] 開始...", flush=True)
        Xs = [rng.normal(size=(L, n)) for _ in range(n_inputs)]  # |x|~N(0,1) (max_input_abs=1.0 と整合は別途留保)
        dom = ti1_dominance(rng, n, n_genes=300)
        n_bases, n_dirs = 40, 3
        modes = ("fresh", "net2net")
        do_two = (n <= 8)  # cert_two は 2^n 頂点ゆえ n<=8 のみ feasible
        sweeps_by_mode = {m: [] for m in modes}
        got, attempts = 0, 0
        while got < n_bases and attempts < n_bases * 10:
            attempts += 1
            base = sample_admitted_base(rng, n)
            if base is None:
                continue
            decay, W, V, sup0 = base
            for _ in range(n_dirs):
                for m in modes:
                    sweeps_by_mode[m].append(sweep(decay, W, V, rng, Xs, mode=m, do_two=do_two))
            got += 1

        per_mode = {}
        for m in modes:
            sweeps = sweeps_by_mode[m]
            certs = ("inf", "two") if do_two else ("inf",)
            per_cert = {}
            for cert in certs:
                metr = [s[cert] for s in sweeps]
                frac_band = sum(1 for x in metr if x["band_exists"]) / len(metr)
                bw = np.array([x["band_width"] for x in metr])
                ce = np.array([x["change_at_eps_max"] for x in metr])
                eps_alive_med = (float(np.median([x["eps_alive"] for x in metr if np.isfinite(x["eps_alive"])]))
                                 if any(np.isfinite(x["eps_alive"]) for x in metr) else None)
                per_cert[cert] = {
                    "frac_coviability_band": float(frac_band),
                    "eps_max_median": float(np.median([x["eps_max"] for x in metr])),
                    "eps_alive_median_finite": eps_alive_med,
                    "band_width_median": float(np.median(bw)),
                    "change_at_eps_max_median": float(np.median(ce)),
                }
                print(f"[n={n}][{m:8s}][{cert}] 両立帯={frac_band:.3f}  ε_max中央={np.median([x['eps_max'] for x in metr]):.3f}  "
                      f"ε_max時変化中央={np.median(ce):.4f}  band幅={np.median(bw):.3f}", flush=True)
            per_mode[m] = per_cert
        results["per_n"][str(n)] = {"ti1_dominance_frac": float(dom), "by_mode": per_mode}

    out = os.path.join(os.path.dirname(__file__), "phase_m1_coviability_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}")
    return results


if __name__ == "__main__":
    main()
