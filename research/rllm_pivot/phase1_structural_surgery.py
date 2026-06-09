# SPDX-License-Identifier: Apache-2.0
"""Phase 1.2: 実構造手術 ``width_grow`` (CoupledNDGene n→n+1) + 成長 gate。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) §7.2 / Phase 1 step 2 の核。

なぜ新規実装が要るか (一次照合済):
  ``src/llcore/verifier/changeop.py`` の ChangeOp は **scalar 3 float (decay/mix/gate_str)**
  のみを操作し ``kernel_swap_mock`` は明記の通り mock。CoupledNDGene の n 次元構造
  (decay∈[0,1]^n, W∈[-2,2]^{n×n}) には一切触れない。本 module は計画が要求する
  「**実** 構造手術 (width n→n+1)」を CoupledNDGene 上で行い、成長操作下の soundness
  gate (F1 訂正済 per-row 不変条件 / cert_two / cert_sdp) を提供する。

設計 (Phase −1 確定の small-n per-component 制約に従う):
  - 進化させるのは **離散トポロジー自由度 = unit 数 (width)**。連続 (decay,W) は gradient 領域。
  - ``width_grow``: 既存 n unit に 1 unit 追加 (n→n+1)。``mode``:
      * ``"fresh"``   — 無から新 unit。incoming/outgoing を ε でスケール (素朴追加)。
      * ``"net2net"`` — Net2Net (arXiv:1511.05641) function-preserving の精神。活性既存
        unit k を複製: 新 unit が k の incoming row + timescale を継承 (s_n が O(1) 駆動)
        → outgoing のみ ε 摂動 (既存出力への影響 O(ε))。Phase −1 verdict が「真 net2net
        は過小評価寄り」と留保した点を、incoming row + decay 継承で改善。
  - 成長 gate (3 種、navigability×scalability トレードオフを構造手術レベルで体現):
      * ``per_row_growth_ok``  — F1 訂正済 cheap-sound gate。新 column 追加後、**各既存 row i**
        の ``ti=1`` abs-sum bound が 1 未満に留まるか (= 新 column が off-sum を越境させない)。
        cert_inf の per-row 端点と同一数学。O(n) で scale するが Phase −1 通り band は trivial。
      * ``cert_two_ok`` / ``cert_sdp_ok`` — navigable だが 2^n 頂点 (small-n 限定)。

honest:
  - net2net は真の重み分割 (downstream split) ではなく incoming-copy 近似。exact function
    preservation は外部入力駆動 unit では成立しない (新 unit に x_n が無いため)。本 module は
    「既存出力への影響を O(ε) に抑えた構造手術」を測る目的に十分な近似で、verdict に留保明記。
  - gate は **安定 certificate のみで篩い、fitness で篩わない** (homeostatic constraint)。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# coupled_nd (CoupledNDGene + cert_inf/two/sdp + empirical_rho + jacobian) を再利用。
_SDP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verified_evolution_sdp_gate"))
if _SDP not in sys.path:
    sys.path.insert(0, _SDP)

import coupled_nd as C  # noqa: E402


# --------------------------------------------------------------------------- #
# forward (既存出力軌跡比較用; coupled_nd.step と整合)
# --------------------------------------------------------------------------- #
def run_states(g: C.CoupledNDGene, X: np.ndarray) -> np.ndarray:
    """s_{t+1} = decay⊙s + (1-decay)⊙tanh(W s + V x)。X:(L,n) → states:(L,n)。"""
    s = np.zeros(g.n)
    out = np.empty((X.shape[0], g.n))
    for t in range(X.shape[0]):
        s = C.step(g, s, X[t])
        out[t] = s
    return out


# --------------------------------------------------------------------------- #
# width_grow: 実構造手術 n → n+1
# --------------------------------------------------------------------------- #
def width_grow(
    g: C.CoupledNDGene,
    *,
    eps: float,
    in_dir: np.ndarray,
    out_dir: np.ndarray,
    self_w: float,
    new_decay: float,
    mode: str = "net2net",
    k: int | None = None,
) -> C.CoupledNDGene:
    """CoupledNDGene を n→n+1 に成長させた新 gene を返す (純関数)。

    Parameters
    ----------
    eps : float
        新 unit の結合強度スケール (両立帯 sweep の制御変数)。
    in_dir, out_dir : (n,)
        新 unit の incoming (row n) / outgoing (col n) 方向 (単位ベクトル想定)。
    self_w : float
        新 unit の自己再帰重み W[n,n] (fresh のみ; net2net は 0 から)。
    new_decay : float
        新 unit の decay (fresh のみ; net2net は decay[k] を継承)。
    mode : {"fresh","net2net"}
    k : int | None
        net2net で複製する活性既存 unit。None なら abs-row-sum 最大の unit。
    """
    decay, W, V = g.decay, g.W, g.V
    n = g.n
    decay2 = np.empty(n + 1)
    decay2[:n] = decay
    W2 = np.zeros((n + 1, n + 1))
    W2[:n, :n] = W
    if mode == "fresh":
        decay2[n] = new_decay
        W2[:n, n] = eps * out_dir       # outgoing: 既存 row の off-sum を増やす (F1 脅威)
        W2[n, :n] = eps * in_dir        # incoming: 新 state は O(ε) 駆動
        W2[n, n] = self_w
    elif mode == "net2net":
        kk = int(np.argmax(np.abs(W).sum(axis=1))) if k is None else int(k)
        decay2[n] = decay[kk]           # 活性 unit k の timescale 継承
        W2[n, :n] = W[kk, :n].copy()    # incoming row を copy → s_n が s_k と同オーダー O(1)
        W2[n, n] = 0.0
        W2[:n, n] = eps * out_dir       # outgoing のみ ε 摂動 (既存への影響 O(ε))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return C.CoupledNDGene.make(decay=decay2, W=W2)


def function_change(g: C.CoupledNDGene, grown: C.CoupledNDGene, Xs: list[np.ndarray]) -> float:
    """新 unit 追加で既存出力 (先頭 n 次元) 軌跡が相対 L2 でどれだけ変わるか (入力平均)。

    新 unit には外部入力を与えない (Xg = [X, 0])。grown[:, :n] を base と比較。
    """
    n = g.n
    rels = []
    for X in Xs:
        base = run_states(g, X)                                    # (L, n)
        Xg = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        grown_states = run_states(grown, Xg)[:, :n]
        num = np.linalg.norm(grown_states - base)
        den = np.linalg.norm(base) + 1e-12
        rels.append(num / den)
    return float(np.mean(rels))


# --------------------------------------------------------------------------- #
# 成長 gate
# --------------------------------------------------------------------------- #
def per_row_ti1_bound(g: C.CoupledNDGene, rows: range | list[int] | None = None) -> np.ndarray:
    """各 row i の ``ti=1`` 端点 abs-sum bound = |d_i+(1-d_i)W_ii| + (1-d_i)Σ_{j≠i}|W_ij|。

    F1 訂正: ``_infnorm_sup`` の sup は 96-100% の row で ti=1 端点が支配 (Phase −1 実測)。
    width_grow の真の脅威 = 新 column が既存 row の off-sum を増やし ti=1 bound を 1 超させる。
    """
    decay, W = g.decay, g.W
    n = g.n
    idx = range(n) if rows is None else rows
    res = []
    for i in idx:
        off = sum(abs(W[i, j]) for j in range(n) if j != i)
        diag = abs(decay[i] + (1.0 - decay[i]) * W[i, i])
        res.append(diag + (1.0 - decay[i]) * off)
    return np.array(res)


def per_row_growth_ok(grown: C.CoupledNDGene, original_n: int) -> bool:
    """F1 訂正済 cheap-sound per-row 不変 gate: **既存 row** が成長後も ti=1 bound<1 を保つ。

    O(n) で scale (頂点列挙なし)。新 row (index original_n) も含めた全 row<1 を要求すると
    cert_inf と一致するが、本 gate は計画 §7.2 の「既存行 abs-sum 不変条件」を直接表現
    (既存 row のみ検査)。soundness は別途 empirical_rho で検証する。
    """
    existing = per_row_ti1_bound(grown, rows=list(range(original_n)))
    new_row = per_row_ti1_bound(grown, rows=[original_n])
    return bool(existing.max() < 1.0 and new_row.max() < 1.0)


def cert_inf_ok(grown: C.CoupledNDGene, mia: float = 1.0) -> bool:
    return bool(C.cert_inf(grown, mia))


def cert_two_ok(grown: C.CoupledNDGene, mia: float = 1.0) -> bool:
    return bool(C.cert_two(grown, mia))


def cert_sdp_ok(grown: C.CoupledNDGene, mia: float = 1.0) -> bool:
    return bool(C.cert_sdp(grown, mia))


# --------------------------------------------------------------------------- #
# base sampler (cert_inf PASS な admit 済 gene; 任意 n で確実に到達)
# --------------------------------------------------------------------------- #
@dataclass
class GrowDirections:
    in_dir: np.ndarray
    out_dir: np.ndarray
    self_w: float
    new_decay: float
    k: int


def sample_admitted_base(rng: np.random.Generator, n: int, mia: float = 1.0, max_tries: int = 60):
    """W を 0.85 倍縮小しながら cert_inf PASS な (decay, W, V=I) gene を返す。失敗時 None。"""
    for _ in range(max_tries):
        decay = rng.uniform(0.0, 1.0, size=n)
        W = (rng.normal(0.0, 1.0, size=(n, n)) / np.sqrt(n)) * float(rng.uniform(0.2, 0.9))
        for _ in range(50):
            g = C.CoupledNDGene.make(decay=decay, W=W)
            if C.cert_inf(g, mia):
                return g
            W = W * 0.85
    return None


def random_directions(rng: np.random.Generator, g: C.CoupledNDGene) -> GrowDirections:
    n = g.n
    in_dir = rng.normal(size=n); in_dir /= (np.linalg.norm(in_dir) + 1e-12)
    out_dir = rng.normal(size=n); out_dir /= (np.linalg.norm(out_dir) + 1e-12)
    return GrowDirections(
        in_dir=in_dir, out_dir=out_dir,
        self_w=float(rng.uniform(-0.5, 0.5)), new_decay=float(rng.uniform(0.0, 1.0)),
        k=int(np.argmax(np.abs(g.W).sum(axis=1))),
    )


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(20260609)
    g = sample_admitted_base(rng, 4)
    assert g is not None and C.cert_inf(g)
    d = random_directions(rng, g)
    Xs = [rng.normal(size=(32, 4)) for _ in range(3)]
    print(f"base n={g.n} cert_inf={C.cert_inf(g)} emp_rho={C.empirical_rho(g, n_samples=500):.3f}")
    for mode in ("fresh", "net2net"):
        for eps in (0.0, 0.5, 1.5, 3.0):
            grown = width_grow(g, eps=eps, in_dir=d.in_dir, out_dir=d.out_dir,
                               self_w=d.self_w, new_decay=d.new_decay, mode=mode, k=d.k)
            assert grown.n == g.n + 1
            ch = function_change(g, grown, Xs)
            print(f"[{mode:8s}] eps={eps:.2f} n→{grown.n} "
                  f"per_row_ok={per_row_growth_ok(grown, g.n)} cert_inf={cert_inf_ok(grown)} "
                  f"cert_two={cert_two_ok(grown)} cert_sdp={cert_sdp_ok(grown)} "
                  f"emp_rho={C.empirical_rho(grown, n_samples=400):.3f} Δfunc={ch:.4f}")
    print("self-test OK")
