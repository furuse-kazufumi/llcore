# SPDX-License-Identifier: Apache-2.0
"""Phase 0: Verified-Plasticity Evaluation Framework — 実機 SmolLM2-135M 上の最小動作 harness。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) の主軸 = 評価枠組み。本 harness はその**最初の実機 deliverable**:
実在小型 LLM(SmolLM2-135M, Apache-2.0)の hidden state 列に **small-n(≤6)verified recurrent adapter**
(CoupledNDGene)を載せ、構造/パラメータ変異を **cert_two sound gate** で篩い、framework の第一級メトリクス
(**certified-stable rate / perturbation 忘却**)を method 比較(cert_two gate / no-gate)で測る。

honest 設計:
- adapter kernel `s' = decay⊙s + (1-decay)⊙tanh(Ws+Vx)` は state を常に有界に保つ(convex+tanh)。
  ∴ 安定性 = 「発散」でなく **収縮(ρ<1)= echo-state property = 摂動された初期条件を忘れる**こと。
  測定 = 近接 2 軌道(s0=0 と s0=δ)を実 hidden state 列で回し、終端 |s_T − s'_T|(摂動忘却度)。
- verified core は実 LLM の高次元 weight でなく **n≤6 の低次元 summary**(固定ランダム射影 576→n)に掛かる
  = Phase −1 で確定した small-n per-component 限定の設計制約に従う。
- cert_two(sound 緩和, n≤6 で 2^n feasible)を gate に(Phase −1: cert_inf は band を開かない / cert_two が navigable)。

第一級メトリクス:
- **certified-stable rate**: cert_two が admit した gene が実データで全て収束(摂動忘却 < ε)か = 0 false-admit。
- **gate load-bearing**: no-gate(全 admit)は ρ≥1 の摂動敏感 gene を通すか = gate ありと比べ stable-rate が下がるか。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llcore.verifier.backends import _infnorm_sup, _t_min, _jac_at_t, _box_vertices  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"
LAYER = 15      # 中間層の hidden state を使う
N = 6           # verified adapter の低次元 (small-n per-component, Phase −1 確定)
EPS_PERT = 1e-2  # 初期摂動
EPS_STABLE = 1e-3  # 終端 |s_T - s'_T| < これ で「収縮=摂動忘却」と判定
SEED = 20260609


def cert_inf_sup(decay, W, V, mia=1.0):
    return _infnorm_sup(decay, W, _t_min(decay, W, V, mia))


def cert_two_admits(decay, W, V, mia=1.0):
    t_lo = _t_min(decay, W, V, mia)
    return all(float(np.linalg.svd(_jac_at_t(decay, W, v), compute_uv=False)[0]) < 1.0
               for v in _box_vertices(t_lo))


def adapter_run(X, decay, W, s0):
    """s' = decay⊙s + (1-decay)⊙tanh(W s + x)。X:(T,n)(=射影済 hidden, V=I で入力). 終端 state を返す。"""
    s = s0.copy()
    for t in range(X.shape[0]):
        s = decay * s + (1.0 - decay) * np.tanh(W @ s + X[t])
    return s


def perturbation_forgetting(X, decay, W, rng):
    """近接 2 軌道(s0=0, s0=δ)を実 hidden 列で回し終端乖離 |s_T - s'_T| を返す(小=収縮=摂動忘却)。"""
    n = decay.shape[0]
    d = rng.normal(size=n); d = EPS_PERT * d / (np.linalg.norm(d) + 1e-12)
    sT = adapter_run(X, decay, W, np.zeros(n))
    spT = adapter_run(X, decay, W, d)
    return float(np.linalg.norm(sT - spT))


def get_real_hidden_sequence():
    """SmolLM2-135M を frozen load し、コーパスの LAYER hidden state を連結して (T,576) を返す。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, output_hidden_states=True)
    model.eval()
    corpus = [
        "The infinite monkey theorem is a probability statement about random typing.",
        "A verified evolution framework gates structural change with a soundness certificate.",
        "Small language models can run on a CPU and still produce coherent text.",
        "Contraction means the system forgets its initial condition over time.",
    ]
    chunks = []
    for text in corpus:
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids)
        chunks.append(out.hidden_states[LAYER][0].numpy())  # (seq, 576)
    H = np.concatenate(chunks, axis=0).astype(np.float64)    # (T, 576)
    print(f"SmolLM2-135M loaded + hidden 抽出 {time.time()-t0:.1f}s  H={H.shape} (layer {LAYER})", flush=True)
    return H


def sample_base(rng, n, mia=1.0, max_tries=60):
    V = np.eye(n)
    for _ in range(max_tries):
        decay = rng.uniform(0.0, 1.0, size=n)
        W = (rng.normal(0.0, 1.0, size=(n, n)) / np.sqrt(n)) * float(rng.uniform(0.3, 1.0))
        for _ in range(50):
            if cert_inf_sup(decay, W, V, mia) < 1.0:
                return decay, W
            W = W * 0.85
    return None


def main():
    rng = np.random.default_rng(SEED)
    H = get_real_hidden_sequence()  # (T, 576)

    # 固定ランダム射影 576 -> n、列を単位スケールに正規化(入力 |x| を程よく)
    P = rng.normal(size=(N, H.shape[1])) / np.sqrt(H.shape[1])
    X = (H @ P.T)                            # (T, n) = 実 LLM hidden の低次元 summary
    X = X / (np.std(X) + 1e-9) * 0.5         # 入力スケールを揃える(max_input_abs と整合は留保)
    print(f"射影後 X={X.shape}  |x| mean={np.mean(np.abs(X)):.3f} max={np.max(np.abs(X)):.3f}", flush=True)

    # 候補 gene 集団: admit 済 base からの変異(一部は ρ≥1 になる=gate の出番)
    V = np.eye(N)
    K = 300
    genes = []
    base = sample_base(rng, N)
    bd, bW = base
    for _ in range(K):
        decay = np.clip(bd + rng.normal(0, 0.25, size=N), 0, 1)
        W = np.clip(bW + rng.normal(0, 0.35, size=(N, N)), -2, 2)
        genes.append((decay, W))

    # method 比較: cert_two gate / no-gate。admit 済 gene の実データ摂動忘却を測る。
    results = {"meta": {"model": MODEL, "layer": LAYER, "n": N, "T": int(X.shape[0]),
                        "eps_stable": EPS_STABLE, "K": K, "seed": SEED}, "methods": {}}
    for method in ("cert_two", "none"):
        admitted = []
        forget = []
        for decay, W in genes:
            ok = cert_two_admits(decay, W, V) if method == "cert_two" else True
            if not ok:
                continue
            f = perturbation_forgetting(X, decay, W, rng)
            admitted.append((decay, W))
            forget.append(f)
        forget = np.array(forget)
        stable = forget < EPS_STABLE
        results["methods"][method] = {
            "n_admitted": len(admitted),
            "admit_rate": len(admitted) / K,
            "certified_stable_rate": float(stable.mean()) if forget.size else None,
            "false_admit_count": int((~stable).sum()) if method == "cert_two" else None,  # cert_two が摂動敏感を admit した数
            "forget_median": float(np.median(forget)) if forget.size else None,
            "forget_p95": float(np.percentile(forget, 95)) if forget.size else None,
            "forget_max": float(forget.max()) if forget.size else None,
        }
        m = results["methods"][method]
        print(f"[{method:8s}] admit={m['admit_rate']:.3f}  certified-stable率={m['certified_stable_rate']:.3f}  "
              f"false-admit={m['false_admit_count']}  忘却中央={m['forget_median']:.2e}  p95={m['forget_p95']:.2e}  max={m['forget_max']:.2e}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase0_framework_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    # framework verdict
    ct = results["methods"]["cert_two"]
    ng = results["methods"]["none"]
    print("\n=== framework verdict ===", flush=True)
    print(f"cert_two gate: 実 SmolLM2 hidden 上で certified-stable率={ct['certified_stable_rate']:.3f}, "
          f"false-admit={ct['false_admit_count']} (0 なら sound)", flush=True)
    print(f"no-gate: certified-stable率={ng['certified_stable_rate']:.3f} "
          f"→ gate が {ct['certified_stable_rate']-ng['certified_stable_rate']:+.3f} 改善 = 実 LLM 上で gate load-bearing", flush=True)


if __name__ == "__main__":
    main()
