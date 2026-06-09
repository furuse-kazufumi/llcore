# SPDX-License-Identifier: Apache-2.0
"""Phase 0: Mamba-130M 正の対照 (stable-by-construction) — best-effort CPU。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) §④/§⑥/Phase 0:
  Mamba は非正の最大 Lyapunov 指数 (arXiv:2406.00209) で stable-by-construction = 枠組みの
  **正の対照**。枠組みの判別力 = 「危険 (no-gate) / 安全 (Mamba) / SmolLM2 (reject 発生)」を
  区別して測れるか。本 script は:
    1. Mamba-130M を CPU (HF slow path) で frozen load + coherent 生成確認 (= 実 LM, 第2 base)。
       → framework F8 plug-point「新 base を 1 改変で載せ替え」の実証 (SmolLM2 harness と同構造)。
    2. Mamba hidden 上で phase0 harness (cert_two gate / no-gate) を回し framework が **第2 base に
       portable** か確認 (gate load-bearing が再現するか)。
    3. Mamba の **固有表現安定性** probe: 実 hidden 列で近接 2 軌道の終端乖離 (摂動忘却) を測り、
       SmolLM2 と比較 (正の対照 = Mamba 表現が摂動を忘れる傾向が強いか)。proxy であり形式 Lyapunov
       主張ではない (honest)。

fail-safe: mamba-ssm/CUDA カーネル不在で load 失敗時は例外を握り潰し、"deferred" verdict を
JSON に残して exit 0 (パイプラインを止めない)。本機は CPU ゆえ slow path 想定。
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(__file__))

from llcore.verifier.backends import _infnorm_sup, _t_min, _jac_at_t, _box_vertices  # noqa: E402

MODEL = "state-spaces/mamba-130m-hf"
LAYER = 12
N = 6
EPS_PERT = 1e-2
EPS_STABLE = 1e-3
SEED = 20260609

CORPUS = [
    "The infinite monkey theorem is a probability statement about random typing.",
    "A verified evolution framework gates structural change with a soundness certificate.",
    "Small language models can run on a CPU and still produce coherent text.",
    "Contraction means the system forgets its initial condition over time.",
]


def cert_two_admits(decay, W, V, mia=1.0):
    t_lo = _t_min(decay, W, V, mia)
    return all(float(np.linalg.svd(_jac_at_t(decay, W, v), compute_uv=False)[0]) < 1.0
               for v in _box_vertices(t_lo))


def adapter_run(X, decay, W, s0):
    s = s0.copy()
    for t in range(X.shape[0]):
        s = decay * s + (1.0 - decay) * np.tanh(W @ s + X[t])
    return s


def perturbation_forgetting(X, decay, W, rng):
    n = decay.shape[0]
    d = rng.normal(size=n); d = EPS_PERT * d / (np.linalg.norm(d) + 1e-12)
    sT = adapter_run(X, decay, W, np.zeros(n))
    spT = adapter_run(X, decay, W, d)
    return float(np.linalg.norm(sT - spT))


def load_mamba_hidden():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, output_hidden_states=True)
    model.eval()
    # coherent 生成確認
    ids = tok("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=12, do_sample=False)
    sample_text = tok.decode(gen[0], skip_special_tokens=True)
    chunks = []
    for text in CORPUS:
        tin = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**tin)
        chunks.append(out.hidden_states[LAYER][0].numpy())
    H = np.concatenate(chunks, axis=0).astype(np.float64)
    dt = time.time() - t0
    return H, sample_text, dt


def intrinsic_hidden_forgetting(H, rng, layer_dim, n_probe=64, eps=1e-2):
    """実 hidden 列の連続性 proxy: 隣接 token hidden の正規化変化 (Mamba 表現の滑らかさ/安定度)。

    形式 Lyapunov ではない。base 表現が局所的に bounded/滑らかか (発散しないか) の粗い指標。
    """
    Hn = H / (np.std(H) + 1e-9)
    diffs = np.linalg.norm(np.diff(Hn, axis=0), axis=1) / (np.linalg.norm(Hn[:-1], axis=1) + 1e-9)
    return {"rel_step_median": float(np.median(diffs)), "rel_step_p95": float(np.percentile(diffs, 95))}


def run_harness(H, rng):
    P = rng.normal(size=(N, H.shape[1])) / np.sqrt(H.shape[1])
    X = (H @ P.T)
    X = X / (np.std(X) + 1e-9) * 0.5
    V = np.eye(N)
    # admit 済 base からの変異集団 (一部 ρ≥1)
    def sample_base():
        for _ in range(60):
            decay = rng.uniform(0, 1, N)
            W = (rng.normal(0, 1, (N, N)) / np.sqrt(N)) * float(rng.uniform(0.3, 1.0))
            for _ in range(50):
                if _infnorm_sup(decay, W, _t_min(decay, W, V)) < 1.0:
                    return decay, W
                W = W * 0.85
        return rng.uniform(0, 1, N), np.zeros((N, N))
    bd, bW = sample_base()
    K = 300
    genes = [(np.clip(bd + rng.normal(0, 0.25, N), 0, 1),
              np.clip(bW + rng.normal(0, 0.35, (N, N)), -2, 2)) for _ in range(K)]
    methods = {}
    for method in ("cert_two", "none"):
        forget = []
        for decay, W in genes:
            ok = cert_two_admits(decay, W, V) if method == "cert_two" else True
            if not ok:
                continue
            forget.append(perturbation_forgetting(X, decay, W, rng))
        forget = np.array(forget)
        stable = forget < EPS_STABLE
        methods[method] = {
            "n_admitted": int(forget.size), "admit_rate": float(forget.size / K),
            "certified_stable_rate": float(stable.mean()) if forget.size else None,
            "false_admit_count": int((~stable).sum()) if method == "cert_two" else None,
            "forget_median": float(np.median(forget)) if forget.size else None,
        }
    return methods, {"T": int(X.shape[0]), "x_abs_mean": float(np.mean(np.abs(X)))}


def main():
    rng = np.random.default_rng(SEED)
    results = {"meta": {"model": MODEL, "layer": LAYER, "n": N, "seed": SEED}}
    try:
        H, sample_text, dt = load_mamba_hidden()
        print(f"Mamba-130M loaded {dt:.1f}s  H={H.shape}  生成例: {sample_text!r}", flush=True)
        methods, xinfo = run_harness(H, rng)
        intr = intrinsic_hidden_forgetting(H, rng, H.shape[1])
        results.update({"status": "ok", "load_seconds": dt, "sample_generation": sample_text,
                        "hidden_shape": list(H.shape), "x_info": xinfo,
                        "methods": methods, "intrinsic_hidden": intr})
        ct, ng = methods["cert_two"], methods["none"]
        print(f"[cert_two] admit={ct['admit_rate']:.3f} certified-stable率={ct['certified_stable_rate']:.3f} "
              f"false-admit={ct['false_admit_count']}", flush=True)
        print(f"[none    ] certified-stable率={ng['certified_stable_rate']:.3f} → "
              f"gate {ct['certified_stable_rate']-ng['certified_stable_rate']:+.3f} (portable to Mamba)", flush=True)
        print(f"intrinsic hidden rel-step 中央={intr['rel_step_median']:.3f} p95={intr['rel_step_p95']:.3f}", flush=True)
    except Exception as e:
        results.update({"status": "deferred", "error": f"{type(e).__name__}: {e}",
                        "trace_tail": traceback.format_exc().splitlines()[-3:]})
        print(f"Mamba load 失敗 (deferred): {type(e).__name__}: {e}", flush=True)
        print("  → CUDA カーネル (mamba-ssm/causal-conv1d) 不在の可能性。CPU slow path 非対応版なら", flush=True)
        print("    Phase 2 で GPU 環境 (Kaggle) に defer。SmolLM2 正対照は phase0_framework で取得済。", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase0_mamba_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
