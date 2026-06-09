# SPDX-License-Identifier: Apache-2.0
"""Phase 2: Mamba-130M 固有安定性 正の対照 — SSM Jacobian Lyapunov (stable-by-construction)。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) §④(L71) / North Star #3 / PHASE_2_VERDICT.md §5 より:
  正の対照 = Mamba-130M。SSM 状態遷移行列の最大 Lyapunov 指数が**非正**(arXiv:2406.00209)で
  収縮 certificate が**自明に PASS**。枠組みの base-level 判別力を示す:
    「Mamba は固有安定で自明 PASS / SmolLM2 は固有の状態再帰安定性を持たず(標準 Transformer)、
     安定性は adapter+gate で初めて課される」。

Phase 1 ではこの正対照は defer(Phase 1 の Mamba は adapter 掛け・弱オラクル・admit n=7)。
本 script は **Mamba 自身の SSM 再帰の固有安定性を直接測る**:

  Mamba の selective SSM は連続系 dh/dt = A h + B x(A は対角、実部負: A = -exp(A_log))を離散化
  h_t = Ā h_{t-1} + B̄ x_t(Ā = exp(Δ·A), Δ>0)。A の実部 < 0 ⇒ |Ā 対角| < 1 ⇒ 状態再帰の
  spectral radius / 最大 Lyapunov 指数 λ_max = max(Δ·A) ≤ 0(収縮)。これが「stable-by-construction」。

honest 留保(本文 verdict + JSON に明記):
  - Δ は入力依存(selective)。ここでは代表 Δ = softplus(dt_proj.bias)(input-independent baseline)+
    Δ を [time_step_min, time_step_max] 帯および広域でスイープし、Δ>0 の全域で A<0 ⇒ λ_max≤0 を論証。
  - A は対角(state-space を対角パラメタライズ)前提。Mamba の S6 は実際 diagonal-A real-valued。
  - 離散化 = zero-order hold Ā = exp(Δ·A)(transformers MambaMixer の discretize と同形)。
  - これは **SSM 状態再帰の安定性**であって Mamba 全体(conv1d / SiLU / MLP / gating)の Lipschitz ではない。
  - 弱オラクルでない: A_log/dt_bias は実重みから直接抽出した一次データ(closed-form の Ā 対角値)。

成果物: phase2_mamba_lyapunov_results.json(層ごと A 統計 / λ_max(Δ) / 全層安定フラグ / SmolLM2 対比 / verdict)。
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

MAMBA_MODEL = "state-spaces/mamba-130m-hf"
SMOLLM_MODEL = "HuggingFaceTB/SmolLM2-135M"
SEED = 20260609

# Mamba-130m-hf config 由来(probe 実測): dt の妥当域 [time_step_min, time_step_max]
DT_MIN = 1e-3
DT_MAX = 1e-1
# 広域スイープ(softplus(dt_bias) が外れても λ_max≤0 が保たれるかを論証するため広く)
DT_SWEEP = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=np.float64)


def _softplus(x: np.ndarray) -> np.ndarray:
    """numerically-stable softplus = log(1+exp(x))。transformers の dt 活性と同形。"""
    return np.logaddexp(0.0, x)


def extract_mamba_ssm() -> dict:
    """Mamba-130M を frozen load し全 Mamba 層の SSM パラメータ A(=-exp(A_log))と
    代表 Δ(=softplus(dt_proj.bias))を抽出する。

    state_dict キー (probe 実測, transformers MambaForCausalLM):
      backbone.layers.{i}.mixer.A_log     shape (d_inner=1536, state_size=16)  -> A = -exp(A_log) < 0
      backbone.layers.{i}.mixer.dt_proj.bias shape (1536,)                      -> Δ_base = softplus(bias) > 0
      backbone.layers.{i}.mixer.D          shape (1536,)                        -> skip 接続(状態再帰に非関与)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(MAMBA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(MAMBA_MODEL, torch_dtype=torch.float32)
    model.eval()
    sd = model.state_dict()

    n_layer = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
    layers = []
    a_log_keys = sorted(k for k in sd if k.endswith("mixer.A_log"))
    for k in a_log_keys:
        prefix = k[: -len("A_log")]  # "backbone.layers.{i}.mixer."
        layer_idx = int(prefix.split(".layers.")[1].split(".")[0])
        A_log = sd[k].float().numpy()                 # (d_inner, state_size)
        A = -np.exp(A_log)                            # 連続系 A(対角要素), 実部 < 0
        dt_bias_key = prefix + "dt_proj.bias"
        dt_bias = sd[dt_bias_key].float().numpy()     # (d_inner,)
        dt_base = _softplus(dt_bias)                  # 代表 Δ(input-independent baseline), > 0
        layers.append({
            "layer": layer_idx,
            "A": A,                                   # (d_inner, state_size)
            "dt_base": dt_base,                       # (d_inner,)
            "a_log_key": k,
            "a_log_shape": list(A_log.shape),
            "dt_bias_key": dt_bias_key,
            "dt_bias_shape": list(dt_bias.shape),
        })
    layers.sort(key=lambda d: d["layer"])
    dt_ms = (time.time() - t0)
    return {
        "n_layer": n_layer,
        "d_inner": int(getattr(cfg, "intermediate_size", getattr(cfg, "d_inner", 0))),
        "state_size": int(getattr(cfg, "state_size", 0)),
        "time_step_min": float(getattr(cfg, "time_step_min", DT_MIN)),
        "time_step_max": float(getattr(cfg, "time_step_max", DT_MAX)),
        "layers": layers,
        "load_seconds": dt_ms,
        # 後続検証用にキー名/形状(代表 = 層0)を残す
        "key_schema": {
            "A_log": {"key_pattern": "backbone.layers.{i}.mixer.A_log",
                      "shape": layers[0]["a_log_shape"] if layers else None,
                      "meaning": "(d_inner, state_size); A = -exp(A_log) (continuous diagonal state matrix, Re<0)"},
            "dt_bias": {"key_pattern": "backbone.layers.{i}.mixer.dt_proj.bias",
                        "shape": layers[0]["dt_bias_shape"] if layers else None,
                        "meaning": "(d_inner,); Delta_base = softplus(dt_proj.bias) > 0 (representative time-step)"},
            "D": {"key_pattern": "backbone.layers.{i}.mixer.D",
                  "meaning": "(d_inner,); skip connection, NOT part of state recurrence"},
        },
    }


def analyze_layer_stability(A: np.ndarray, dt_base: np.ndarray) -> dict:
    """1 層の SSM 固有安定性。

    A: (d_inner, state_size) 連続系対角要素(実部 < 0 のはず)。
    dt_base: (d_inner,) 代表 Δ(channel ごと)。

    離散 Ā 対角 = exp(Δ·A)。A<0, Δ>0 ⇒ |Ā|<1。Lyapunov λ = Δ·A ≤ 0(連続)/ log|Ā| = Δ·A(離散)。
    各 channel d の Δ_d を全 state-dim に broadcast し Ā = exp(Δ_d · A[d, :])。
    """
    d_inner, state_size = A.shape
    # --- A 統計 ---
    A_min = float(A.min()); A_max = float(A.max()); A_mean = float(A.mean())
    all_A_negative = bool((A < 0).all())
    frac_A_negative = float((A < 0).mean())

    # --- 代表 Δ(softplus(dt_bias)) での離散 Ā 対角と λ ---
    # Δ を channel 次元で broadcast: (d_inner,1) * (d_inner,state) = (d_inner,state)
    DtA = dt_base[:, None] * A                        # = Δ·A,  λ(連続/離散指数) per (channel, state)
    lam_max_base = float(DtA.max())                   # 最大 Lyapunov 指数(全 channel×state)= max(Δ·A)
    abar_base = np.exp(DtA)                            # 離散対角要素 |Ā|(A<0 なので (0,1])
    max_abs_abar_base = float(abar_base.max())        # 状態再帰の spectral radius(対角系)
    stable_base = bool(lam_max_base <= 0.0)           # λ_max ≤ 0 ⇒ 固有安定
    # honest: Δ·A = 0 になるのは A=0(=A_log→-inf, 数値上は到達しない)。<0 が正常。
    dt_base_min = float(dt_base.min()); dt_base_max = float(dt_base.max())

    # --- Δ スイープ: Δ>0 の全域で λ_max ≤ 0(=A<0 ゆえ Δ に依らず安定)を論証 ---
    sweep = []
    for dt in DT_SWEEP:
        lam = float((dt * A).max())                   # max(Δ·A) over all (channel,state)
        sweep.append({"dt": float(dt), "lam_max": lam, "max_abs_abar": float(np.exp(lam)),
                      "stable": bool(lam <= 0.0)})

    return {
        "d_inner": d_inner, "state_size": state_size,
        "A_min": A_min, "A_max": A_max, "A_mean": A_mean,
        "all_A_negative": all_A_negative, "frac_A_negative": frac_A_negative,
        "dt_base_min": dt_base_min, "dt_base_max": dt_base_max,
        "lam_max_base": lam_max_base,                 # 代表 Δ での λ_max
        "max_abs_abar_base": max_abs_abar_base,       # 代表 Δ での max|Ā 対角|(< 1 なら収縮)
        "stable_base": stable_base,
        "dt_sweep": sweep,
        "stable_all_dt": bool(all(s["stable"] for s in sweep)),
    }


def check_smollm_has_no_ssm() -> dict:
    """SmolLM2-135M に SSM 状態再帰 A 行列が無い(標準 Transformer)ことを直接確認。
    固有安定 certificate の概念自体が base に存在しない = 安定性は後付け adapter+gate に依存。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    import re

    cfg = AutoConfig.from_pretrained(SMOLLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(SMOLLM_MODEL, torch_dtype=torch.float32)
    sd = model.state_dict()
    patterns = sorted({re.sub(r"\.\d+\.", ".N.", k) for k in sd})
    ssm_markers = ("a_log", "ssm", "dt_proj", "mixer", "state_proj", "selective")
    ssm_keys = [k for k in sd if any(m in k.lower() for m in ssm_markers)]
    return {
        "model": SMOLLM_MODEL,
        "model_type": str(getattr(cfg, "model_type", "?")),
        "architectures": list(getattr(cfg, "architectures", []) or []),
        "n_unique_key_patterns": len(patterns),
        "key_patterns": patterns,
        "ssm_recurrence_keys": ssm_keys,
        "has_ssm_state_recurrence": bool(ssm_keys),
        "conclusion": (
            "SmolLM2-135M is a standard Transformer (model_type=llama): only self_attn (q/k/v/o) + "
            "mlp (gate/up/down) projections. NO A_log / dt_proj / state-recurrence matrix exists. "
            "The concept of an intrinsic state-recurrence contraction certificate (spectral radius / "
            "Lyapunov of a state-transition matrix) is structurally absent from the base — attention is "
            "a non-recurrent global mixing, not a contractive state recursion. Therefore stability must "
            "be imposed by the bolted-on verified adapter + cert gate (Phase 0 harness), it is NOT "
            "stable-by-construction."
        ),
    }


def main():
    rng = np.random.default_rng(SEED)  # noqa: F841 (determinism anchor; no stochastic step here)
    results = {
        "meta": {
            "mamba_model": MAMBA_MODEL, "smollm_model": SMOLLM_MODEL, "seed": SEED,
            "purpose": ("Phase 2 positive control: measure Mamba SSM state-recurrence intrinsic "
                        "stability (stable-by-construction) via discrete diagonal A-bar = exp(dt*A), "
                        "A = -exp(A_log) < 0  =>  lambda_max = max(dt*A) <= 0. Contrast: SmolLM2 has "
                        "no such SSM A matrix (standard Transformer)."),
            "method": ("A = -exp(A_log) (continuous diagonal state matrix); representative "
                       "Delta = softplus(dt_proj.bias); discrete A-bar = exp(Delta*A) (zero-order hold); "
                       "lambda_max = max over (channel,state) of Delta*A; swept Delta>0 to show "
                       "stability holds for all positive time-steps."),
            "honest_caveats": [
                "Delta is input-dependent (selective SSM). We use representative Delta = softplus(dt_proj.bias) "
                "(input-independent baseline) and sweep Delta over [1e-4 .. 1e2] to show lambda_max<=0 holds "
                "for ALL Delta>0 (since A<0, sign(Delta*A) is independent of Delta>0).",
                "Assumes diagonal A (Mamba S6 is real-valued diagonal in this hf checkpoint: A_log shape "
                "(d_inner, state_size), per-(channel,state) scalar => eigenvalues ARE the diagonal entries).",
                "Discretization = zero-order-hold A-bar = exp(Delta*A), matching transformers MambaMixer.",
                "This is the stability of the SSM STATE RECURRENCE only — NOT the full Mamba block Lipschitz "
                "constant (conv1d, SiLU gating, in/out projections, MLP are excluded).",
                "Not a weak oracle: A_log / dt_proj.bias are first-party weights; A-bar diagonal is closed-form, "
                "no sampling/proxy. lambda_max is exact for the diagonal SSM eigenspectrum.",
            ],
        }
    }

    try:
        # === Mamba SSM 抽出 + 層ごと固有安定性 ===
        ext = extract_mamba_ssm()
        print(f"Mamba-130M loaded {ext['load_seconds']:.1f}s  n_layer={ext['n_layer']}  "
              f"d_inner={ext['d_inner']}  state_size={ext['state_size']}", flush=True)
        print(f"key schema: A_log {ext['key_schema']['A_log']['shape']}  "
              f"dt_bias {ext['key_schema']['dt_bias']['shape']}", flush=True)

        per_layer = []
        for L in ext["layers"]:
            st = analyze_layer_stability(L["A"], L["dt_base"])
            st["layer"] = L["layer"]
            per_layer.append(st)

        # 全層集約
        all_layers_A_neg = all(s["all_A_negative"] for s in per_layer)
        all_layers_stable_base = all(s["stable_base"] for s in per_layer)
        all_layers_stable_all_dt = all(s["stable_all_dt"] for s in per_layer)
        global_lam_max_base = max(s["lam_max_base"] for s in per_layer)
        global_max_abs_abar_base = max(s["max_abs_abar_base"] for s in per_layer)
        global_A_max = max(s["A_max"] for s in per_layer)  # 最も 0 に近い A(最弱の収縮)
        global_A_min = min(s["A_min"] for s in per_layer)

        results["mamba"] = {
            "n_layer": ext["n_layer"], "d_inner": ext["d_inner"], "state_size": ext["state_size"],
            "load_seconds": ext["load_seconds"],
            "time_step_min": ext["time_step_min"], "time_step_max": ext["time_step_max"],
            "key_schema": ext["key_schema"],
            "per_layer": per_layer,
            "summary": {
                "all_layers_A_real_part_negative": all_layers_A_neg,
                "all_layers_stable_at_base_delta": all_layers_stable_base,
                "all_layers_stable_for_all_delta_sweep": all_layers_stable_all_dt,
                "global_lambda_max_base": global_lam_max_base,        # 全層・全 channel での最大 λ(<0 なら固有安定)
                "global_max_abs_abar_base": global_max_abs_abar_base, # 全層での最大 |Ā 対角|(<1 なら全層収縮)
                "global_A_max_real_part": global_A_max,               # 最も弱い収縮(0 に最も近い A)
                "global_A_min_real_part": global_A_min,               # 最も強い収縮
                "intrinsically_stable": bool(all_layers_A_neg and all_layers_stable_base
                                             and all_layers_stable_all_dt),
            },
        }

        # === SmolLM2 対比(SSM A 不在の直接確認)===
        smol = check_smollm_has_no_ssm()
        results["smollm_contrast"] = smol

        # === base-level 判別 verdict ===
        mamba_stable = results["mamba"]["summary"]["intrinsically_stable"]
        smol_has_ssm = smol["has_ssm_state_recurrence"]
        results["verdict"] = {
            "mamba_intrinsically_stable_trivial_pass": mamba_stable,
            "smollm_has_intrinsic_stability_certificate": smol_has_ssm,
            "base_level_discrimination": (
                "PASS" if (mamba_stable and not smol_has_ssm) else "REVIEW"),
            "statement": (
                "Mamba is stable-by-construction: every Mamba layer's SSM state-recurrence has a "
                "continuous diagonal A with strictly negative real part (A = -exp(A_log) < 0), so the "
                "discrete A-bar = exp(Delta*A) has |diagonal| < 1 and the max Lyapunov exponent "
                f"lambda_max = max(Delta*A) <= 0 for ALL Delta > 0 (global lambda_max_base = "
                f"{global_lam_max_base:.4g}, global max|A-bar| = {global_max_abs_abar_base:.6f} < 1). "
                "=> the contraction certificate is TRIVIALLY satisfied at base level, no adapter/gate needed. "
                "SmolLM2-135M (standard Llama Transformer) has NO SSM state-recurrence A matrix at all; the "
                "intrinsic-stability-certificate concept does not exist in its base, so stability must be "
                "imposed by the bolted-on verified adapter + cert gate. This is the framework's base-level "
                "discriminative power: it cleanly separates a stable-by-construction base (Mamba, trivial PASS) "
                "from a base that requires the gate to be made safe (SmolLM2, gate-mandatory)."
            ),
        }

        s = results["mamba"]["summary"]
        print(f"\n=== Mamba SSM 固有安定性 (全 {ext['n_layer']} 層) ===", flush=True)
        print(f"  全層 A 実部 < 0:           {s['all_layers_A_real_part_negative']}", flush=True)
        print(f"  全層 λ_max ≤ 0 (代表 Δ):   {s['all_layers_stable_at_base_delta']}", flush=True)
        print(f"  全層 λ_max ≤ 0 (Δ スイープ): {s['all_layers_stable_for_all_delta_sweep']}", flush=True)
        print(f"  global λ_max (代表 Δ):     {s['global_lambda_max_base']:.6g}  (≤0 = 固有安定)", flush=True)
        print(f"  global max|Ā 対角|:        {s['global_max_abs_abar_base']:.6f}  (<1 = 収縮)", flush=True)
        print(f"  A 実部 範囲:               [{s['global_A_min_real_part']:.4g}, {s['global_A_max_real_part']:.4g}]", flush=True)
        print(f"  → 固有安定 (自明 PASS):    {s['intrinsically_stable']}", flush=True)
        print(f"\n=== SmolLM2 対比 ===", flush=True)
        print(f"  model_type = {smol['model_type']}  SSM 再帰キー数 = {len(smol['ssm_recurrence_keys'])}", flush=True)
        print(f"  固有安定 certificate を base に持つ: {smol['has_ssm_state_recurrence']}", flush=True)
        print(f"\n=== base-level 判別 verdict = {results['verdict']['base_level_discrimination']} ===", flush=True)
        print("  Mamba: 固有安定 = 自明 PASS / SmolLM2: SSM A 不在 = gate 必須", flush=True)

        results["status"] = "ok"

    except Exception as e:
        results["status"] = "error"
        results["error"] = f"{type(e).__name__}: {e}"
        results["trace_tail"] = traceback.format_exc().splitlines()[-5:]
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        for line in results["trace_tail"]:
            print("  " + line, flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_mamba_lyapunov_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
