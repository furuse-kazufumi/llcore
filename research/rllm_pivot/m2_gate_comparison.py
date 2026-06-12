# SPDX-License-Identifier: Apache-2.0
"""M2.1 本測定: 会話連結性教師 (T1 ターン境界) 下の gate 4 種比較 (15 seed)。

設計 = docs/M2_CERT_CONNECTIVITY_DESIGN_2026_06_12.md §2.3-2.5。
smoke = m2_connectivity_poc.py (v4: 識別力 + cert_inf false-admit 0 を確認済み)。
地形・探索器・readout は smoke と同一 (import 再利用) — 変更は gate 軸の拡張のみ。

gate 4 種:
  - none        : 負対照 (無 gate)
  - stable_exp  : STABLE 風経験 gate (phase2_discriminative.perturbation_forgetting
                  < EPS_FORGET=1e-2, T=64, K_PROBE=8 — Phase 2 と同一設定)
  - cert_inf    : sound (閉形式, µs 判定) — sound gate の主代表
  - cert_sdp    : sound (最 navigable, SDP) — **seed 0-2 のみのサブ実験**。
                  1 判定 ~数百 ms × resample で full 15 seed は非現実的なため。
                  n=3 の小ささは結論の限定として honest に開示する。

事前登録判定 (本ファイルのコミットが登録。結果を見た後の変更禁止):
  G1 (guarantee 主役): cert_inf / cert_sdp の archive 全 gene + best で
      empirical_rho >= 1 がゼロ。stable_exp は >= 1 が 1 個以上 (sound と分離)。
      ※ Phase 2 の「84%」は設定依存値のため率の再現は主張しない。
  G2 (運用上の危険): none / stable_exp の **best gene** (fitness 頂点 = 採用対象)
      の rho >= 1 率を seed 横断で開示。
  C1 (non-degenerate): 各 gate で train CE < floor - 0.02 (会話教師が gate 下でも
      学習可能な信号であること)。
  C2 (capability tax): sound gate と none の held-out CE 差を開示 (優劣の主張なし
      — capability は Phase 2 で NEGATIVE 確定済み)。

honest 留保 (事前):
  - empirical_rho は from-below オラクル — 「>=1 検出」は確実、「<1」は近傍
    取りこぼしがありうる (soundness の絶対証明でない)。
  - stable_exp は確率的 gate (rng 依存) — seed ごとに gate rng を固定し再現可能に。
  - 系列 122 annotations は経験 gate のホライズン T=64 と同オーダー =
    経験 gate に有利寄りの設定 (設計書 §2.5、不利設定での負対照)。

使い方::

    py -3.11 research/rllm_pivot/m2_gate_comparison.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate")))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import coupled_nd as C  # noqa: E402
from m2_connectivity_poc import (  # noqa: E402
    BUDGET,
    RHO_SAMPLES,
    ConnectivityTerrain,
    get_annotation_hiddens,
    load_conversation_sequence,
    mapelites_archive,
)
from phase2_capability_realce import _json_native, _to_gene  # noqa: E402
from phase2_discriminative import EPS_FORGET, perturbation_forgetting  # noqa: E402

N_SEEDS = 15
N_SEEDS_SDP = 3            # cert_sdp サブ実験 (SDP コストのため。n=3 を開示)
SEED0 = 20260612


def make_admit(gate: str, gate_rng: np.random.Generator):
    """gate 名 → admit(theta) callable。stable_exp は gate_rng 依存 (確率的判定)。"""
    if gate == "none":
        return lambda th: True
    if gate == "stable_exp":
        return lambda th: bool(
            perturbation_forgetting(_to_gene(th), gate_rng) < EPS_FORGET)
    if gate == "cert_inf":
        return lambda th: bool(C.cert_inf(_to_gene(th)))
    if gate == "cert_sdp":
        return lambda th: bool(C.cert_sdp(_to_gene(th)))
    raise ValueError(gate)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="M2.1 gate 比較")
    parser.add_argument("--mode", choices=("main", "sdp"), default="main",
                        help="main = none/stable_exp/cert_inf 15 seed / "
                             "sdp = cert_sdp のみ seed 0-2 (分離サブ実験 — "
                             "SDP 判定が seed 0 で 6 分超過したため本判定と分離。"
                             "gate ごとに RNG 独立のため分離実行でも結果同一)")
    args = parser.parse_args()

    t_start = time.time()
    anns, flags, n_turns = load_conversation_sequence()
    H = get_annotation_hiddens(anns)
    print(f"M2.1 ({args.mode}): {len(anns)} annotations / {n_turns} turns / "
          f"budget {BUDGET} / {N_SEEDS} seeds (cert_sdp は {N_SEEDS_SDP})", flush=True)

    results: dict = {"n_annotations": len(anns), "n_turns": n_turns,
                     "budget": BUDGET, "n_seeds": N_SEEDS,
                     "n_seeds_sdp": N_SEEDS_SDP, "rho_samples": RHO_SAMPLES,
                     "eps_forget": EPS_FORGET, "runs": []}
    seed_range = range(N_SEEDS) if args.mode == "main" else range(N_SEEDS_SDP)
    for seed_i in seed_range:
        terrain = ConnectivityTerrain(H, flags, np.random.default_rng(SEED0 + seed_i))
        info: dict = {"seed": SEED0 + seed_i, "split": terrain.split,
                      "floor_train_ce": round(terrain.floor_train, 4),
                      "floor_heldout_ce": round(terrain.floor_heldout, 4),
                      "gates": {}}
        gates = (["none", "stable_exp", "cert_inf"] if args.mode == "main"
                 else ["cert_sdp"])
        for gate in gates:
            rng = np.random.default_rng(SEED0 + 1000 + seed_i)   # gate 間で同一系列
            gate_rng = np.random.default_rng(SEED0 + 2000 + seed_i)
            t0 = time.time()
            archive = mapelites_archive(terrain, rng, BUDGET, make_admit(gate, gate_rng))
            elapsed = time.time() - t0
            best_th, best_f = max(archive.values(), key=lambda v: v[1])
            held = terrain.heldout(best_th)
            rhos = [C.empirical_rho(_to_gene(th), n_samples=RHO_SAMPLES,
                                    seed=SEED0 + seed_i)
                    for th, _ in archive.values()]
            best_rho = C.empirical_rho(_to_gene(best_th), n_samples=RHO_SAMPLES,
                                       seed=SEED0 + seed_i)
            info["gates"][gate] = {
                "archive_size": len(archive),
                "best_train_ce": round(-best_f, 4),
                "heldout_ce_of_best": round(-held, 4),
                "best_rho": round(best_rho, 4),
                "rho_admitted_max": round(max(rhos), 4),
                "n_admitted_rho_ge_1": int(sum(r >= 1.0 for r in rhos)),
                "opt_seconds": round(elapsed, 1),
            }
            v = info["gates"][gate]
            print(f"[seed {seed_i:2d} {gate:10s}] trainCE {v['best_train_ce']:.4f} "
                  f"(floor {terrain.floor_train:.4f}) heldCE "
                  f"{v['heldout_ce_of_best']:.4f} | arch {v['archive_size']:3d} | "
                  f"best_rho {v['best_rho']:.3f} | rho>=1 "
                  f"{v['n_admitted_rho_ge_1']:3d} | {elapsed:.0f}s", flush=True)
        results["runs"].append(info)

    # --- 事前登録判定の集計 ---------------------------------------------------
    def _all_gate(gate: str, key: str) -> list:
        return [r["gates"][gate][key] for r in results["runs"] if gate in r["gates"]]

    if args.mode == "main":
        g1_inf = all(v == 0 for v in _all_gate("cert_inf", "n_admitted_rho_ge_1")) \
            and all(v < 1.0 for v in _all_gate("cert_inf", "best_rho"))
        g1_stable_unsafe = sum(_all_gate("stable_exp", "n_admitted_rho_ge_1")) > 0
        g2_none_best_div = sum(v >= 1.0 for v in _all_gate("none", "best_rho"))
        g2_stable_best_div = sum(v >= 1.0 for v in _all_gate("stable_exp", "best_rho"))
        c1 = {g: all(ce < r["floor_train_ce"] - 0.02
                     for r, ce in zip(results["runs"], _all_gate(g, "best_train_ce")))
              for g in ("none", "stable_exp", "cert_inf")}
        tax = [a - b for a, b in zip(_all_gate("cert_inf", "heldout_ce_of_best"),
                                     _all_gate("none", "heldout_ce_of_best"))]
        results["verdict"] = {
            "G1_cert_inf_sound": g1_inf,
            "G1_stable_exp_admits_divergent": g1_stable_unsafe,
            "G2_none_best_divergent_seeds": f"{g2_none_best_div}/{N_SEEDS}",
            "G2_stable_best_divergent_seeds": f"{g2_stable_best_div}/{N_SEEDS}",
            "C1_learnable_under_gate": c1,
            "C2_cert_inf_heldout_tax_mean": round(float(np.mean(tax)), 4),
        }
    else:
        g1_sdp = all(v == 0 for v in _all_gate("cert_sdp", "n_admitted_rho_ge_1")) \
            and all(v < 1.0 for v in _all_gate("cert_sdp", "best_rho"))
        results["verdict"] = {"G1_cert_sdp_sound_n3": g1_sdp}
    results["total_seconds"] = round(time.time() - t_start, 1)

    out = os.path.join(
        _ROOT, "out",
        "m2_gate_comparison.json" if args.mode == "main"
        else "m2_gate_comparison_sdp.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_json_native)
    print("\n=== 事前登録判定 ===", flush=True)
    for k, v in results["verdict"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"total {results['total_seconds']}s\nresults: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
