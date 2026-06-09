# SPDX-License-Identifier: Apache-2.0
"""Phase 1.5: feasibility 実測 — 変異/cert/fitness の wall-time + cert コストの n スケーリング。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) Phase 1 step 5 / §⑨:
  「変異1回+cert1回+CE1回の wall-time/MB → 30h 外挿、成長 n で再計測」。
  最大の計算リスク = 構造変更 1 回ごとの cert × N 世代 × M block。width_grow が n を成長させると
  cert コストが増大 (cert_inf は O(n²)、cert_two/sdp は **2^n 頂点**で指数)。

測定:
  - per-op wall-time: mutate / cert_inf / cert_two / cert_sdp / fitness(adapter forward) / width_grow。
  - cert コストの n スケーリング (n=4..14): cert_inf(多項式) vs cert_two/sdp(2^n) の壁を実測。
  - 代表 budget 外挿: pop × gens × blocks × (mutate + gate + fitness) を 30h と比較。
  - メモリ: MAP-Elites archive (G genes × (n+n²) float64) の理論サイズ + (psutil 在れば) RSS。

honest:
  - 本機は **CPU** (torch+cpu)。実 GPU 訓練 (Kaggle T4) では base forward (CE) が dominant で、
    cert/mutate 数値演算は GPU でなく CPU で回る (numpy)。∴ 本測定の cert/mutate 時間は GPU 環境でも
    概ね同等 (CPU-bound) で外挿に使える。CE は別途 base forward コスト (本 script は adapter proxy)。
  - cert_sdp は cert_two fast-path で大半が解決 (genuine SDP solve は cert_two FAIL 時のみ)。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
import phase1_structural_surgery as S  # noqa: E402
import coupled_nd as C  # noqa: E402


def timeit(fn, reps):
    # warmup
    for _ in range(min(3, reps)):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


TWO_MAX_N = 12   # cert_two は 2^n 頂点 SVD; n>12 は測定せず外挿 (ハング回避)
SDP_MAX_N = 8    # cert_sdp は 2^n 頂点 LMI (cvxpy); n>8 は実用外 (genuine solve がハング)


def measure_n(rng, n, reps_cheap=2000, reps_cert=200, L=48):
    codec = C.CoupledNDGeneCodec(n)
    g = S.sample_admitted_base(rng, n)
    while g is None:
        g = S.sample_admitted_base(rng, n)
    gtype = np.concatenate([g.decay, g.W.reshape(-1)])
    obj = C.RotationNDObjective(n)
    X = rng.normal(size=(L, n))
    d = S.random_directions(rng, g)

    t_mut = timeit(lambda: codec.mutate(gtype, 0.15, rng), reps_cheap)
    t_inf = timeit(lambda: C.cert_inf(g), reps_cert)
    # cert_two/sdp は 2^n; 上限超は None (外挿) でハング回避
    rc = max(15, reps_cert // (2 ** max(0, n - 6)))
    t_two = timeit(lambda: C.cert_two(g), rc) * 1e6 if n <= TWO_MAX_N else None
    t_sdp = timeit(lambda: C.cert_sdp(g), max(5, rc // 4)) * 1e6 if n <= SDP_MAX_N else None
    t_fit = timeit(lambda: obj.fitness(g), reps_cheap)
    t_grow = timeit(lambda: S.width_grow(g, eps=0.5, in_dir=d.in_dir, out_dir=d.out_dir,
                                         self_w=d.self_w, new_decay=d.new_decay, mode="net2net", k=d.k),
                    reps_cheap)
    return {"n": n, "dim": codec.dim,
            "t_mutate_us": t_mut * 1e6, "t_cert_inf_us": t_inf * 1e6,
            "t_cert_two_us": t_two, "t_cert_sdp_us": t_sdp,
            "t_fitness_us": t_fit * 1e6, "t_width_grow_us": t_grow * 1e6}


def extrapolate(per_n, pop, gens, blocks, gate, budget_h=30.0):
    """代表 budget 外挿: gens × pop × blocks × (mutate + gate + fitness)。"""
    out = {}
    for rec in per_n:
        n = rec["n"]
        gate_us = rec[f"t_cert_{gate}_us"]
        if gate_us is None:
            out[str(n)] = {"per_eval_us": None, "gate_us": None, "total_hours": None,
                           "fits_budget": None, "note": f"cert_{gate} not measured (n>cap, 2^n 外挿)"}
            continue
        per_eval = rec["t_mutate_us"] + gate_us + rec["t_fitness_us"]
        total_s = gens * pop * blocks * per_eval / 1e6
        out[str(n)] = {
            "per_eval_us": per_eval, "gate_us": gate_us,
            "total_seconds": total_s, "total_hours": total_s / 3600.0,
            "fits_budget": total_s / 3600.0 <= budget_h,
        }
    return out


def archive_mem_mb(n, n_cells):
    floats = n_cells * (n + n * n)
    return floats * 8 / (1024 ** 2)


def main():
    rng = np.random.default_rng(20260609)
    ns = (4, 6, 8, 10, 12, 14)
    per_n = []
    print("per-op wall-time (μs):", flush=True)
    for n in ns:
        rec = measure_n(rng, n)
        per_n.append(rec)
        ct = f"{rec['t_cert_two_us']:.1f}" if rec['t_cert_two_us'] is not None else "—(2^n外挿)"
        cs = f"{rec['t_cert_sdp_us']:.1f}" if rec['t_cert_sdp_us'] is not None else "—(2^n外挿)"
        print(f"  n={n:2d} dim={rec['dim']:3d}  mutate={rec['t_mutate_us']:.2f}  "
              f"cert_inf={rec['t_cert_inf_us']:.2f}  cert_two={ct}  "
              f"cert_sdp={cs}  fitness={rec['t_fitness_us']:.2f}  "
              f"width_grow={rec['t_width_grow_us']:.2f}", flush=True)

    # cert スケーリング比 (n に対する cert_two 増大 = 2^n 壁)
    two = {r["n"]: r["t_cert_two_us"] for r in per_n}
    print("\ncert_two コスト n スケーリング:", flush=True)
    prev = None
    for n in ns:
        if two[n] is None:
            print(f"  n={n:2d} cert_two=測定上限超 (2^{n} 頂点 = 外挿域)", flush=True)
            continue
        ratio = (two[n] / prev) if prev else float("nan")
        print(f"  n={n:2d} cert_two={two[n]:.1f}μs  ×前測定={ratio:.2f}", flush=True)
        prev = two[n]

    # 代表 budget 外挿 (small-n per-component: n=6, cert_two gate)
    cfg = {"pop": 64, "gens": 200, "blocks": 4, "gate": "two", "budget_h": 30.0}
    extr = extrapolate(per_n, cfg["pop"], cfg["gens"], cfg["blocks"], cfg["gate"], cfg["budget_h"])
    print(f"\n外挿 (pop={cfg['pop']} gens={cfg['gens']} blocks={cfg['blocks']} gate=cert_two, budget={cfg['budget_h']}h):", flush=True)
    for n in ns:
        e = extr[str(n)]
        if e["per_eval_us"] is None:
            print(f"  n={n:2d}  cert_two 未測定 (n>{TWO_MAX_N}=2^n 外挿域)", flush=True)
            continue
        print(f"  n={n:2d}  per-eval={e['per_eval_us']:.2f}μs  総時間={e['total_hours']:.4f}h  "
              f"30h 収まる={e['fits_budget']}", flush=True)

    mem = {str(n): archive_mem_mb(n, 4096) for n in ns}
    try:
        import psutil  # type: ignore
        rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        rss_mb = None

    results = {"meta": {"seed": 20260609, "host": "CPU (torch+cpu)", "budget_h": cfg["budget_h"],
                        "extrapolation_config": cfg},
               "per_op_us": per_n,
               "budget_extrapolation_cert_two": extr,
               "archive_mem_mb_4096cells": mem,
               "process_rss_mb": rss_mb}
    out = os.path.join(os.path.dirname(__file__), "phase1_feasibility_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\narchive(4096 cells) MB: " + " ".join(f"n{n}={mem[str(n)]:.2f}" for n in ns), flush=True)
    if rss_mb:
        print(f"process RSS={rss_mb:.0f}MB", flush=True)
    print(f"結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
