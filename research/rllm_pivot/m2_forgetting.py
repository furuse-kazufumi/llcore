# SPDX-License-Identifier: Apache-2.0
"""M2.2: 会話連結性の保持 (忘却) 測定 — gate は忘却を悪化させないか。

設計 = docs/M2_CERT_CONNECTIVITY_DESIGN_2026_06_12.md §2.4 の 3。
M2.0/M2.1 と同一の地形・探索器を再利用し、2 段階適応を加える:

  Phase A: 前半窓 [0, 50%] を train として MAP-Elites → archive_A, best_A
  Phase B: 後半窓 [50%, 100%] を train として、archive_A を初期集団に
           継続進化 (budget は A と同じ) → best_B
  忘却量 = CE_A(best_B) − CE_A(best_A)
           (CE_A = 前半窓の CE。readout はどちらも前半窓で fit — 同一の物差し)

事前登録判定 (本ファイルのコミットが登録):
  F1: 各 gate (none / stable_exp / cert_inf) の忘却量 mean ± per-seed を開示。
  F2: cert_inf の忘却量が none より系統的に悪化しない (paired 差の符号と分布を
      開示)。**cert が忘却を「防ぐ」とは主張しない** — cert は発散 (ρ≥1) を防ぐ
      のであって忘却を防ぐ保証ではない (over-claim 禁止)。確認するのは中立性のみ。
  F3 (副次): Phase B 後の best_B の empirical_rho — 継続適応後も sound gate の
      false-admit 0 が保たれるか (G1 の継続版)。

honest 留保 (事前):
  - 前半/後半は同一会話ストリームの時間分割 — 「タスク A → タスク B」の古典的
    CL 設定より相関が強く、忘却が出にくい設定の可能性 (出なければ「この設定では
    観測されず」と報告する。「忘却は起きない」への一般化はしない)。
  - readout は gene 依存 fit (v2 設計) のため、忘却は「表現の劣化」でなく
    「状態ダイナミクスの劣化」を測る (readout は測定ごとに最適 fit し直される)。
  - Phase B の探索は archive_A から始まる = warm start。cold start との比較は
    本測定の範囲外。

使い方::

    py -3.11 research/rllm_pivot/m2_forgetting.py
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
)
from m2_gate_comparison import make_admit  # noqa: E402
from phase2_capability_realce import (  # noqa: E402
    N,
    _descriptor,
    _json_native,
    _rand_theta,
    _to_gene,
)

N_SEEDS = 10
SEED0 = 20260612
GATES_F = ("none", "stable_exp", "cert_inf")


class WindowedTerrain(ConnectivityTerrain):
    """ConnectivityTerrain の additive 拡張 — readout fit 窓を指定できる変種。

    親クラスは「fit = [0, split-1) 固定」。本クラスは fit/eval 窓を独立指定する
    `fit_eval(theta, fit_lo, fit_hi, lo, hi)` を足す (親メソッドは無変更 =
    M2.0/M2.1 の数値との互換を保証)。
    """

    def fit_eval(self, theta: np.ndarray, fit_lo: int, fit_hi: int,
                 lo: int, hi: int) -> float:
        decay = np.clip(theta[:N], 0.0, 1.0)
        W = np.clip(theta[N:].reshape(N, N), -2.0, 2.0)
        S = self._states(decay, W)
        s_tr = S[fit_lo:fit_hi]
        y_tr = self.y[fit_lo:fit_hi]
        if y_tr.min() == y_tr.max():
            raise ValueError("fit window has a single class")
        c0 = s_tr[y_tr == 0].mean(0)
        c1 = s_tr[y_tr == 1].mean(0)
        centers = np.stack([c0, c1])
        within = float(((s_tr - centers[y_tr]) ** 2).sum(1).mean())
        beta = 1.0 / (2.0 * within + 1e-12)
        p1 = float(y_tr.mean())
        log_prior = np.log(np.array([1.0 - p1, p1]) + 1e-12)
        Se = S[lo:hi]
        ye = self.y[lo:hi]
        logits = -beta * ((Se[:, None, :] - centers[None, :, :]) ** 2).sum(2) + log_prior
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        return -float(-np.log(P[np.arange(len(ye)), ye] + 1e-12).mean())


def mapelites_seeded(fitness, rng, B, admit, init_thetas=None,
                     sigma=0.2, init=64, resample_cap=20):
    """mapelites_archive の変種 — fitness callable + 初期集団 (warm start) 対応。

    m2_connectivity_poc.mapelites_archive と同一構造 (縮小 fallback 含む)。
    差分 = (1) terrain.train でなく fitness(theta) を直接受ける
          (2) init_thetas を最初に評価して archive へ (admit を通った個体のみ —
          fail-closed。warm start 個体も gate の検査対象であることが M2.2 の要点:
          Phase A の発散 gene を Phase B の sound gate は引き継がない)。
    """
    archive: dict = {}
    used = 0

    def _eval_and_place(th):
        nonlocal used
        f = fitness(th)
        used += 1
        cell = _descriptor(th)
        cur = archive.get(cell)
        if cur is None or f > cur[1]:
            archive[cell] = (th, f)

    for th0 in (init_thetas or []):
        if used >= B:
            break
        if admit(th0):
            _eval_and_place(np.array(th0, dtype=float))

    for _ in range(init):
        if used >= B:
            break
        th = _rand_theta(rng)
        if not admit(th):
            ok = False
            for _ in range(resample_cap):
                th = _rand_theta(rng)
                if admit(th):
                    ok = True
                    break
            if not ok:
                for _ in range(40):
                    th[N:] *= 0.5
                    if admit(th):
                        ok = True
                        break
            if not ok:
                continue
        _eval_and_place(th)

    while used < B:
        keys = list(archive.keys())
        if not keys:
            th = _rand_theta(rng)
        else:
            parent = archive[keys[rng.integers(len(keys))]][0]
            th = parent + sigma * rng.standard_normal(parent.shape)
            th[:N] = np.clip(th[:N], 0, 1)
            th[N:] = np.clip(th[N:], -2, 2)
        if not admit(th):
            admitted = False
            for _ in range(resample_cap):
                if used >= B:
                    break
                parent = (archive[keys[rng.integers(len(keys))]][0]
                          if keys else _rand_theta(rng))
                th = parent + sigma * rng.standard_normal(parent.shape)
                th[:N] = np.clip(th[:N], 0, 1)
                th[N:] = np.clip(th[N:], -2, 2)
                if admit(th):
                    admitted = True
                    break
            if not admitted:
                for _ in range(40):
                    th[N:] *= 0.5
                    if admit(th):
                        admitted = True
                        break
            if not admitted:
                continue
        _eval_and_place(th)
    return archive


def main() -> int:
    t_start = time.time()
    anns, flags, n_turns = load_conversation_sequence()
    H = get_annotation_hiddens(anns)
    print(f"M2.2: {len(anns)} annotations / {n_turns} turns / budget {BUDGET}×2 / "
          f"{N_SEEDS} seeds", flush=True)

    results: dict = {"n_annotations": len(anns), "budget_per_phase": BUDGET,
                     "n_seeds": N_SEEDS, "runs": []}
    for seed_i in range(N_SEEDS):
        # train_frac=0.5: Phase A 窓 = [0, split)、Phase B 窓 = [split, T-1)
        terrain = WindowedTerrain(H, flags, np.random.default_rng(SEED0 + seed_i),
                                  train_frac=0.5)
        sp = terrain.split
        T1 = terrain.T - 1
        ce_a = lambda th: terrain.fit_eval(th, 0, sp - 1, 0, sp - 1)        # noqa: E731
        ce_b = lambda th: terrain.fit_eval(th, sp - 1, T1, sp - 1, T1)      # noqa: E731
        info: dict = {"seed": SEED0 + seed_i, "split": sp, "gates": {}}
        for gate in GATES_F:
            rng = np.random.default_rng(SEED0 + 1000 + seed_i)
            gate_rng = np.random.default_rng(SEED0 + 2000 + seed_i)
            admit = make_admit(gate, gate_rng)
            t0 = time.time()
            arch_a = mapelites_seeded(ce_a, rng, BUDGET, admit)
            best_a = max(arch_a.values(), key=lambda v: v[1])[0]
            # Phase B: archive_A 全個体を warm start に (admit 再検査付き)
            init_b = [th for th, _ in arch_a.values()]
            arch_b = mapelites_seeded(ce_b, rng, BUDGET, admit, init_thetas=init_b)
            best_b = max(arch_b.values(), key=lambda v: v[1])[0]
            cea_a = -ce_a(best_a)
            cea_b = -ce_a(best_b)
            forget = round(cea_b - cea_a, 4)
            rho_b = C.empirical_rho(_to_gene(best_b), n_samples=RHO_SAMPLES,
                                    seed=SEED0 + seed_i)
            info["gates"][gate] = {
                "ce_a_of_best_a": round(cea_a, 4),
                "ce_a_of_best_b": round(cea_b, 4),
                "forgetting": forget,
                "ce_b_of_best_b": round(-ce_b(best_b), 4),
                "best_b_rho": round(rho_b, 4),
                "seconds": round(time.time() - t0, 1),
            }
            v = info["gates"][gate]
            print(f"[seed {seed_i:2d} {gate:10s}] CE_A(A) {v['ce_a_of_best_a']:.4f} "
                  f"→ CE_A(B) {v['ce_a_of_best_b']:.4f} | forget {forget:+.4f} | "
                  f"CE_B(B) {v['ce_b_of_best_b']:.4f} | rho_B {v['best_b_rho']:.3f} "
                  f"| {v['seconds']:.0f}s", flush=True)
        results["runs"].append(info)

    # F1/F2 集計
    def _fg(gate):
        return [r["gates"][gate]["forgetting"] for r in results["runs"]]

    f1 = {g: {"mean": round(float(np.mean(_fg(g))), 4),
              "values": _fg(g)} for g in GATES_F}
    diff = [a - b for a, b in zip(_fg("cert_inf"), _fg("none"))]
    f3 = sum(r["gates"]["cert_inf"]["best_b_rho"] >= 1.0 for r in results["runs"])
    results["verdict"] = {
        "F1_forgetting_by_gate": f1,
        "F2_cert_minus_none_diffs": [round(d, 4) for d in diff],
        "F2_cert_not_worse_mean": float(np.mean(diff)) <= 0.0,
        "F3_cert_best_b_rho_ge_1_seeds": f"{f3}/{N_SEEDS}",
    }
    results["total_seconds"] = round(time.time() - t_start, 1)
    out = os.path.join(_ROOT, "out", "m2_forgetting.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_json_native)
    print("\n=== 事前登録判定 ===", flush=True)
    for k, v in results["verdict"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"total {results['total_seconds']}s\nresults: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
