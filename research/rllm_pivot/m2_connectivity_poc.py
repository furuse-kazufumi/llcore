# SPDX-License-Identifier: Apache-2.0
"""M2.0 PoC: 会話連結性教師 (T1 ターン境界) × cert gate の最小実証 smoke。

設計 = docs/M2_CERT_CONNECTIVITY_DESIGN_2026_06_12.md §2 / 自走順 1。
phase2_capability_realce.py の実 LLM hidden 枠を踏襲し、地形 (Objective) のみ
「実会話の連結構造 (T1 ターン境界予測)」に差し替える。

M2.0 で答える smoke の問い (本測定 M2.1 の前提条件):
  1. **識別力**: T1 CE は floor (クラス事前の定数予測 CE) から動くか。
     floor に張り付くなら non-discriminating で M2.1 に進めない (realce の教訓)。
  2. **配線**: ConnectivityTerrain 上で MAP-Elites + gate (none / cert_sdp) が動くか。
  3. **guarantee 初見**: 無 gate の archive に真に発散する gene (empirical_rho >= 1)
     が混入するか / cert_sdp gate 下で 0 か。
     (注: empirical_rho は from-below オラクル — ρ>=1 の検出は確実、見逃しはありうる)

タスク T1 (ターン境界):
  会話 annotation 系列 x_1..x_T (SmolLM2-135M layer15 hidden の mean-pool を
  train-PCA top-M → per-seed ランダム射影 n=6) を adapter
  s' = decay ⊙ s + (1-decay) ⊙ tanh(W s + x_t) で流し、各位置 t で
  「annotation t+1 が新しい turn の先頭か (y_t=1) 継続か (0)」を予測する CE。

readout の変更経緯 (honest, smoke が配線調整の場であることの記録):
  v1 = X 空間クラス平均 centroid (gene 非依存) + 分離スケール β。smoke 1 回目
  (seed 0) で **train CE 0.9022 > floor 0.6269** = readout 族に定数予測 (クラス
  事前) すら含まれず non-discriminating と判明 (logits が s に強制依存するため)。
  v2 (現行) = **s 空間 LDA centroid + within-class β + log-prior bias**:
  c_k = train 範囲の状態 s_t のクラス別平均 (gene 依存・train のみで fit =
  リークなし)、β = 1/(2·within-class 平均二乗距離)、logits_k = -β‖s_t-c_k‖² +
  log prior_k。クラスが分離しない gene では c0≈c1 → logits ≈ log prior =
  **floor (定数予測) を族が包含**し、分離した分だけ floor を下回れる。
  判定基準 (CE < floor - 0.02) は v1/v2 共通で結果を見る前から不変。

honest 留保 (事前):
  - 系列は 1 本 (3 会話連結 ~数百 annotations)。train/held-out は時系列分割
    (前半/後半) — 会話の連続性を壊す文単位シャッフルはしない。per-seed 多様性は
    射影 P_s と分割境界の揺らぎ (±5%) で出す。
  - held-out 評価も系列の頭から状態を流す (入力の観測はリークでない — gold を
    fitness に使わないため)。
  - gold = 会話 JSON の turn 構造 (外部事実)。AnnotationStore の連結グラフ実装には
    依存しない (circularity 回避、設計書 §1)。
  - クラス不均衡 (境界は少数派) のため floor CE / 境界率を必ず開示する。

使い方::

    py -3.11 research/rllm_pivot/m2_connectivity_poc.py
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
from phase2_capability_realce import (  # noqa: E402
    LAYER,
    M_PCA,
    MODEL,
    N,
    _descriptor,
    _json_native,
    _rand_theta,
    _to_gene,
)

from llcore.clip.annotations import split_annotations  # noqa: E402

SOURCES = [
    os.path.join(_ROOT, "out", "chat_staged_smoke_results.json"),
    os.path.join(_ROOT, "out", "chat_endurance_results.json"),
    os.path.join(_HERE, "phase2_demo_verified_chat_results.json"),
]
BUDGET = 2000          # train-fitness 評価予算 (realce と同一)
N_SEEDS_SMOKE = 3      # smoke のみ。M2.1 本測定は >=15
RHO_SAMPLES = 2000     # empirical_rho の from-below サンプル数 (smoke 軽量化; 既定 8000)
SEED0 = 20260612


# --------------------------------------------------------------------------- #
# 会話 annotation 系列 + turn 境界 gold
# --------------------------------------------------------------------------- #
def load_conversation_sequence() -> tuple[list[str], np.ndarray, int]:
    """3 会話 JSON → (annotations 出現順, new_turn_flags, n_turns)。

    flags[i] = annotation i がその turn の先頭 annotation か。
    キー走査順は connectivity_bench.ingest と同一 (prompt→reply→user→assistant)。
    """
    anns: list[str] = []
    flags: list[bool] = []
    n_turns = 0
    for src in SOURCES:
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as f:
            d = json.load(f)
        turns = d.get("turns") or d.get("conversation") or []
        for turn in turns:
            first = True
            for key in ("prompt", "reply", "user", "assistant"):
                v = turn.get(key)
                if isinstance(v, str):
                    for a in split_annotations(v):
                        anns.append(a)
                        flags.append(first)
                        first = False
            if not first:  # この turn から annotation が 1 つ以上出た
                n_turns += 1
    return anns, np.array(flags, dtype=bool), n_turns


def get_annotation_hiddens(anns: list[str]) -> np.ndarray:
    """各 annotation を SmolLM2 frozen forward → layer LAYER hidden の mean-pool (T, 576)。

    内容ハッシュをキーに npz キャッシュ (モデル/レイヤ固定の決定論的出力のため安全)。
    """
    import hashlib

    key = hashlib.md5(("\x00".join(anns) + f"|{MODEL}|{LAYER}").encode()).hexdigest()
    cache = os.path.join(_ROOT, "out", f"m2_hidden_cache_{key[:12]}.npz")
    if os.path.exists(cache):
        H = np.load(cache)["H"]
        print(f"SmolLM2 hidden キャッシュ再利用 {cache} T={H.shape[0]}", flush=True)
        return H

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, output_hidden_states=True)
    model.eval()
    H = []
    for a in anns:
        ids = tok(a, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids)
        h = out.hidden_states[LAYER][0].numpy().astype(np.float64)  # (seq, 576)
        H.append(h.mean(0))
    print(f"SmolLM2 hidden 抽出 {time.time()-t0:.1f}s  T={len(H)} dim={H[0].shape[0]}",
          flush=True)
    out = np.asarray(H)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, H=out)
    return out


# --------------------------------------------------------------------------- #
# T1 地形 (per-seed view)
# --------------------------------------------------------------------------- #
class ConnectivityTerrain:
    """実会話 hidden 系列 + per-seed (射影 P_s, 時系列分割) = T1 ターン境界 CE 地形。"""

    def __init__(self, H: np.ndarray, flags: np.ndarray, rng: np.random.Generator,
                 train_frac: float = 0.6):
        T = H.shape[0]
        jitter = max(1, int(0.05 * T))
        self.split = int(train_frac * T) + int(rng.integers(-jitter, jitter + 1))
        train_h = H[: self.split]
        self.mu = train_h.mean(0)
        # train のみで PCA (リークなし) → per-seed ランダム回転 (realce と同じ構成)
        _, _, Vt = np.linalg.svd(train_h - self.mu, full_matrices=False)
        U = Vt[:M_PCA]
        Rp = rng.normal(size=(N, M_PCA)) / np.sqrt(M_PCA)
        self.P = Rp @ U                                   # (N, 576)
        X = (H - self.mu) @ self.P.T
        self.scale = float(np.std(X[: self.split]) + 1e-9)  # scale も train のみ
        self.X = X / self.scale * 0.5
        # y_t = flags[t+1] (有効 t = 0..T-2)
        self.y = flags[1:].astype(int)
        self.T = T
        tr_y = self.y[: self.split - 1]
        if tr_y.min() == tr_y.max():  # 片クラスしか無い分割は使えない (fail-loud)
            raise ValueError("train range has a single class; adjust train_frac")
        # floor (定数予測 = クラス事前) CE — train / held-out 各範囲で開示。
        # v2 readout: train 範囲のクラス事前を log-prior bias として持つ (floor 包含)。
        p1 = float(tr_y.mean())
        self.log_prior = np.log(np.array([1.0 - p1, p1]) + 1e-12)
        self.floor_train = self._entropy(tr_y)
        self.floor_heldout = self._ce_const_prior(self.y[self.split - 1:], p1)
        self.boundary_rate = float(self.y.mean())

    @staticmethod
    def _ce_const_prior(y: np.ndarray, p1: float) -> float:
        """train クラス事前 p1 の定数予測を範囲 y に適用した CE (held-out floor の正定義)。"""
        p1 = min(max(p1, 1e-12), 1 - 1e-12)
        return float(-(y * np.log(p1) + (1 - y) * np.log(1 - p1)).mean())

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        p1 = float(y.mean())
        p1 = min(max(p1, 1e-12), 1 - 1e-12)
        return float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))

    def _states(self, decay: np.ndarray, W: np.ndarray) -> np.ndarray:
        """系列を頭から流した状態列 S[t] = s_t (有効 t = 0..T-2、1 forward pass)。"""
        S = np.zeros((self.T - 1, N))
        s = np.zeros(N)
        for t in range(self.T - 1):
            s = decay * s + (1.0 - decay) * np.tanh(W @ s + self.X[t])
            S[t] = s
        return S

    def _fit(self, theta: np.ndarray, lo: int, hi: int) -> float:
        """t ∈ [lo, hi) の CE (fitness = -CE)。readout (v2) は **train 範囲のみ**で
        fit する gene 依存 LDA centroid + within-class β + log-prior bias。
        held-out 評価 (lo >= split-1) でも fit は train 範囲 = リークなし。"""
        decay = np.clip(theta[:N], 0.0, 1.0)
        W = np.clip(theta[N:].reshape(N, N), -2.0, 2.0)
        S = self._states(decay, W)
        s_tr = S[: self.split - 1]
        y_tr = self.y[: self.split - 1]
        c0 = s_tr[y_tr == 0].mean(0)
        c1 = s_tr[y_tr == 1].mean(0)
        centers = np.stack([c0, c1])                      # (2, N)
        within = float(((s_tr - centers[y_tr]) ** 2).sum(1).mean())
        beta = 1.0 / (2.0 * within + 1e-12)
        Se = S[lo:hi]
        ye = self.y[lo:hi]
        logits = -beta * ((Se[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        logits = logits + self.log_prior                   # floor 包含 (c0≈c1 → 事前予測)
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        ce = float(-np.log(P[np.arange(len(ye)), ye] + 1e-12).mean())
        return -ce

    def train(self, theta: np.ndarray) -> float:
        return self._fit(theta, 0, self.split - 1)

    def heldout(self, theta: np.ndarray) -> float:
        return self._fit(theta, self.split - 1, self.T - 1)


# --------------------------------------------------------------------------- #
# MAP-Elites (realce 同構造、gate を callable 化 + archive 全体を返す)
# --------------------------------------------------------------------------- #
def mapelites_archive(terrain, rng, B, admit, sigma=0.2, init=64, resample_cap=20):
    """realce.opt_mapelites と同一構造。差分 = (1) admit: callable (2) archive を返す。"""
    archive: dict = {}
    used = 0

    def _eval_and_place(th):
        nonlocal used
        f = terrain.train(th)
        used += 1
        cell = _descriptor(th)
        cur = archive.get(cell)
        if cur is None or f > cur[1]:
            archive[cell] = (th, f)

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
                # known-safe へ向かう反復縮小 fallback (evolve() の fallback 規律の
                # MAP-Elites 版)。W→0 で J = diag(decay) となり decay<1 なら
                # cert ladder は必ず admit する — random init が admit 領域に
                # ほぼ入らない gate (cert_inf 実測: init 64 全滅で空転) でも
                # archive を立ち上げられる。fail-closed は維持 (admit されない限り
                # 評価しない)。honest 留保: gate が探索を W 縮小方向へ誘導する
                # バイアスを持つ — gated 進化の挙動として明示的に受け入れる。
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
                # init と同じ反復縮小 fallback (空転防止 — admit 領域が狭い gate でも
                # used が単調に進むことを保証する)。
                for _ in range(40):
                    th[N:] *= 0.5
                    if admit(th):
                        admitted = True
                        break
            if not admitted:
                continue
        _eval_and_place(th)
    return archive


# smoke の sound gate は cert_inf (閉形式 O(n²)、µs 判定)。実測の経緯 (honest):
# 当初 cert_sdp を使ったが、random init の ~94% が発散側 (v2 seed 0 で rho>=1 が
# 61/65) のため gate がほぼ全 reject → resample で SDP solve (数百 ms) が数千回 =
# smoke が 1 時間級に停滞した。sound 性 (false-admit 0) は cert ladder 共通なので、
# 配線確認の smoke は inf で行い、cert_sdp は M2.1 本測定で inf-first cascade
# (inf admit → 即 admit、inf reject のみ sdp 再判定 — ladder の包含関係を利用) を
# 設計して使う。
GATES = {
    "none": lambda th: True,
    "cert_inf": lambda th: bool(C.cert_inf(_to_gene(th))),
}


# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    anns, flags, n_turns = load_conversation_sequence()
    print(f"会話系列: {len(anns)} annotations / {n_turns} turns "
          f"(境界率 {flags[1:].mean():.3f})", flush=True)
    H = get_annotation_hiddens(anns)

    results: dict = {"n_annotations": len(anns), "n_turns": n_turns,
                     "budget": BUDGET, "n_seeds": N_SEEDS_SMOKE,
                     "rho_samples": RHO_SAMPLES, "runs": []}
    for seed_i in range(N_SEEDS_SMOKE):
        rng_terrain = np.random.default_rng(SEED0 + seed_i)
        terrain = ConnectivityTerrain(H, flags, rng_terrain)
        info = {
            "seed": SEED0 + seed_i,
            "split": terrain.split,
            "boundary_rate": terrain.boundary_rate,
            "floor_train_ce": round(terrain.floor_train, 4),
            "floor_heldout_ce": round(terrain.floor_heldout, 4),
            "gates": {},
        }
        for gate_name, admit in GATES.items():
            rng = np.random.default_rng(SEED0 + 1000 + seed_i)  # gate 間で同一 RNG 系列
            t0 = time.time()
            archive = mapelites_archive(terrain, rng, BUDGET, admit)
            elapsed = time.time() - t0
            best_th, best_f = max(archive.values(), key=lambda v: v[1])
            held = terrain.heldout(best_th)
            # guarantee 監査: archive 全 gene の empirical_rho (from-below)
            rhos = [C.empirical_rho(_to_gene(th), n_samples=RHO_SAMPLES,
                                    seed=SEED0 + seed_i)
                    for th, _ in archive.values()]
            n_div = int(sum(r >= 1.0 for r in rhos))
            info["gates"][gate_name] = {
                "archive_size": len(archive),
                "best_train_ce": round(-best_f, 4),
                "heldout_ce_of_best": round(-held, 4),
                "rho_admitted_max": round(max(rhos), 4),
                "n_admitted_rho_ge_1": n_div,
                "opt_seconds": round(elapsed, 1),
            }
            print(f"[seed {seed_i} {gate_name:9s}] train CE {-best_f:.4f} "
                  f"(floor {terrain.floor_train:.4f}) | held-out {-held:.4f} "
                  f"(floor {terrain.floor_heldout:.4f}) | archive {len(archive)} "
                  f"| rho_max {max(rhos):.3f} | rho>=1: {n_div} | {elapsed:.0f}s",
                  flush=True)
        results["runs"].append(info)

    # smoke 判定 (設計書 §3 自走順 1 の合格条件)
    floors = [r["floor_train_ce"] for r in results["runs"]]
    bests = [r["gates"]["none"]["best_train_ce"] for r in results["runs"]]
    discriminating = all(b < f - 0.02 for b, f in zip(bests, floors))
    sound_gates = [g for g in GATES if g != "none"]
    sound_clean = all(r["gates"][g]["n_admitted_rho_ge_1"] == 0
                      for r in results["runs"] for g in sound_gates)
    results["smoke_discriminating"] = discriminating
    results["smoke_sound_gate_zero_false_admit"] = sound_clean
    results["total_seconds"] = round(time.time() - t_start, 1)

    out = os.path.join(_ROOT, "out", "m2_connectivity_poc.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_json_native)
    print(f"\n[smoke] 識別力 (train CE < floor-0.02 全 seed): {discriminating}", flush=True)
    print(f"[smoke] sound gate 採用 gene の rho>=1 ゼロ: {sound_clean}", flush=True)
    print(f"total {results['total_seconds']}s\nresults: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
