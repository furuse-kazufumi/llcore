# SPDX-License-Identifier: Apache-2.0
"""Phase 2 残り (F12 本番): 実 SmolLM2-CE 地形での capability 再測定。

PHASE_2_VERDICT.md §4 honest 留保 #1 / §5 次セッション候補:
  phase2_capability_terrain.py の capability NULL_TIE は **synthetic K-basin Gaussian 地形**上の結果。
  「synthetic 地形であり実 SmolLM2-CE 損失地形ではない」「実地形は単峰の可能性すらあり、その場合
   capability はさらに立たない」と明記された。**本実験はその heavier follow-up**: synthetic 地形を
  **実 SmolLM2-135M の hidden state 由来の次トークン(=次 hidden クラスタ)予測 CE 地形** に置換し、
  capability NULL_TIE を実地形で確証 or 反証する。

honest 設計(phase2_capability_terrain.py の枠を踏襲、地形のみ実 LLM 由来へ):
  - 実 SmolLM2-135M (Apache-2.0) を CPU frozen load → corpus 各文の layer-L hidden 列 H_i (seq_i, 576)。
  - **地形の族(per-seed)**: 実 hidden は固定だが、各 seed で (射影 P_s: 576→n, readout R_s: n→K,
    train/held-out 文分割) を変えて **実 LLM 由来 CE 地形のインスタンス**を作る(phase2 が per-seed で
    basin 中心を振ったのと同じ役割。地形は real-LLM-derived だが view が seed 依存=地形族)。
  - **gene = small-n verified recurrent adapter** (CoupledNDGene, n≤6, Phase −1 確定の small-n per-component)。
    s_t = decay⊙s + (1-decay)⊙tanh(W s + x_t), x_t = X[t] (=H[t]@P^T 正規化)。
  - **CE 地形(実 LLM 由来)**: 目的 = 各位置 t で「次 hidden の cluster y_t = cluster(X[t+1])」を予測。
    readout = **centroid 最近傍分類器**(GMM 事後確率の原理的選択): logits_t[k] = -β‖s_t - center_k‖²
    (β=1/(2σ²), σ²=train 内クラスタ分散)。CE = -log softmax(logits_t)[y_t]。fitness = -mean CE。
    = 「adapter が次 hidden のクラスタ重心へ状態を寄せる」実 forecasting タスク。固定ランダム readout は
    n=4 状態から 8 クラス予測が floor に張り付き non-discriminating だったため、原理的な centroid readout へ
    (smoke で確認・修正、feedback_scenario_iterative)。cluster は **train 文の射影 hidden のみ**で
    KMeans fit(held-out へのリークなし)。adapter (decay,W) のみ可変=verified adapter 思想を維持。
  - **train/held-out 分離**: train 文で fitness 最適化、held-out **文**(最適化中 未観測)で汎化 CE を評価。
  - 4+1 optimizer を **同予算 B (train-fitness 評価回数)** で: random / gradient(有限差分+restart) /
    gradient_strong(restart64=meta-gate) / MAP-Elites / MAP-Elites+gate(ρ<1 cert_inf gate)。
  - **honest_eval 4 条件 AND** (phase2 と同一): diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15 ∧
    |paired_sign_delta|≥0.147。meta-gate(BG10) で EXISTS/NULL/ARTIFACT を確定。
  - **多峰性(F9)**: 実 CE 地形が多峰か単峰かを F9 find_basins で測る(「実地形は単峰の可能性」の検証)。

honest 留保(本実験):
  - readout R は固定ランダム(base 凍結 + adapter のみ可変の思想)。R が表現を絞るため CE 地形が
    floor に張り付けば non-discriminating(verdict の証拠力低下)→ 識別力を必ず実測し明示。
  - cluster target は射影 n 空間の KMeans(実 vocab の next-token softmax CE そのものではない=
    「実 hidden 由来の次内部表現クラス CE」)。実 full-vocab CE は n≤6 readout では degenerate ゆえ採らない。
  - 予算 = train-fitness 評価回数で厳密に揃える。gated ME の cert 受理 resample は budget 非計上
    (phase2 と同じ「予算=fitness 評価回数」定義、honest 留保継承)。
  - paired_sign_delta = net-win-fraction (教科書 Cliff's delta でない、計画 §⑬整合)。
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verified_evolution_sdp_gate")))
import coupled_nd as C  # noqa: E402

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

MODEL = "HuggingFaceTB/SmolLM2-135M"
LAYER = 15
N = 4                  # small-n per-component (Phase −1 確定, phase2 と同一)
K_CLUSTERS = 8         # 次 hidden クラスタ数 (CE クラス数)
T_MAX = None           # 文ごとの最大トークン(None=全部)
BUDGET = 2000          # 1 optimizer/seed あたりの train-fitness 評価予算 (phase2 と同一)
N_SEEDS = 20           # ≥15 (phase2 と同一)
GRID = 12              # MAP-Elites archive grid
SEED0 = 20260609
N_TRAIN_SENT = 12      # train 文数 / 残りが held-out
READOUT_SCALE = 1.5    # 固定 readout のスケール(識別力確保。結果に合わせ調整=p-hack ゆえ固定)

CORPUS = [
    "The infinite monkey theorem is a probability statement about random typing.",
    "A verified evolution framework gates structural change with a soundness certificate.",
    "Small language models can run on a CPU and still produce coherent text.",
    "Contraction means the system forgets its initial condition over time.",
    "Gradient descent follows the steepest direction of a differentiable loss surface.",
    "Evolutionary search maintains a population and recombines promising candidates.",
    "A multimodal landscape has several local optima separated by valleys.",
    "Cross entropy measures the distance between a prediction and a target distribution.",
    "The spectral radius of a Jacobian bounds the local growth of perturbations.",
    "State space models discretize a continuous linear recurrence with a time step.",
    "Honest disclosure requires reporting null results as first class findings.",
    "Overfitting happens when a model memorizes training inputs instead of generalizing.",
    "A quality diversity archive stores the best solution found in each behavior cell.",
    "Lyapunov exponents quantify how nearby trajectories diverge or converge.",
    "The adapter projects a high dimensional hidden state into a small summary.",
    "Reproducibility means another researcher can rerun the experiment and agree.",
    "Selective recurrence lets the model decide how much past context to retain.",
    "A frozen backbone keeps the base model fixed while a small head adapts.",
    "Power analysis tells whether a study could detect an effect of a given size.",
    "The certificate is sound when it never admits a divergent configuration.",
]


# --------------------------------------------------------------------------- #
# 実 SmolLM2 hidden 列(文ごと)
# --------------------------------------------------------------------------- #
def get_real_hidden_sentences():
    """SmolLM2-135M frozen load → corpus 各文の LAYER hidden 列 list[(seq_i, 576)] を返す。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, output_hidden_states=True)
    model.eval()
    sents = []
    for text in CORPUS:
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids)
        h = out.hidden_states[LAYER][0].numpy().astype(np.float64)  # (seq, 576)
        if T_MAX is not None:
            h = h[:T_MAX]
        if h.shape[0] >= 4:  # 予測に最低限の長さ
            sents.append(h)
    print(f"SmolLM2-135M loaded + hidden 抽出 {time.time()-t0:.1f}s  "
          f"文数={len(sents)} 平均seq={np.mean([s.shape[0] for s in sents]):.1f} dim={sents[0].shape[1]}",
          flush=True)
    return sents


# --------------------------------------------------------------------------- #
# tiny numpy KMeans (依存追加を避ける自己完結実装)
# --------------------------------------------------------------------------- #
def tiny_kmeans(X, k, rng, iters=50):
    """X:(M,d) → (centers:(k,d), labels:(M,)). k-means++ 風 init + Lloyd。決定論的(rng 固定)。"""
    M = X.shape[0]
    # k-means++ init
    idx = [int(rng.integers(M))]
    d2 = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / (d2.sum() + 1e-12)
        idx.append(int(rng.choice(M, p=probs)))
        d2 = np.minimum(d2, ((X - X[idx[-1]]) ** 2).sum(1))
    centers = X[idx].copy()
    labels = np.zeros(M, dtype=int)
    for _ in range(iters):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)  # (M,k)
        new = dists.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(0)
    return centers, labels


def _assign(X, centers):
    return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2).argmin(1)


# --------------------------------------------------------------------------- #
# 実 LLM CE 地形(per-seed の view)
# --------------------------------------------------------------------------- #
class RealCETerrain:
    """実 SmolLM2 hidden 列 + per-seed (射影 P, readout R, train/held-out 分割, cluster) = 実 CE 地形。"""

    def __init__(self, sents, rng):
        d = sents[0].shape[1]
        # per-seed 射影 576->N(地形の view を seed で振る)
        self.P = rng.normal(size=(N, d)) / np.sqrt(d)
        # 全文を射影 → スケール正規化(phase0_framework と整合: 入力 |x| を程よく)
        Xs = [h @ self.P.T for h in sents]
        allX = np.concatenate(Xs, 0)
        self.scale = float(np.std(allX) + 1e-9)
        self.Xs = [x / self.scale * 0.5 for x in Xs]
        # train/held-out 文分割(per-seed)
        order = rng.permutation(len(sents))
        self.train_idx = sorted(order[:N_TRAIN_SENT].tolist())
        self.test_idx = sorted(order[N_TRAIN_SENT:].tolist())
        # cluster は **train 文の射影 hidden のみ** で fit(held-out リークなし)
        train_concat = np.concatenate([self.Xs[i] for i in self.train_idx], 0)
        self.centers, train_labels = tiny_kmeans(train_concat, K_CLUSTERS, rng)
        # 各文の target = 次位置の cluster(y_t = cluster(X[t+1]))。位置 0..seq-2 が有効。
        self.Y = [_assign(x, self.centers) for x in self.Xs]
        # readout = centroid 最近傍分類器(GMM 事後)。β=1/(2σ²), σ²=train 内クラスタ平均分散。
        resid2 = ((train_concat - self.centers[train_labels]) ** 2).sum(1)  # 各点の重心二乗距離
        sigma2 = float(resid2.mean()) + 1e-9
        self.beta = 1.0 / (2.0 * sigma2)

    def _ce_sentence(self, decay, W, i):
        """文 i の mean CE(位置 t の状態 s_t で次 cluster y_t を centroid 最近傍で予測)。"""
        x = self.Xs[i]
        y = self.Y[i]
        seq = x.shape[0]
        s = np.zeros(N)
        tot = 0.0
        cnt = 0
        for t in range(seq - 1):  # 最終位置は次が無い
            s = decay * s + (1.0 - decay) * np.tanh(W @ s + x[t])
            logits = -self.beta * ((self.centers - s) ** 2).sum(1)  # (K,) = -β‖s-center_k‖²
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            tot += -np.log(p[y[t + 1]] + 1e-12)
            cnt += 1
        return tot / max(cnt, 1)

    def _fit_idx(self, theta, idxs):
        decay = np.clip(theta[:N], 0.0, 1.0)
        W = np.clip(theta[N:].reshape(N, N), -2.0, 2.0)
        ce = np.mean([self._ce_sentence(decay, W, i) for i in idxs])
        return -float(ce)  # fitness = -CE(高いほど良い)

    def train(self, theta):
        return self._fit_idx(theta, self.train_idx)

    def heldout(self, theta):
        return self._fit_idx(theta, self.test_idx)


# --------------------------------------------------------------------------- #
# optimizers(全て予算 B = train 評価回数で厳密に揃える。phase2 と同一構造)
# --------------------------------------------------------------------------- #
def _rand_theta(rng):
    return np.concatenate([rng.uniform(0, 1, N), rng.uniform(-2, 2, N * N) / np.sqrt(N)])


def opt_random(terrain, rng, B):
    best_t, best_f = None, -1e18
    for _ in range(B):
        th = _rand_theta(rng)
        f = terrain.train(th)
        if f > best_f:
            best_f, best_t = f, th
    return best_t


def opt_gradient(terrain, rng, B, eps=1e-3, lr=0.15, restarts=8):
    dim = N + N * N
    per_restart = max(1, B // restarts)
    best_t, best_f = None, -1e18
    used = 0
    while used < B:
        th = _rand_theta(rng)
        terrain.train(th); used += 1
        steps = max(1, (per_restart - 1) // dim)
        for _ in range(steps):
            if used >= B:
                break
            g = np.zeros(dim)
            base = terrain.train(th); used += 1
            for i in range(dim):
                if used >= B:
                    break
                tp = th.copy(); tp[i] += eps
                g[i] = (terrain.train(tp) - base) / eps; used += 1
            th = th + lr * g
            th[:N] = np.clip(th[:N], 0, 1)
            th[N:] = np.clip(th[N:], -2, 2)
        f = terrain.train(th); used += 1
        if f > best_f:
            best_f, best_t = f, th
    return best_t


def _descriptor(theta):
    decay = np.clip(theta[:N], 0, 1)
    W = theta[N:].reshape(N, N)
    d0 = float(np.mean(decay))
    d1 = float(np.tanh(np.mean(np.abs(W))))
    c0 = min(GRID - 1, int(d0 * GRID))
    c1 = min(GRID - 1, int(d1 * GRID))
    return (c0, c1)


def _to_gene(theta):
    decay = np.clip(theta[:N], 0.0, 1.0)
    W = np.clip(theta[N:].reshape(N, N), -2.0, 2.0)
    return C.CoupledNDGene.make(decay=decay, W=W)


def opt_mapelites(terrain, rng, B, gate=False, sigma=0.2, init=64, resample_cap=20):
    archive: dict = {}
    used = 0

    def _eval_and_place(th):
        nonlocal used
        f = terrain.train(th); used += 1
        cell = _descriptor(th)
        cur = archive.get(cell)
        if cur is None or f > cur[1]:
            archive[cell] = (th, f)

    def _admit(th):
        if not gate:
            return True
        return bool(C.cert_inf(_to_gene(th)))

    for _ in range(init):
        if used >= B:
            break
        th = _rand_theta(rng)
        if gate and not _admit(th):
            ok = False
            for _ in range(resample_cap):
                th = _rand_theta(rng)
                if _admit(th):
                    ok = True
                    break
            if not ok:
                th[N:] *= 0.3
        _eval_and_place(th)

    keys = list(archive.keys())
    while used < B and (keys or archive):
        keys = list(archive.keys())
        if not keys:
            th = _rand_theta(rng)
        else:
            parent = archive[keys[rng.integers(len(keys))]][0]
            th = parent + sigma * rng.standard_normal(parent.shape)
            th[:N] = np.clip(th[:N], 0, 1)
            th[N:] = np.clip(th[N:], -2, 2)
        if gate and not _admit(th):
            admitted = False
            for _ in range(resample_cap):
                if used >= B:
                    break
                parent = archive[keys[rng.integers(len(keys))]][0] if keys else _rand_theta(rng)
                th = parent + sigma * rng.standard_normal(parent.shape)
                th[:N] = np.clip(th[:N], 0, 1); th[N:] = np.clip(th[N:], -2, 2)
                if _admit(th):
                    admitted = True
                    break
            if not admitted:
                continue
        _eval_and_place(th)

    if not archive:
        return _rand_theta(rng)
    return max(archive.values(), key=lambda v: v[1])[0]


# --------------------------------------------------------------------------- #
# honest_eval 4 条件 AND (phase2_capability_terrain.py と同一)
# --------------------------------------------------------------------------- #
def honest_eval(a, b, alt="greater"):
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    mean_diff = float(diff.mean())
    n_seeds = len(diff)
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    paired_sign_delta = (wins - losses) / n_seeds
    if _HAVE_SCIPY and np.any(diff != 0):
        try:
            p = float(_wilcoxon(a, b, alternative=alt, zero_method="wilcox").pvalue)
        except Exception:
            p = float("nan")
    else:
        from math import comb
        k = wins; n = wins + losses
        p = float(sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)) if n > 0 else 1.0
    cond = {
        "diff_positive": mean_diff > 0,
        "wilcoxon_p_lt_0.05": (p < 0.05),
        "n_seeds_ge_15": n_seeds >= 15,
        "abs_paired_sign_delta_ge_0.147": abs(paired_sign_delta) >= 0.147,
    }
    return {
        "mean_diff": mean_diff, "p_value": p, "n_seeds": n_seeds,
        "wins": wins, "losses": losses, "paired_sign_delta": paired_sign_delta,
        "conditions": cond, "all_pass": all(cond.values()),
    }


def main():
    t_start = time.time()
    sents = get_real_hidden_sentences()

    # --- 多峰性検証(H-multimodal 前提, F9): seed-0 の実 CE 地形が多峰か単峰か ---
    import phase0_multimodality_instrument as F9  # noqa: E402
    terr0 = RealCETerrain(sents, np.random.default_rng(SEED0))
    dim = N + N * N

    def theta_field(z):  # z in [-1,1]^dim → train fitness
        th = np.empty(dim)
        th[:N] = (z[:N] + 1) / 2
        th[N:] = z[N:] * 2
        return terr0.train(th)

    basins = F9.find_basins(theta_field, np.random.default_rng(SEED0 + 1), dim, n_starts=40, merge_radius=0.5)
    n_basins = len(basins)
    print(f"[H-multimodal] 実 CE 地形(seed0)の theta 空間 basin 数 = {n_basins} "
          f"(>1 で多峰。単峰なら capability さらに立たず=計画予告)", flush=True)

    # --- seed ごとに 5 optimizer を同予算で走らせ held-out CE 比較 ---
    res = {"random": [], "gradient": [], "gradient_strong": [], "mapelites": [], "mapelites_gate": []}
    train_res = {k: [] for k in res}
    _name_off = {"random": 1, "gradient": 2, "gradient_strong": 5, "mapelites": 3, "mapelites_gate": 4}
    for s in range(N_SEEDS):
        terr = RealCETerrain(sents, np.random.default_rng(SEED0 + 100 + s))
        for name, fn in (("random", opt_random), ("gradient", opt_gradient),
                         ("gradient_strong", lambda t, r, B: opt_gradient(t, r, B, restarts=64)),
                         ("mapelites", lambda t, r, B: opt_mapelites(t, r, B, gate=False)),
                         ("mapelites_gate", lambda t, r, B: opt_mapelites(t, r, B, gate=True))):
            r = np.random.default_rng(SEED0 + 1000 + s * 17 + _name_off[name])
            best = fn(terr, r, BUDGET)
            res[name].append(terr.heldout(best))
            train_res[name].append(terr.train(best))
        print(f"  seed {s+1}/{N_SEEDS}: held-out fitness(=-CE) "
              f"rand={res['random'][-1]:.3f} grad={res['gradient'][-1]:.3f} "
              f"grad+={res['gradient_strong'][-1]:.3f} "
              f"ME={res['mapelites'][-1]:.3f} ME+gate={res['mapelites_gate'][-1]:.3f}", flush=True)

    cmp_me_grad = honest_eval(res["mapelites"], res["gradient"])
    cmp_me_gradstrong = honest_eval(res["mapelites"], res["gradient_strong"])
    cmp_me_rand = honest_eval(res["mapelites"], res["random"])
    cmp_gate_ungate = honest_eval(res["mapelites_gate"], res["mapelites"])
    cmp_grad_me = honest_eval(res["gradient"], res["mapelites"])

    # 識別力: random held-out が floor(-log K)/ceiling(0) に張り付いていないか
    floor = -np.log(K_CLUSTERS)  # fitness=-CE, uniform 予測 CE=log K → fitness=-log K
    rand_mean = float(np.mean(res["random"]))
    # 正規化識別スコア: 0=floor(random相当), 1=ceiling(CE=0)
    norm = (rand_mean - floor) / (0.0 - floor) if floor < 0 else 0.0
    discriminating = 0.05 < norm < 0.95

    me_beats_grad = cmp_me_grad["all_pass"]
    me_beats_gradstrong = cmp_me_gradstrong["all_pass"]
    if me_beats_grad and me_beats_gradstrong:
        verdict = "EXISTS (ME が gradient も meta-gate(restart64)も 4条件AND で上回る=実地形で genuine capability)"
    elif me_beats_grad and not me_beats_gradstrong:
        verdict = "ARTIFACT (ME は弱gradient を上回るが meta-gate で利得消失=navigability 現象 → guarantee 主軸が正当)"
    elif cmp_grad_me["all_pass"]:
        verdict = "NULL (gradient ≥ evolution = 実 LLM 地形で capability decisive NEGATIVE)"
    else:
        verdict = "NULL_TIE (有意差なし=進化は実 LLM 地形でも勾配/ランダムに勝てない; capability 立たず)"
    if not discriminating:
        verdict += f" [⚠地形 non-discriminating: norm_score={norm:.3f} (0=floor/1=ceiling)。証拠力低下]"

    elapsed = time.time() - t_start
    summary = {
        "meta": {"model": MODEL, "layer": LAYER, "n": N, "k_clusters": K_CLUSTERS,
                 "budget": BUDGET, "n_seeds": N_SEEDS, "grid": GRID, "seed0": SEED0,
                 "n_train_sent": N_TRAIN_SENT, "readout_scale": READOUT_SCALE,
                 "scipy": _HAVE_SCIPY, "elapsed_s": round(elapsed, 1),
                 "terrain": "REAL SmolLM2-135M hidden-derived next-cluster CE (synthetic でない=実地形 follow-up)"},
        "h_multimodal_basins": n_basins,
        "ce_floor_fitness": float(floor),
        "heldout_means": {k: float(np.mean(v)) for k, v in res.items()},
        "train_means": {k: float(np.mean(v)) for k, v in train_res.items()},
        "heldout_raw": {k: [float(x) for x in v] for k, v in res.items()},
        "ME_vs_gradient": cmp_me_grad,
        "ME_vs_gradient_strong_metagate": cmp_me_gradstrong,
        "ME_vs_random": cmp_me_rand,
        "gradient_vs_ME": cmp_grad_me,
        "gate_vs_ungate": cmp_gate_ungate,
        "random_heldout_mean": rand_mean,
        "norm_discrimination_score": float(norm),
        "terrain_discriminating": discriminating,
        "verdict": verdict,
    }
    print("\n=== 実 SmolLM2-CE 地形 capability verdict ===", flush=True)
    print(f"held-out fitness(=-CE) 平均: " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in res.items()), flush=True)
    print(f"CE floor(=-log K)={floor:.3f}  random norm_score={norm:.3f} "
          f"(0=floor/1=ceiling, discriminating={discriminating})", flush=True)
    print(f"ME vs gradient: diff={cmp_me_grad['mean_diff']:+.3f} p={cmp_me_grad['p_value']:.3f} "
          f"sign_delta={cmp_me_grad['paired_sign_delta']:+.3f} → 4条件AND={cmp_me_grad['all_pass']}", flush=True)
    print(f"ME vs gradient_strong (meta-gate restart64): diff={cmp_me_gradstrong['mean_diff']:+.3f} "
          f"p={cmp_me_gradstrong['p_value']:.3f} → 4条件AND={cmp_me_gradstrong['all_pass']}", flush=True)
    print(f"gradient vs ME: diff={cmp_grad_me['mean_diff']:+.3f} p={cmp_grad_me['p_value']:.3f} "
          f"→ 4条件AND={cmp_grad_me['all_pass']}", flush=True)
    print(f"gate vs ungate (ρ<1 が可塑性を殺すか): diff={cmp_gate_ungate['mean_diff']:+.3f} "
          f"p={cmp_gate_ungate['p_value']:.3f}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"(elapsed {elapsed:.0f}s)", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase2_capability_realce_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return summary


if __name__ == "__main__":
    main()
