# SPDX-License-Identifier: Apache-2.0
"""Phase 2: capability terrain-bet (F12) — 進化は多峰地形で勾配/ランダムに勝つか。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) §⑧ H-multimodal / H-EXISTS / meta-gate(BG10) / §⑩ Phase 2:
  capability 副線(必須)。**多峰地形**で ρ<1-gate 付き MAP-Elites が同予算の gradient/random を
  **held-out fitness** で honest_eval 4 条件 AND で上回る constellation が在るか。meta-gate で
  EXISTS / NULL / ARTIFACT を 1 つ確定する。

事前知見(honest): M3 で capability は decisive NEGATIVE(進化は perplexity/CE で勾配に勝たない)、
計画も「最尤 NULL」。本実験は **capability NEGATIVE を proper power で確定する confirmatory** 寄り。
驚きの EXISTS が出れば内訳を疑う([[feedback_benchmark_honest_disclosure]])。

設計(small-n per-component, $0/CPU, synthetic 多峰地形):
  - 基質 = CoupledNDGene (n)。地形 = behavior 空間の **K-basin max-of-Gaussian**(多峰を構成的に保証、
    F9 instrument の find_basins で多峰性を検証 = H-multimodal の前提確認)。
  - behavior(g, X) = 固定入力列 X で T step 回した状態の時間平均特徴(n 次元)。fitness は微分可能(勾配可)。
  - **train/held-out 分離**: train fitness は X_train、held-out は X_test(別 seed)で評価。best-by-train gene の
    held-out fitness を比較 = 過学習(地形 gaming)を排除した汎化能力。
  - 4 optimizer を **同予算 B(train-fitness 評価回数)** で:
      random(B sample)/ gradient(有限差分 ascent + restart)/ MAP-Elites(QD archive)/
      MAP-Elites+gate(ρ<1 cert_inf gate 付き verified 変種)。
  - **honest_eval 4 条件 AND**(計画 §8.2): diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15 ∧
    |paired_sign_delta|≥0.147。S seed で paired 比較。
  - **meta-gate(BG10)**: ME が gradient を上回ったら gradient-on-same-terrain(restart 増)で
    利得が消えるか確認 → 消えれば ARTIFACT(navigability 現象)、消えねば EXISTS。勝たねば NULL。

honest 留保:
  - synthetic 多峰地形であり **実 SmolLM2-CE 損失地形ではない**(実 LLM CE terrain は heavier follow-up)。
    本実験は「多峰が保証された地形ですら進化が勾配に勝つか」の clean probe。負ければ実地形でも期待薄。
  - paired_sign_delta = net-win-fraction (n_wins−n_losses)/n_seeds(教科書 Cliff's delta ではない、計画 §⑬整合)。
  - 予算 = train-fitness 評価回数で厳密に揃える(gradient の有限差分も 1 評価ずつ計上)。
"""
from __future__ import annotations

import json
import os
import sys

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

N = 4                  # small-n per-component
T = 24                 # behavior 軌道長
K_BASINS = 6           # 地形の basin 数(多峰)
SIGMA = 0.15           # basin 幅(狭め=識別力確保。random が天井に張り付かない難度。結果に合わせ調整=p-hack なので原理的に固定)
BUDGET = 2000          # 1 optimizer/seed あたりの train-fitness 評価予算
N_SEEDS = 20           # ≥15
GRID = 12              # MAP-Elites archive grid (GxG)
SEED0 = 20260609


# --------------------------------------------------------------------------- #
# 地形(多峰 fitness)+ behavior
# --------------------------------------------------------------------------- #
def behavior(theta, X):
    """gene(theta=[decay(n),W(n^2)]) を入力列 X で回し、状態の時間平均 |s| 特徴(n 次元)を返す。"""
    g = _to_gene(theta)
    s = np.zeros(N)
    acc = np.zeros(N)
    for t in range(X.shape[0]):
        s = C.step(g, s, X[t])
        acc += np.abs(s)
    return acc / X.shape[0]


def _to_gene(theta):
    decay = np.clip(theta[:N], 0.0, 1.0)
    W = np.clip(theta[N:].reshape(N, N), -2.0, 2.0)
    return C.CoupledNDGene.make(decay=decay, W=W)


class Terrain:
    """K-basin max-of-Gaussian over behavior 空間(多峰を構成的に保証)。"""
    def __init__(self, rng):
        self.centers = rng.uniform(0.0, 1.0, (K_BASINS, N))   # behavior 空間の basin 中心
        self.X_train = rng.uniform(-1, 1, (T, N))
        self.X_test = rng.uniform(-1, 1, (T, N))              # held-out 入力

    def _fit(self, theta, X):
        b = behavior(theta, X)
        d2 = ((self.centers - b) ** 2).sum(axis=1)
        return float(np.exp(-d2 / (2 * SIGMA ** 2)).max())

    def train(self, theta):
        return self._fit(theta, self.X_train)

    def heldout(self, theta):
        return self._fit(theta, self.X_test)


# --------------------------------------------------------------------------- #
# optimizers(全て予算 B = train 評価回数で厳密に揃える)
# --------------------------------------------------------------------------- #
def _rand_theta(rng):
    return np.concatenate([rng.uniform(0, 1, N), rng.uniform(-2, 2, N * N) / np.sqrt(N)])


def opt_random(terrain, rng, B):
    best_t, best_f = None, -1.0
    for _ in range(B):
        th = _rand_theta(rng)
        f = terrain.train(th)
        if f > best_f:
            best_f, best_t = f, th
    return best_t


def opt_gradient(terrain, rng, B, eps=1e-3, lr=0.15, restarts=8):
    """有限差分(前進)勾配 ascent + restart。1 評価ずつ予算計上。"""
    dim = N + N * N
    per_restart = max(1, B // restarts)
    best_t, best_f = None, -1.0
    used = 0
    while used < B:
        th = _rand_theta(rng)
        f0 = terrain.train(th); used += 1
        steps = max(1, (per_restart - 1) // (dim + 0))
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
    """MAP-Elites の behavior descriptor(2D cell): (mean decay, tanh(mean|W|))。"""
    decay = np.clip(theta[:N], 0, 1)
    W = theta[N:].reshape(N, N)
    d0 = float(np.mean(decay))
    d1 = float(np.tanh(np.mean(np.abs(W))))
    c0 = min(GRID - 1, int(d0 * GRID))
    c1 = min(GRID - 1, int(d1 * GRID))
    return (c0, c1)


def opt_mapelites(terrain, rng, B, gate=False, sigma=0.2, init=64, resample_cap=20):
    """MAP-Elites QD archive。gate=True で ρ<1(cert_inf)を満たす個体のみ admit(verified 変種)。"""
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

    # init
    for _ in range(init):
        if used >= B:
            break
        th = _rand_theta(rng)
        if gate and not _admit(th):
            # gated init: resample until cert_inf or fallback to shrunk W
            ok = False
            for _ in range(resample_cap):
                th = _rand_theta(rng)
                if _admit(th):
                    ok = True
                    break
            if not ok:
                th[N:] *= 0.3  # known-safe-ish shrink
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
# honest_eval 4 条件 AND
# --------------------------------------------------------------------------- #
def honest_eval(a, b, alt="greater"):
    """paired held-out: a (例 ME) が b (例 gradient) を上回るか 4 条件 AND。a,b: shape (S,)。"""
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
        # fallback: 片側 sign test (二項)
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
    rng = np.random.default_rng(SEED0)

    # 多峰性検証(H-multimodal 前提) — F9 instrument で地形の basin 数を確認
    sys.path.insert(0, os.path.dirname(__file__))
    import phase0_multimodality_instrument as F9  # noqa: E402
    terr0 = Terrain(np.random.default_rng(SEED0))
    # theta 空間上の多峰性を F9 find_basins で検証(theta を [-1,1]^dim に正規化した proxy)
    dim = N + N * N
    def theta_field(z):  # z in [-1,1]^dim → theta
        th = np.empty(dim)
        th[:N] = (z[:N] + 1) / 2
        th[N:] = z[N:] * 2
        return terr0.train(th)
    basins = F9.find_basins(theta_field, np.random.default_rng(SEED0 + 1), dim, n_starts=40, merge_radius=0.5)
    n_basins = len(basins)
    print(f"[H-multimodal] 地形の theta 空間 basin 数 = {n_basins} (>1 で多峰=capability 前提成立)", flush=True)

    # seed ごとに 4 optimizer を同予算で走らせ held-out 比較
    res = {"random": [], "gradient": [], "gradient_strong": [], "mapelites": [], "mapelites_gate": []}
    train_res = {k: [] for k in res}
    _name_off = {"random": 1, "gradient": 2, "gradient_strong": 5, "mapelites": 3, "mapelites_gate": 4}
    for s in range(N_SEEDS):
        terr = Terrain(np.random.default_rng(SEED0 + 100 + s))
        for name, fn in (("random", opt_random), ("gradient", opt_gradient),
                         ("gradient_strong", lambda t, r, B: opt_gradient(t, r, B, restarts=64)),  # meta-gate: 多 restart
                         ("mapelites", lambda t, r, B: opt_mapelites(t, r, B, gate=False)),
                         ("mapelites_gate", lambda t, r, B: opt_mapelites(t, r, B, gate=True))):
            r = np.random.default_rng(SEED0 + 1000 + s * 17 + _name_off[name])
            best = fn(terr, r, BUDGET)
            res[name].append(terr.heldout(best))
            train_res[name].append(terr.train(best))
        print(f"  seed {s+1}/{N_SEEDS}: held-out "
              f"rand={res['random'][-1]:.3f} grad={res['gradient'][-1]:.3f} "
              f"grad+={res['gradient_strong'][-1]:.3f} "
              f"ME={res['mapelites'][-1]:.3f} ME+gate={res['mapelites_gate'][-1]:.3f}", flush=True)

    # honest_eval
    cmp_me_grad = honest_eval(res["mapelites"], res["gradient"])
    cmp_me_gradstrong = honest_eval(res["mapelites"], res["gradient_strong"])  # meta-gate(BG10)
    cmp_me_rand = honest_eval(res["mapelites"], res["random"])
    cmp_gate_ungate = honest_eval(res["mapelites_gate"], res["mapelites"])  # gate が可塑性を殺すか
    cmp_grad_me = honest_eval(res["gradient"], res["mapelites"])            # 逆向き(gradient 優位か)

    # 識別力チェック(地形が天井/床でないか)
    rand_mean = float(np.mean(res["random"]))
    discriminating = 0.05 < rand_mean < 0.95

    # meta-gate verdict(BG10)
    me_beats_grad = cmp_me_grad["all_pass"]
    me_beats_gradstrong = cmp_me_gradstrong["all_pass"]
    if me_beats_grad and me_beats_gradstrong:
        verdict = "EXISTS (ME が gradient も gradient_strong(meta-gate, restart64)も 4条件AND で上回る=genuine capability)"
    elif me_beats_grad and not me_beats_gradstrong:
        verdict = "ARTIFACT (ME は弱gradient を上回るが gradient_strong=多restart で利得消失=navigability 現象であり capability でない → guarantee 主軸が正当)"
    elif cmp_grad_me["all_pass"]:
        verdict = "NULL (gradient ≥ evolution = capability decisive NEGATIVE; M3 を多峰地形で再確認)"
    else:
        verdict = "NULL_TIE (有意差なし=進化は勾配/ランダムに勝てない; capability 立たず)"
    if not discriminating:
        verdict += f" [⚠地形 non-discriminating: random held-out 平均={rand_mean:.3f}=天井/床。verdict の証拠力低下]"

    summary = {
        "meta": {"n": N, "T": T, "K_basins": K_BASINS, "budget": BUDGET, "n_seeds": N_SEEDS,
                 "grid": GRID, "seed0": SEED0, "scipy": _HAVE_SCIPY,
                 "terrain": "synthetic K-basin max-of-Gaussian (実SmolLM2-CE地形ではない=clean probe)"},
        "h_multimodal_basins": n_basins,
        "heldout_means": {k: float(np.mean(v)) for k, v in res.items()},
        "train_means": {k: float(np.mean(v)) for k, v in train_res.items()},
        "ME_vs_gradient": cmp_me_grad,
        "ME_vs_gradient_strong_metagate": cmp_me_gradstrong,
        "ME_vs_random": cmp_me_rand,
        "gradient_vs_ME": cmp_grad_me,
        "gate_vs_ungate": cmp_gate_ungate,
        "random_heldout_mean": rand_mean,
        "terrain_discriminating": discriminating,
        "verdict": verdict,
    }
    print("\n=== capability terrain-bet verdict ===", flush=True)
    print(f"held-out 平均: " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in res.items()), flush=True)
    print(f"ME vs gradient: diff={cmp_me_grad['mean_diff']:+.3f} p={cmp_me_grad['p_value']:.3f} "
          f"sign_delta={cmp_me_grad['paired_sign_delta']:+.3f} → 4条件AND={cmp_me_grad['all_pass']}", flush=True)
    print(f"ME vs gradient_strong (meta-gate restart64): diff={cmp_me_gradstrong['mean_diff']:+.3f} "
          f"p={cmp_me_gradstrong['p_value']:.3f} → 4条件AND={cmp_me_gradstrong['all_pass']}", flush=True)
    print(f"gradient vs ME: diff={cmp_grad_me['mean_diff']:+.3f} p={cmp_grad_me['p_value']:.3f} "
          f"→ 4条件AND={cmp_grad_me['all_pass']}", flush=True)
    print(f"地形識別力: random held-out 平均={rand_mean:.3f} (0.05-0.95 で discriminating={discriminating})", flush=True)
    print(f"gate vs ungate (ρ<1 が可塑性を殺すか): diff={cmp_gate_ungate['mean_diff']:+.3f} "
          f"p={cmp_gate_ungate['p_value']:.3f}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase2_capability_terrain_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return summary


if __name__ == "__main__":
    main()
