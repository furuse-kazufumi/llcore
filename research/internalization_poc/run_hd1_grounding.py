# SPDX-License-Identifier: Apache-2.0
"""HD-1 接地 feasibility runner — 記憶形成機構を gradient 基質で比較する (設計 v3 準拠)。

正本設計 = HD1_GROUNDING_DESIGN.md v3 (敵対レビュー HD1_DESIGN_REVIEW_2026_06_07.md 反映済)。
本ファイルは **feasibility 段階** (設計 §5 step 2) の runner:
事前登録の最終化に必要な設計データ (F 条項 active か / ρ(step) plateau = measure 窓 /
OBSERVE β 感度 / REVIVE_ABLATE / Adam-sync ablation) を収集する。
**confirmatory 検定は行わない** — 本登録 (§5 step 3) は feasibility 結果で空欄を埋めてから
結果取得前 commit する 2 段階登録。

## arms (設計 §2)

- NONE      : 無拘束 (接地サニティ: ρ→1.95 帯の再現)
- EXO_init  : 初期化時のみ cert_inf 充足、以後放置
- ENDO      : cert_inf を cadence k=4 で検査 → fail で rollback (**Adam state も同期復元**)
- REVIVE    : gate では検査しない。独立判定 (cadence m=5 の empirical_rho≥1) が契約死を検出した
              時に死を記録 → raw_W ← c·raw_W (c を admit まで二分探索, 修復後 cert_inf 検査) →
              当該 layer の Adam state リセット → 続行。**死は踏む** (toy 意味論の忠実移植)
- OBSERVE   : cert_inf を呼ばない。proxy = probe rollout の state-norm 増大率 g_t (cert_inf と
              構造独立)。死イベント時 proxy の 10pct を閾値とし、超えたら直近 m step の更新を
              β 縮小 (θ ← θ_snap + β(θ − θ_snap))。feasibility は自己履歴 (pass1) のみ。
- REVIVE_ABLATE (feasibility 限定): REVIVE と同じ縮小を死と無関係に固定周期 (4m=20 step) で発動
              (純正則化との切り分け用反事実)
- ENDO_NOSYNC (feasibility 限定): Adam-sync ablation — rollback で core のみ復元し
              Adam state は流す (設計 §5 step 2 (c) の Adam state 交絡切り分け用反事実)

## 二層の死 (設計 §1)

- 契約死: empirical_rho ≥ 1 (from-below 実測, cadence m)。
- 実害死: state-separation probe — 初期差を箱端 (±1 成分) に取り T_probe step 流し
  ‖s_a−s_b‖ の平均 log 減衰率 ≥ 0 (縮まない = echo-state 喪失)。
全測定点の (step, rho_hat, sep_rate, proxy_g) 軌跡を記録 (plateau 同定 + E2 step 単位相関用)。

実行::  py -3.11 research/internalization_poc/run_hd1_grounding.py
出力::  research/internalization_poc/results_hd1_grounding_feas.json (resumable)
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_HD1 = _HERE.parents[0] / "highdim_evolution"
for _p in (str(_HD1), str(_HERE.parents[1] / "src"), str(_HERE.parents[0] / "verified_memory_poc"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hd1_highdim_evo as H  # noqa: E402  (基質・cert_inf・empirical_rho を流用)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from run_3arm_ab import _ensure_utf8_stdout  # noqa: E402

# ---- feasibility 設定 (設計 §5 step 2; 本走値は事前登録で別途固定) -------------
NS = [8, 32]
SEEDS = [2026 + i for i in range(4)]
ARMS = ["NONE", "EXO_init", "ENDO", "REVIVE", "OBSERVE", "REVIVE_ABLATE"]
OBSERVE_BETAS = [0.25, 0.5, 0.75]          # β 3 点 (OBSERVE のみ複数走行; 本走は 1 点固定)
CFG = dict(layers=1, d=48, T=48, B=12, lr=3e-3, grad_steps=120, max_chars=16000)
GATE_K = 4                                  # ENDO gate cadence (本走も固定)
MEASURE_M = 5                               # 独立判定/軌跡記録 cadence
ABLATE_PERIOD = 4 * MEASURE_M               # REVIVE_ABLATE の固定周期
OBSERVE_PCTL = 10.0                         # 死イベント proxy の閾値 percentile
SEP_T = 60                                  # state-separation probe の horizon
RHO_SAMPLES = 200                           # feasibility は軽量 (本走は rho_samples_for(n))
DEVICE = H.DEVICE


# ---- 測定 (二層の死 + proxy; numpy で core を直接反復 = cert_inf と独立な観測) ----
def _np_rollout_pair(decay, W, T, rng):
    """箱端初期差ペアを同一入力で流し、平均 log 減衰率と state-norm 増大率 g を返す。"""
    n = decay.shape[0]
    s_a = rng.choice([-1.0, 1.0], size=n)            # 箱端 (大擾乱 regime, 設計 §1)
    s_b = -s_a
    s = np.zeros(n)
    rates, norms = [], []
    for t in range(T):
        x = rng.uniform(-1, 1, n)
        d_prev = float(np.linalg.norm(s_a - s_b)) or 1e-12
        s_a = decay * s_a + (1 - decay) * np.tanh(W @ s_a + x)
        s_b = decay * s_b + (1 - decay) * np.tanh(W @ s_b + x)
        s = decay * s + (1 - decay) * np.tanh(W @ s + x)
        d_now = float(np.linalg.norm(s_a - s_b))
        if d_now > 0 and d_prev > 0:
            rates.append(np.log(d_now / d_prev))
        norms.append(float(np.linalg.norm(s)))
    half = len(norms) // 2
    g = float(np.log((np.mean(norms[half:]) + 1e-12) / (np.mean(norms[:half]) + 1e-12)))
    return float(np.mean(rates)), g


def measure_point(model, li_list, n, seed, step):
    """1 測定点: 契約死 (rho_hat) + 実害 (sep_rate) + proxy (g)。"""
    rng = np.random.default_rng(seed * 100003 + step)
    rho = max(H.empirical_rho(*model.core_np(li), n_samples=RHO_SAMPLES, seed=seed + li)
              for li in li_list)
    sep_rates, gs = [], []
    for li in li_list:
        d_, W_ = model.core_np(li)
        r, g = _np_rollout_pair(d_, W_, SEP_T, rng)
        sep_rates.append(r)
        gs.append(g)
    return {"step": step, "rho_hat": float(rho), "sep_rate": float(max(sep_rates)),
            "proxy_g": float(max(gs)),
            "contract_death": bool(rho >= 1.0), "harm_death": bool(max(sep_rates) >= 0.0)}


# ---- REVIVE 修復 (raw_W 空間, admit まで二分探索, 修復後 cert_inf 必須検査) -------
def repair_raw(model, li):
    raw = model.raw_W[li].detach().clone()
    lo, hi = 0.0, 1.0                       # c=hi が「縮小最小」側
    ok_c = 0.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        with torch.no_grad():
            model.raw_W[li].copy_(raw * mid)
        d_, W_ = model.core_np(li)
        if H.cert_inf(d_, W_):
            ok_c = mid
            lo = mid                        # もっと縮小を緩められるか (c↑)
        else:
            hi = mid
    with torch.no_grad():
        model.raw_W[li].copy_(raw * ok_c)
    d_, W_ = model.core_np(li)
    assert H.cert_inf(d_, W_), "repair failed to land in admit set"
    return ok_c


def _reset_adam_state(opt, params):
    for p in params:
        if p in opt.state:
            del opt.state[p]


# ---- 1 run (arm 別の訓練ループ) ------------------------------------------------
def run_arm(arm, n, seed, data, beta=0.5):
    tr, va, vocab = data
    cfg = dict(CFG)
    cfg["n"] = n
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    m = H.GatedRecurrentLM(vocab, n, cfg["layers"], cfg["d"]).to(DEVICE)
    li_list = list(range(cfg["layers"]))

    # 初期化: EXO_init / ENDO は cert_inf 充足から開始 (既存 init ループ流用)。
    # NONE / REVIVE / OBSERVE / ABLATE も同一初期化 (公平比較; 初期は全 arm admit 状態)。
    for li in li_list:
        for _ in range(50):
            d_, W_ = m.core_np(li)
            if H.cert_inf(d_, W_):
                break
            with torch.no_grad():
                m.raw_W[li].mul_(0.5)

    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    prev_core = None
    prev_opt_state = None
    snap_theta = None                       # OBSERVE 用 (m step 前の全パラメータ)
    death_proxy_log = []                    # OBSERVE: 死イベント時の proxy (自己履歴 pass1)
    obs_threshold = None
    traj = []
    deaths_contract = deaths_harm = repairs = rollbacks = avoids = 0

    def core_snapshot():
        return [(m.raw_decay[li].detach().clone(), m.raw_W[li].detach().clone()) for li in li_list]

    def core_restore(snap):
        with torch.no_grad():
            for li in li_list:
                m.raw_decay[li].copy_(snap[li][0])
                m.raw_W[li].copy_(snap[li][1])

    if arm == "ENDO":
        prev_core = core_snapshot()
        prev_opt_state = copy.deepcopy(opt.state_dict())
    if arm == "OBSERVE":
        snap_theta = copy.deepcopy(m.state_dict())

    for it in range(cfg["grad_steps"]):
        x, y = H.batches(tr, cfg["T"], cfg["B"], rng)
        opt.zero_grad()
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        opt.step()

        # --- ENDO: gate cadence k で cert 検査 → fail なら core+Adam を同期 rollback
        if arm == "ENDO" and ((it + 1) % GATE_K == 0 or it == cfg["grad_steps"] - 1):
            failed = any(not H.cert_inf(*m.core_np(li)) for li in li_list)
            if failed:
                core_restore(prev_core)
                opt.load_state_dict(prev_opt_state)
                rollbacks += 1
            else:
                prev_core = core_snapshot()
                prev_opt_state = copy.deepcopy(opt.state_dict())

        # --- REVIVE_ABLATE: 死と無関係の固定周期縮小 (反事実)
        if arm == "REVIVE_ABLATE" and (it + 1) % ABLATE_PERIOD == 0:
            for li in li_list:
                repair_raw(m, li)
                _reset_adam_state(opt, [m.raw_W[li]])
            repairs += 1

        # --- 独立判定 (cadence m): 全 arm 共通の測定 + REVIVE/OBSERVE の反応
        if (it + 1) % MEASURE_M == 0 or it == cfg["grad_steps"] - 1:
            pt = measure_point(m, li_list, n, seed, it + 1)
            traj.append(pt)
            if pt["contract_death"]:
                deaths_contract += 1
            if pt["harm_death"]:
                deaths_harm += 1

            if arm == "REVIVE" and pt["contract_death"]:
                death_proxy_log.append(pt["proxy_g"])
                for li in li_list:
                    repair_raw(m, li)
                    _reset_adam_state(opt, [m.raw_W[li]])
                repairs += 1

            if arm == "OBSERVE":
                if pt["contract_death"]:
                    death_proxy_log.append(pt["proxy_g"])
                    obs_threshold = float(np.percentile(death_proxy_log, OBSERVE_PCTL))
                if obs_threshold is not None and pt["proxy_g"] >= obs_threshold:
                    # 回避: 直近 m step 分の更新を β 縮小 (θ ← snap + β(θ − snap))
                    cur = m.state_dict()
                    with torch.no_grad():
                        for k_ in cur:
                            cur[k_].copy_(snap_theta[k_] + beta * (cur[k_] - snap_theta[k_]))
                    m.load_state_dict(cur)
                    avoids += 1
                snap_theta = copy.deepcopy(m.state_dict())

    ce = H.eval_ce(m, va, cfg["T"], cfg["B"], 4, np.random.default_rng(seed + 7))
    return {
        "arm": arm, "n": n, "seed": seed, "beta": (beta if arm == "OBSERVE" else None),
        "final_ce": float(ce),
        "deaths_contract": deaths_contract, "deaths_harm": deaths_harm,
        "repairs": repairs, "rollbacks": rollbacks, "avoids": avoids,
        "trajectory": traj,
    }


def main():
    _ensure_utf8_stdout()
    t0 = time.time()
    rpath = _HERE / "results_hd1_grounding_feas.json"
    data = H.make_data(CFG["max_chars"], False)
    tr, va, vocab = data
    uni = H.unigram_ce(tr, va, vocab)
    out = {"preregistration_note": (
        "feasibility 段階 (設計 v3 §5 step 2) — confirmatory 検定なし。"
        "目的 = F 条項 active 確認 / plateau 窓同定 / OBSERVE β 感度 / ABLATE / Adam-sync。"
        "本走の confirmatory 仮説・固定値は本登録 doc (結果取得前 commit) で確定する。"),
        "config": {**CFG, "NS": NS, "seeds": SEEDS, "gate_k": GATE_K, "measure_m": MEASURE_M,
                   "ablate_period": ABLATE_PERIOD, "observe_pctl": OBSERVE_PCTL,
                   "sep_T": SEP_T, "rho_samples": RHO_SAMPLES, "betas": OBSERVE_BETAS,
                   "unigram_ce": uni, "device": DEVICE},
        "records": []}
    done = set()
    if rpath.exists():
        try:
            prior = json.loads(rpath.read_text(encoding="utf-8"))
            out["records"] = prior.get("records", [])
            done = {(r["arm"], r["n"], r["seed"], r.get("beta")) for r in out["records"]}
            if done:
                print(f"resume: {len(done)} records done")
        except Exception:
            pass

    def save():
        rpath.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    for n in NS:
        for arm in ARMS:
            betas = OBSERVE_BETAS if arm == "OBSERVE" else [0.5]
            for beta in betas:
                for seed in SEEDS:
                    key = (arm, n, seed, (beta if arm == "OBSERVE" else None))
                    if key in done:
                        continue
                    r = run_arm(arm, n, seed, data, beta=beta)
                    out["records"].append(r)
                    save()
                    print(f"[{time.time()-t0:6.0f}s] n={n:3d} {arm:13s}"
                          + (f" b={beta}" if arm == "OBSERVE" else "      ")
                          + f" seed={seed} ce={r['final_ce']:.4f} "
                          f"d_con={r['deaths_contract']:2d} d_harm={r['deaths_harm']:2d} "
                          f"rep={r['repairs']} rb={r['rollbacks']} av={r['avoids']}", flush=True)

    # --- 要約 (記述のみ; 検定なし) ---
    print(f"\n=== HD-1 grounding feasibility (unigram={uni:.4f}, wall={time.time()-t0:.0f}s) ===")
    recs = out["records"]
    for n in NS:
        print(f"-- n={n}")
        for arm in ARMS:
            rs = [r for r in recs if r["arm"] == arm and r["n"] == n]
            if not rs:
                continue
            for beta in ({r["beta"] for r in rs} if arm == "OBSERVE" else [None]):
                sub = [r for r in rs if r.get("beta") == beta] if arm == "OBSERVE" else rs
                mean = lambda k_: float(np.mean([r[k_] for r in sub]))
                tag = f"{arm}" + (f"(b={beta})" if beta is not None else "")
                print(f"   {tag:18s} ce={mean('final_ce'):.4f} d_con={mean('deaths_contract'):5.1f} "
                      f"d_harm={mean('deaths_harm'):5.1f} rep={mean('repairs'):4.1f} "
                      f"rb={mean('rollbacks'):4.1f} av={mean('avoids'):4.1f}")
    # F 条項チェック (NONE の契約死 step 比率; 窓は全期間ベースの参考値)
    for n in NS:
        rs = [r for r in recs if r["arm"] == "NONE" and r["n"] == n]
        if rs:
            n_meas = len(rs[0]["trajectory"])
            frac = float(np.mean([r["deaths_contract"] / max(n_meas, 1) for r in rs]))
            print(f"F-check n={n}: NONE contract-death fraction (all-window) = {frac:.2%} "
                  f"({'ACTIVE' if frac >= 0.05 else 'inactive?'})")
    out["wall_seconds"] = round(time.time() - t0, 1)
    save()
    print(f"wrote {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
