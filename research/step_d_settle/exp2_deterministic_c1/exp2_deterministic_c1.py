# SPDX-License-Identifier: Apache-2.0
"""EXP2 (step_d_settle): 決定論 fitness で C1 midpoint-valley を測り step6 text proxy
地形が真に多峰(欺瞞)か単峰(滑らか)かを noise-free で判定する.

背景 (一次情報, 再導出しない):
- research/normalization_confound/NORMALIZATION_CONFOUND_VERDICT.md:
  C1 midpoint-valley が stochastic fitness の noise floor (谷閾 ≪ eval noise std) に
  埋もれて「計測不能」=判定(c)不確定だった。推奨=決定論 fitness で noise floor を回避。
- research/step6_real_proxy/exp7_method_comparison.py の `_acc_3param` / `_acc_perneuron`
  は ESN(固定 reservoir seed=0) + ridge readout(closed-form np.linalg.solve)で
  **rng を一切取らない決定論 fitness**。stochastic だったのは MAP-Elites の探索 rng のみ。
  → 同 gene には常に同 fitness が返る → 谷閾 0.05*|fit| を eval noise が侵さない。

設計 (read-only import 再利用, src/research 非改変):
- 診断器 = research/step_c_memory_tasks/landscape_map.py:multimodality_report (改造禁止)。
  _hillclimb の restart rng (base_seed+i) と midpoint 平均 rng (base_seed+999+k) は
  **探索方向の rng であって fitness 評価ノイズではない** → 決定論 fitness なら
  同 gene→同 fitness が保証される (eval_once は rng 引数を受けるが無視する)。
- 実 fitness = exp7_method_comparison の _acc_3param / _acc_perneuron を import 再利用。
  exp7 import 時に ESN(seed=0) + corpus(24000 chars) が構築される (exp7 と同 substrate)。

測る地形 (exp7 と同 substrate):
  (A) 3-param ESN gene (dim=3, [0,1]^3): exp6 で「滑らか broad ridge」視認地形。
  (B) per-neuron leak (dim=40): high-dim, MAP-E 本領域。多峰なら ③ load-bearing 余地。
診断器健全性 control:
  (C+) positive control = make_corridor_eval(0.16) を **決定論化** (noise を 0 に固定)
       → C1 が多峰 (is_multimodal=True) を返すべき (診断器が決定論地形で多峰検出可)。
  (C-) negative control = noiseless 単峰二次関数 → smooth (vf≈0) を返すべき。
  両 control 成立で diagnostic_valid=True (G3)。

破綻ゲート:
- G1: per-run wall_clock<900s。実 substrate は重い (3-param ~57ms/call, per-neuron ~39ms/call)
  ので **chunked-resumable** (cell 単位で partial JSON 追記、複数回起動で全 cell 完走)。
- G2: 決定論再現 = 同 gene で eval_once 2 回呼び bit 一致 (assert) +
  3 base_seed で is_multimodal flag 全一致。
- G3: diagnostic_valid = corridor control 多峰 ∧ quadratic control smooth。
- G4: 書込は本 dir 配下のみ、git 非実行。

判定 (3 値):
- 欺瞞(多峰)確定: 決定論 C1 で is_multimodal=True が全 base_seed 一致 ∧ 両 control 健全
  → 真の多峰 = ③ load-bearing 余地 (load_bearing)。
- 滑らか(単峰)確定 = null確定@power: eval noise std=0 ゆえ noisy-flat 偽陽性経路が消え、
  それでも valley_fraction<0.2 が全 seed 一致 → 実 text proxy 地形は真に単峰=③不要を
  noise-free で確定 (null_confirmed_at_power)。
- なお不確定: control が期待挙動を返さない (診断器が決定論でも分離不能) 場合のみ
  (still_inconclusive)。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]  # research/step_d_settle/exp2_deterministic_c1 -> repo
assert _REPO.name == "llcore", f"unexpected repo root: {_REPO}"

# read-only import paths (改造禁止の既存 lab)
for p in (
    _REPO / "research" / "step6_real_proxy",
    _REPO / "research" / "step_c_memory_tasks",
    _REPO / "research" / "step_c_applicability",
    _REPO / "research" / "step4_selection",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_ensure_utf8_stdout()

# ---- 既存 lab を read-only import (改造禁止) ----------------------------------
from landscape_map import multimodality_report  # noqa: E402

# G4 安全: 書込先 assert -------------------------------------------------------
_PARTIAL = _HERE / "exp2_partial.json"
_RESULTS = _HERE / "exp2_results.json"
_LOG = _HERE / "exp2_run.log"


def _assert_write_path(p: Path) -> None:
    rp = p.resolve()
    assert str(rp).startswith(str(_HERE)), f"write outside step_d_settle exp2 dir: {rp}"


# ---- 決定論性検証用ヘルパ -----------------------------------------------------
class _ZeroNormRng:
    """make_corridor_eval が呼ぶ rng.normal を 0 に固定 = corridor を noise-free 決定論化.

    corridor_eval は `+ rng.normal(0, _NOISE)` だけ rng を使う。これを 0 にすると
    ramp-with-dip の決定論地形になり、診断器が決定論下で多峰を検出できるか試せる。
    """

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return float(loc)
        return np.full(size, float(loc))


# ---- substrate (exp7 と同一) を lazy import (重い corpus/ESN 構築) ------------
_EXP7 = None
_CORRIDOR = None


def _exp7():
    global _EXP7
    if _EXP7 is None:
        import exp7_method_comparison as m  # noqa: E402  (imports build ESN+corpus)
        _EXP7 = m
    return _EXP7


def _corridor_det():
    """決定論化した corridor eval (d=0.16, noise=0). eval_once(g, rng) 形, rng は無視.

    重要 (honest disclosure): corridor は **C1 midpoint-valley 診断の正 control にならない**。
    検証結果 (本 dir log) — corridor は behavior=mean(g) のみで fitness が決まる genotypic
    corridor で、24-D の hill-climb(sigma=0.12)は random start(mean≈0.5)から mean を
    大域峰 0.9 まで押し上げられず **全 restart が同一 basin (mean≈0.542, fit≈0.679) に trap**
    する。distinct な第二峰が得られないので midpoint も同 mean→同 fit→valley_fraction=0。
    = corridor の欺瞞性は「MAP-E が reach できる範囲(=behavioral niche)」の欺瞞であって、
    C1 が測る「複数 basin + 谷」の欺瞞ではない。よって corridor は C1 の正 control 失格。
    本実験では corridor を「C1 診断の限界を示す診断ノート」として記録し (group=control_note)、
    C1 の正 control には **dim 一致の多峰 Gaussian (峰が gene 空間で分離) ** を使う。
    """
    global _CORRIDOR
    if _CORRIDOR is None:
        from exp_knob_sweep import make_corridor_eval  # noqa: E402
        raw = make_corridor_eval(0.16)
        zrng = _ZeroNormRng()
        _CORRIDOR = lambda g, rng=None: raw(g, zrng)  # noqa: E731
    return _CORRIDOR


def _make_multipeak(dim, peaks):
    """分離した複数 Gaussian 峰 (高さ不均一) の max = 真の多基底地形 (C1 正 control).

    峰が gene 空間で十分離れているので hill-climb は distinct 峰に収束し、その midpoint は
    谷に落ちる → C1 が is_multimodal=True を返すべき正 control (診断器の health check)。
    """
    peaks = [(np.asarray(c, float), float(h), float(w)) for c, h, w in peaks]

    def ev(g, rng=None):
        g = np.asarray(g, float)
        return float(max(h * np.exp(-np.sum((g - c) ** 2) / (2 * w * w))
                         for c, h, w in peaks))
    return ev


def _multipeak3():
    """dim=3 多峰 (ESN_3param と dim 一致の C1 正 control)."""
    return _make_multipeak(3, [
        ([0.2, 0.2, 0.2], 1.00, 0.12),
        ([0.8, 0.8, 0.8], 0.85, 0.12),
        ([0.2, 0.8, 0.5], 0.90, 0.12),
    ])


def _multipeak40():
    """dim=40 多峰 (ESN_perneuron40 と dim 一致の C1 正 control). 峰中心は固定 corner-ish."""
    rng = np.random.default_rng(7)
    cs = [rng.integers(0, 2, 40).astype(float) * 0.8 + 0.1 for _ in range(5)]
    peaks = [(c, 1.0 - 0.05 * i, 0.30) for i, c in enumerate(cs)]
    return _make_multipeak(40, peaks)


def _quadratic_eval(g, rng=None):
    """noiseless 単峰二次関数 (中心 0.5). smooth → vf≈0 が正解 (negative control)."""
    center = 0.5
    return -float(np.mean((np.asarray(g) - center) ** 2))


def _eval_3param(g, rng=None):
    return _exp7()._acc_3param(np.asarray(g, dtype=float))


def _eval_perneuron(g, rng=None):
    return _exp7()._acc_perneuron(np.asarray(g, dtype=float))


# ---- cell 定義 (landscape × budget × base_seed) -------------------------------
BASE_SEEDS = (20260530, 20260531, 20260601)

# 各 landscape: (label, eval_fn, dim, bounds, [budget configs], group)
# budget config = (n_restarts, n_evals, sigma)
# 実 substrate (A/B) は重いので budget を 12/24 restarts × n_evals=200/300 に抑え G1<900s。
# control (C) は安価 (analytic / 簡易) なので restarts 多めでも軽い。
_DIM_CORRIDOR = 24  # exp_knob_sweep.D


def _landscapes():
    return [
        # (A) 3-param ESN gene — exp6 で滑らか視認地形
        ("ESN_3param", _eval_3param, 3, (0.0, 1.0),
         [(12, 200, 0.12), (24, 300, 0.12)], "real"),
        # (B) per-neuron leak — high-dim, MAP-E 本領域
        ("ESN_perneuron40", _eval_perneuron, 40, (0.0, 1.0),
         [(12, 200, 0.12), (24, 300, 0.12)], "real"),
        # (C+ dim3) positive control: 分離多峰 Gaussian (ESN_3param と dim 一致) → 多峰必須
        ("ctrl_multipeak_dim3", _multipeak3(), 3, (0.0, 1.0),
         [(12, 200, 0.12), (24, 300, 0.12)], "control_pos"),
        # (C+ dim40) positive control: 分離多峰 Gaussian (ESN_perneuron40 と dim 一致) → 多峰必須
        ("ctrl_multipeak_dim40", _multipeak40(), 40, (0.0, 1.0),
         [(12, 200, 0.12), (24, 300, 0.12)], "control_pos"),
        # (C- dim3) negative control: noiseless 単峰二次 (dim3) → smooth であるべき
        ("ctrl_quadratic_dim3", _quadratic_eval, 3, (0.0, 1.0),
         [(12, 200, 0.12), (24, 300, 0.12)], "control_neg"),
        # (C- dim40) negative control: noiseless 単峰二次 (dim40) → smooth であるべき
        ("ctrl_quadratic_dim40", _quadratic_eval, 40, (0.0, 1.0),
         [(12, 200, 0.12), (24, 300, 0.12)], "control_neg"),
        # (note) corridor d=0.16 決定論版 = C1 の正 control にならない事を記録する診断ノート
        # (genotypic corridor は単一 basin trap で C1 谷を出さない, 上記 _corridor_det docstring)
        ("note_corridor_d016", _corridor_det(), _DIM_CORRIDOR, (0.0, 1.0),
         [(24, 300, 0.12)], "control_note"),
    ]


def _cell_id(label, nr, ne, base_seed):
    return f"{label}|nr={nr}|ne={ne}|seed={base_seed}"


# ---- determinism self-check (G2): 同 gene 2 回で bit 一致 ----------------------
def _check_determinism(eval_fn, dim, bounds, label):
    lo, hi = bounds
    rng = np.random.default_rng(424242)
    g = lo + (hi - lo) * rng.random(dim)
    a1 = eval_fn(g, np.random.default_rng(1))
    a2 = eval_fn(g, np.random.default_rng(99999))
    assert a1 == a2, f"NON-DETERMINISTIC fitness for {label}: {a1} != {a2}"
    return float(a1)


# ---- eval-noise std (決定論なら exactly 0) ------------------------------------
def _eval_noise_std(eval_fn, dim, bounds, K=12, base_seed=20260530):
    lo, hi = bounds
    rng0 = np.random.default_rng(base_seed)
    g = lo + (hi - lo) * rng0.random(dim)
    vals = np.array([eval_fn(g, np.random.default_rng(base_seed + 1000 + k))
                     for k in range(K)])
    return float(vals.mean()), float(vals.std())


def _load_partial():
    if _PARTIAL.exists():
        try:
            return json.loads(_PARTIAL.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_partial(d):
    _assert_write_path(_PARTIAL)
    _PARTIAL.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def log(msg, *, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def run_cells(per_run_budget_s=820.0, smoke=False):
    """未完了 cell を per_run_budget_s 内で走らせ partial に追記 (chunked-resumable)."""
    _assert_write_path(_LOG)
    fh = open(_LOG, "a", encoding="utf-8")
    t_start = time.time()
    partial = _load_partial()
    cells = partial.setdefault("cells", {})
    meta = partial.setdefault("_meta", {})
    meta.setdefault("substrate", "exp7 ESN(seed=0)+corpus(24000 chars) ridge readout")
    meta.setdefault("diagnostic", "landscape_map.multimodality_report (read-only)")
    meta.setdefault("base_seeds", list(BASE_SEEDS))
    meta.setdefault("budget_note",
                    "real substrate (A/B) は重いので restarts 12/24 × n_evals 200/300 に抑制。"
                    "silent truncation 禁止: 未完 cell は partial に残し再起動で完走。")

    landscapes = _landscapes()

    log(f"EXP2 start (smoke={smoke}) per_run_budget={per_run_budget_s}s", fh=fh)

    # determinism + eval-noise self-check (一度だけ, 安価)
    if "self_check" not in partial:
        sc = {}
        for label, ev, dim, bounds, _b, group in landscapes:
            fval = _check_determinism(ev, dim, bounds, label)
            m, sd = _eval_noise_std(ev, dim, bounds)
            sc[label] = {"det_fitness": fval, "eval_noise_mean": m,
                         "eval_noise_std": sd, "dim": dim, "group": group}
            # 注: 同 gene K 回評価は bit 一致 (det_fitness 検証 a1==a2 で確認済)。
            # np.std が ~1e-16 を返すのは constant 配列の分散計算の ULP 誤差であって
            # 評価ノイズではない。閾 1e-12 (谷閾 ~0.05*|fit|≈0.03 を 11 桁下回る)。
            log(f"self-check {label}: det_fit={fval:.4f} eval_noise_std={sd:.3e} "
                f"(machine-epsilon; valley_thr ~0.05*|fit|)", fh=fh)
            assert sd < 1e-12, (f"eval_noise_std={sd:.3e} >=1e-12 for {label} "
                                f"(not deterministic enough)")
        partial["self_check"] = sc
        _save_partial(partial)

    # cell リスト構築
    all_cells = []
    for label, ev, dim, bounds, budgets, group in landscapes:
        bud = budgets[:1] if smoke else budgets
        seeds = (BASE_SEEDS[:1],) if smoke else (BASE_SEEDS,)
        seedlist = BASE_SEEDS[:1] if smoke else BASE_SEEDS
        for (nr, ne, sigma) in bud:
            nr_eff = 4 if smoke else nr
            ne_eff = 60 if smoke else ne
            for bs in seedlist:
                cid = _cell_id(label, nr_eff, ne_eff, bs)
                all_cells.append((cid, label, ev, dim, bounds, nr_eff, ne_eff, sigma, bs, group))

    done_before = sum(1 for c in all_cells if c[0] in cells)
    log(f"total cells={len(all_cells)} done_before={done_before}", fh=fh)

    ran = 0
    for (cid, label, ev, dim, bounds, nr, ne, sigma, bs, group) in all_cells:
        if cid in cells:
            continue
        if time.time() - t_start > per_run_budget_s:
            log(f"per-run budget reached, stopping. {ran} cells this run.", fh=fh)
            break
        t0 = time.time()
        rep = multimodality_report(ev, dim=dim, bounds=bounds,
                                   n_restarts=nr, n_evals=ne, sigma=sigma,
                                   base_seed=bs)
        dt = time.time() - t0
        cells[cid] = {
            "label": label, "group": group, "dim": dim,
            "n_restarts": nr, "n_evals": ne, "sigma": sigma, "base_seed": bs,
            "valley_fraction": rep["valley_fraction"],
            "n_optima": rep["n_optima"],
            "is_multimodal": rep["is_multimodal"],
            "wall_s": round(dt, 2),
        }
        ran += 1
        log(f"cell {cid}: vf={rep['valley_fraction']:.3f} "
            f"n_optima={rep['n_optima']} is_mm={rep['is_multimodal']} ({dt:.1f}s)", fh=fh)
        _save_partial(partial)

    total_run_s = time.time() - t_start
    meta["last_run_wall_s"] = round(total_run_s, 2)
    meta["last_run_cells"] = ran
    _save_partial(partial)
    log(f"run finished: {ran} cells, wall={total_run_s:.1f}s "
        f"(per-run G1 {'OK' if total_run_s < 900 else 'OVER'})", fh=fh)

    remaining = sum(1 for c in all_cells if c[0] not in cells)
    fh.close()
    return partial, remaining, len(all_cells)


def _aggregate_and_verdict(partial):
    """全 cell 完走後に landscape 別集計 + 3 値判定."""
    cells = partial.get("cells", {})
    by_label = {}
    for c in cells.values():
        by_label.setdefault(c["label"], []).append(c)

    summary = {}
    for label, lst in by_label.items():
        group = lst[0]["group"]
        flags = sorted(set(bool(c["is_multimodal"]) for c in lst))
        vf = [c["valley_fraction"] for c in lst]
        flag_consistent = (len(flags) == 1)
        summary[label] = {
            "group": group,
            "n_cells": len(lst),
            "is_multimodal_set": flags,
            "is_multimodal_consistent": flag_consistent,
            "valley_fraction_min": min(vf),
            "valley_fraction_max": max(vf),
            "valley_fraction_mean": round(float(np.mean(vf)), 4),
            "n_optima_range": [min(c["n_optima"] for c in lst),
                               max(c["n_optima"] for c in lst)],
            # consistent multimodal = 全 cell で True
            "all_multimodal": all(c["is_multimodal"] for c in lst),
            "all_smooth": all(not c["is_multimodal"] for c in lst),
        }

    # G3 diagnostic validity
    pos = summary.get("ctrl_corridor_d016", {})
    neg = summary.get("ctrl_quadratic_unimodal", {})
    diagnostic_valid = bool(pos.get("all_multimodal")) and bool(neg.get("all_smooth"))

    # 実 substrate 判定
    real_labels = [k for k, v in summary.items() if v["group"] == "real"]
    verdicts = {}
    overall_real_load_bearing = False
    overall_real_smooth = True
    for lbl in real_labels:
        s = summary[lbl]
        consistent = s["is_multimodal_consistent"]
        if not diagnostic_valid:
            v = "still_inconclusive"
        elif s["all_multimodal"] and consistent:
            v = "load_bearing"            # 欺瞞(多峰)確定
            overall_real_load_bearing = True
            overall_real_smooth = False
        elif s["all_smooth"] and consistent:
            v = "null_confirmed_at_power"  # 滑らか(単峰)確定 = ③不要
        else:
            v = "still_inconclusive"       # flag が seed 間不一致
            overall_real_smooth = False
        verdicts[lbl] = v

    # overall third_axis_verdict
    if not diagnostic_valid:
        overall = "still_inconclusive"
    elif overall_real_load_bearing:
        overall = "load_bearing"
    elif all(verdicts[l] == "null_confirmed_at_power" for l in real_labels):
        overall = "null_confirmed_at_power"
    else:
        overall = "still_inconclusive"

    return {
        "per_landscape": summary,
        "diagnostic_valid": diagnostic_valid,
        "control_pos_corridor": {
            "all_multimodal": pos.get("all_multimodal"),
            "vf_mean": pos.get("valley_fraction_mean"),
        },
        "control_neg_quadratic": {
            "all_smooth": neg.get("all_smooth"),
            "vf_mean": neg.get("valley_fraction_mean"),
        },
        "real_landscape_verdicts": verdicts,
        "third_axis_verdict": overall,
    }


def finalize():
    partial = _load_partial()
    agg = _aggregate_and_verdict(partial)
    sc = partial.get("self_check", {})
    out = {
        "experiment": "exp2_deterministic_c1",
        "_meta": partial.get("_meta", {}),
        "self_check": sc,
        "n_cells_total": len(partial.get("cells", {})),
        "cells": partial.get("cells", {}),
        "aggregate": agg,
        # eval noise が全 landscape で machine-epsilon 以下 (<1e-12) = 決定論。
        # 谷閾 0.05*|fit|~0.03 を 11 桁下回る → noise floor 完全回避を実測で示す。
        "eval_noise_std_max": max((v["eval_noise_std"] for v in sc.values()), default=0.0),
        "eval_noise_below_machine_eps": all(v["eval_noise_std"] < 1e-12 for v in sc.values()),
        "honest_caveat": (
            "決定論 fitness (eval_noise_std=0 全 landscape 実測) で normalization_confound の "
            "noise floor (谷閾 ≪ eval noise std で計測不能) を構造的に回避した。"
            "本 proxy は reservoir computing (固定 reservoir 力学 + ridge readout closed-form) で "
            "あって backprop で学習する full LLM ではない (exp7 と同 honest 留保)。"
            "判定は exp7 と同 substrate (corpus 24000 chars, ESN seed=0, N=40) 上で成立。"
        ),
    }
    _assert_write_path(_RESULTS)
    _RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv):
    smoke = "--smoke" in argv
    budget = 820.0
    for a in argv:
        if a.startswith("--budget="):
            budget = float(a.split("=", 1)[1])
    partial, remaining, total = run_cells(per_run_budget_s=budget, smoke=smoke)
    print(f"\n[run] cells done={total - remaining}/{total} remaining={remaining}")
    if remaining == 0:
        out = finalize()
        agg = out["aggregate"]
        print("\n=== EXP2 FINAL ===")
        print(f"  eval_noise_std_max = {out['eval_noise_std_max']:.2e} "
              f"(below_machine_eps={out['eval_noise_below_machine_eps']})")
        print(f"  diagnostic_valid = {agg['diagnostic_valid']} "
              f"(corridor multimodal={agg['control_pos_corridor']['all_multimodal']}, "
              f"quadratic smooth={agg['control_neg_quadratic']['all_smooth']})")
        for lbl, v in agg["real_landscape_verdicts"].items():
            s = agg["per_landscape"][lbl]
            print(f"  {lbl}: vf_mean={s['valley_fraction_mean']} "
                  f"is_mm_set={s['is_multimodal_set']} consistent={s['is_multimodal_consistent']} "
                  f"-> {v}")
        print(f"  THIRD AXIS VERDICT = {agg['third_axis_verdict']}")
        print(f"  results -> {_RESULTS}")
    else:
        print(f"[resumable] re-run to continue ({remaining} cells left). "
              f"partial -> {_PARTIAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
