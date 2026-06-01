# SPDX-License-Identifier: Apache-2.0
"""統計的検出力 自己監査 — 共有ユーティリティ (read-only import のみ).

本 module は research/statistical_power_audit/ 配下の 4 スクリプト
(calibrate_known_positive / repower_real_negatives / ablate_suppression_knobs /
type1_guard_sweep) が共通で使う:

- import path 設定 (src/llcore と research/step4_selection を read-only で path に)
- UTF-8 stdout 強制
- 既知真陽性 corridor (exp_knob_sweep) の薄い wrapper
- 完全 strict gate (4 条件 AND) を任意閾値で再計算するヘルパ (src 本体は無改変)
- 実 negative の per-seed paired delta を既存 results JSON から復元
- src/ ツリーの不変監査 (G4 破綻ゲート) と書込先 path の assert (research 配下のみ)

src/llcore のシンボルは **import 再利用のみ** (改造禁止)。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# --- repo root / 隔離 dir ---
_THIS = Path(__file__).resolve()
AUDIT_DIR = _THIS.parent  # research/statistical_power_audit/
REPO_ROOT = AUDIT_DIR.parents[1]  # D:/projects/llcore
SRC_DIR = REPO_ROOT / "src"
STEP4_DIR = REPO_ROOT / "research" / "step4_selection"
KNOB_SWEEP_DIR = REPO_ROOT / "research" / "step_c_applicability"
EA_DIR = REPO_ROOT / "research" / "ea_multitask"

# read-only import path (src 改造禁止)
for p in (str(SRC_DIR), str(STEP4_DIR), str(KNOB_SWEEP_DIR), str(EA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


ensure_utf8_stdout()

# --- read-only import: src/llcore の honest_eval ---
from llcore.evolution.honest_eval import (  # noqa: E402
    _paired_p,  # 片側 paired Wilcoxon (scipy 不在時 符号検定)
    _paired_sign_delta,  # (#正-#負)/n
    honest_reevaluate,
)

# --- read-only import: exp_knob_sweep (既知真陽性 corridor + CRN runner) ---
from exp_knob_sweep import (  # noqa: E402
    D as CORRIDOR_D,
    behavior_mean,
    make_corridor_eval,
    run_methods_crn,
)


# ===========================================================================
# 完全 strict gate を任意閾値で再計算 (src 無改変。返り値の生統計を組み直すだけ)
# ===========================================================================


def textbook_cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    """教科書的 Cliff's delta = (#(a_i > b_j) - #(a_i < b_j)) / (n*m).

    paired_sign_delta ((#正-#負)/n) とは別物。min_effect=0.147 がどちらの尺度で
    効いているかを可視化するため両方を出す。値域 [-1,1]。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    n_pairs = a.size * b.size
    if n_pairs == 0:
        return 0.0
    return (gt - lt) / n_pairs


@dataclass
class GateEval:
    """a が b を上回るかの完全 strict gate 評価 (任意閾値)."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    diff: float
    win_rate: float
    wilcoxon_p: float
    paired_sign_delta: float
    cliff_delta_textbook: float
    n_seeds: int
    cohen_dz: float
    # 各条件の個別 pass (どの条件が最初に倒れるかの分解用)
    cond_diff_pos: bool
    cond_p: bool
    cond_n: bool
    cond_effect: bool
    passes: bool  # 4 条件 AND


def eval_gate(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
    *,
    alpha: float = 0.05,
    min_seeds: int = 15,
    min_effect: float = 0.147,
    effect_metric: str = "paired_sign",  # 'paired_sign' (現行 gate) | 'cliff'
) -> GateEval:
    """src の honest_eval._paired_p / _paired_sign_delta を流用し、任意閾値で gate を再計算.

    honest_eval / strict_compare / exp_knob_sweep.strict_gate と **完全同一の 4 条件 AND**
    (diff>0 ∧ 片側 Wilcoxon p<alpha ∧ n>=min_seeds ∧ |effect|>=min_effect)。
    閾値だけ research 側で振る (src 本体は無改変)。effect_metric で床に課す効果量尺度を選べる。
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    deltas = a - b
    diff = float(np.mean(deltas))
    p = _paired_p(a, b)
    psd = _paired_sign_delta(deltas)
    cd = textbook_cliff_delta(a, b)
    sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    dz = float(diff / sd) if sd > 0 else (np.inf if diff > 0 else 0.0)
    eff = psd if effect_metric == "paired_sign" else cd
    cond_diff = diff > 0.0
    cond_p = p < alpha
    cond_n = len(a) >= min_seeds
    cond_eff = abs(eff) >= min_effect
    return GateEval(
        name_a=name_a, name_b=name_b,
        mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
        diff=diff, win_rate=float(np.mean(a > b)),
        wilcoxon_p=p, paired_sign_delta=psd, cliff_delta_textbook=cd,
        n_seeds=len(a), cohen_dz=dz,
        cond_diff_pos=cond_diff, cond_p=cond_p, cond_n=cond_n, cond_effect=cond_eff,
        passes=bool(cond_diff and cond_p and cond_n and cond_eff),
    )


# ===========================================================================
# 既知真陽性 corridor の薄い wrapper (exp_knob_sweep をそのまま再利用)
# ===========================================================================


def corridor_method_scores(
    d: float,
    *,
    n_seeds: int,
    n_evals: int,
    honest_n_trials: int = 30,
    sigma: float = 0.10,
    base_seed: int = 20260530,
) -> dict[str, np.ndarray]:
    """dip depth d の corridor で MAP-E/RR/panmictic/random の honest スコア配列を返す.

    exp_knob_sweep.run_methods_crn を read-only で呼ぶだけ (corridor 構造・CRN seed 設計を継承)。
    返り値 keys = ('map_elites','rr_hillclimb','panmictic_ga','random')。
    """
    eval_once = make_corridor_eval(d)
    bounds = (np.zeros(CORRIDOR_D), np.ones(CORRIDOR_D))
    behavior_bounds = (np.zeros(1), np.ones(1))
    return run_methods_crn(
        eval_once, behavior_mean, dim=CORRIDOR_D, bounds=bounds,
        behavior_bounds=behavior_bounds, grid_shape=(24,),
        n_evals=n_evals, n_seeds=n_seeds, honest_n_trials=honest_n_trials,
        sigma=sigma, base_seed=base_seed,
    )


# ===========================================================================
# 実 negative の per-seed paired delta を既存 results JSON から復元
# ===========================================================================


def load_real_negative_deltas() -> dict[str, dict]:
    """既存 results JSON から実 per-seed paired delta を復元 (仮定値でなく実分布).

    返り値: case 名 -> {a, b, delta, diff, n, source} (numpy 配列入り)。
    """
    out: dict[str, dict] = {}

    ea_path = EA_DIR / "exp_ea3_results.json"
    if ea_path.exists():
        ea = json.loads(ea_path.read_text(encoding="utf-8"))
        sc = ea["scores"]
        me = np.array(sc["map_elites"]["test_per_seed"], dtype=np.float64)
        rnd = np.array(sc["random"]["test_per_seed"], dtype=np.float64)
        pan = np.array(sc["panmictic_ga"]["test_per_seed"], dtype=np.float64)
        rs = np.array(sc["map_elites_randselect"]["test_per_seed"], dtype=np.float64)
        out["C-gen4b_MAPE_vs_random"] = {
            "a": me, "b": rnd, "delta": me - rnd, "diff": float((me - rnd).mean()),
            "n": len(me), "source": str(ea_path), "note": "diff>0 符号一貫 underpowered 候補",
        }
        out["C-gen4a_MAPE_vs_panmictic"] = {
            "a": me, "b": pan, "delta": me - pan, "diff": float((me - pan).mean()),
            "n": len(me), "source": str(ea_path), "note": "diff<0 = 真に効果無し対照",
        }
        out["C-gen3_MAPE_vs_randselect"] = {
            "a": me, "b": rs, "delta": me - rs, "diff": float((me - rs).mean()),
            "n": len(me), "source": str(ea_path), "note": "PASS 対照 (negativeではない)",
        }

    cc_path = REPO_ROOT / "research" / "step_c_memory_tasks" / "exp_c2c3_results.json"
    if cc_path.exists():
        cc = json.loads(cc_path.read_text(encoding="utf-8"))
        for task in ("flip_flop", "delayed_parity"):
            if task not in cc:
                continue
            ps = cc[task]["per_seed_scores"]
            me = np.array(ps["map_elites"], dtype=np.float64)
            for bname in ("random", "rr_hillclimb", "panmictic_ga"):
                b = np.array(ps[bname], dtype=np.float64)
                out[f"{task}_MAPE_vs_{bname}"] = {
                    "a": me, "b": b, "delta": me - b, "diff": float((me - b).mean()),
                    "n": len(me), "source": str(cc_path),
                }
    return out


# ===========================================================================
# G4 破綻ゲート: src/ 不変監査 + 書込先 assert
# ===========================================================================


def _src_tree_fingerprint() -> dict[str, tuple[int, float]]:
    """src/llcore 配下の全 .py の (size, mtime) を記録 (G4 用)."""
    fp: dict[str, tuple[int, float]] = {}
    for f in sorted((SRC_DIR / "llcore").rglob("*.py")):
        st = f.stat()
        fp[str(f.relative_to(REPO_ROOT))] = (st.st_size, st.st_mtime)
    return fp


def assert_research_write_path(path: Path) -> None:
    """書込先が research/statistical_power_audit/ 配下であることを assert (G4)."""
    rp = Path(path).resolve()
    if AUDIT_DIR not in rp.parents and rp != AUDIT_DIR:
        raise AssertionError(f"G4 違反: 書込先が audit dir 外 {rp}")


@dataclass
class RunGuard:
    """各スクリプト start/finish で wall-clock + src 不変 + 書込先を監査 (G1/G4)."""

    name: str
    t0: float
    src_fp_start: dict

    @classmethod
    def start(cls, name: str) -> "RunGuard":
        return cls(name=name, t0=time.time(), src_fp_start=_src_tree_fingerprint())

    def finish(self) -> dict:
        elapsed = time.time() - self.t0
        fp_end = _src_tree_fingerprint()
        src_unchanged = fp_end == self.src_fp_start
        changed = [k for k in fp_end if fp_end.get(k) != self.src_fp_start.get(k)]
        return {
            "script": self.name,
            "wall_clock_s": round(elapsed, 3),
            "src_unchanged": src_unchanged,
            "src_changed_files": changed,
            "exit_ok": True,
        }


def dump_json(path: Path, payload: dict) -> Path:
    """research 配下 assert つき JSON dump (UTF-8)."""
    path = Path(path)
    assert_research_write_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")
    return path


def _json_default(o: object):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
