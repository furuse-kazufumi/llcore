# SPDX-License-Identifier: Apache-2.0
"""P1 capability 比較集計 — smoke baseline vs 実 config p1 の held-out PPL を並べる。

`docs/LM_P0_PLAN.md` §7 の P1 ablation を、実 `--config p1`(256ctx/384d/6L) まで
拡張するための集計ツール。各 run dir の `verdict.json` と、隣接する `<dir>_run.log`
(trainer の `[model]` / `[train]` 行) を読み、比較表 + val 曲線を出す。

honest disclosure（このツールが明示する留保）:
  - **主指標は `ratio_model_over_unigram`**。cross-run の絶対 `model_ppl` は eval 窓
    サイズ(block_size 64 vs 256)が異なり、unigram baseline も run ごとに同一トークン上で
    再計算されるため、絶対値の直接比較は厳密でない。ratio は run 内で自己整合。
  - verdict の `model_ppl` は **最終モデル**(early-stop なし)で評価。run log の
    `best_val` と `final_val` の乖離(過学習ギャップ)を併記して開示する。
  - 小コーパス(aozora 289K字)では大容量モデルは過学習しうる(既知教訓)。

使い方::

    py -3.11 scripts/p1_compare.py                      # 既定 run セット
    py -3.11 scripts/p1_compare.py out/lm_aozora out/lm_aozora_realp1
    py -3.11 scripts/p1_compare.py --json out/p1_compare.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

# trainer 出力行のパーサ
_MODEL_RE = re.compile(
    r"\[model\]\s+(?P<config>\S+):\s+(?P<params>[\d,]+)\s+params"
    r"(?:.*?L(?P<L>\d+)\s+H(?P<H>\d+)\s+D(?P<D>\d+)\s+ctx(?P<ctx>\d+)\s+dropout(?P<drop>[\d.]+))?"
)
# 旧 trainer の log 形式: ``[model] smoke: ... cfg={'n_layer': 4, ...}``
# flat dict 前提。最初の ``}`` で切るため、ネストした cfg には対応しない。
_CFG_BLOCK_RE = re.compile(r"cfg=(?P<cfg>\{[^}]*\})")
_CFG_FIELD_RES = {
    "L": re.compile(r"'n_layer':\s*(\d+)"),
    "H": re.compile(r"'n_head':\s*(\d+)"),
    "D": re.compile(r"'n_embd':\s*(\d+)"),
    "ctx": re.compile(r"'block_size':\s*(\d+)"),
    "dropout": re.compile(r"'dropout':\s*([\d.]+)"),
}
_TRAIN_RE = re.compile(
    r"\[train\]\s+iter\s+(?P<iter>\d+)\s+train_loss\s+(?P<tr>[\d.]+)\s+val_loss\s+(?P<va>[\d.]+)"
)

# 既定の比較対象（存在するものだけ使う）
_DEFAULT_RUNS = [
    "out/lm_aozora",          # smoke 2000it dropout0 (P0)
    "out/lm_aozora_drop",     # smoke 3500it dropout0.1 (P1 best so far)
    "out/lm_aozora_realp1",   # 実 config p1
    "out/lm_shakespeare",     # smoke (P0)
    "out/lm_shakespeare_realp1",
]


def _parse_log(log_path: Path) -> dict[str, Any]:
    """run log から architecture と val 軌跡を抽出する。"""
    info: dict[str, Any] = {"arch": None, "traj": []}
    if not log_path.exists():
        return info
    text = log_path.read_text(encoding="utf-8", errors="replace")
    model_line = next((line for line in text.splitlines() if re.match(r"\[model\]\s", line)), "")
    m = _MODEL_RE.search(model_line)
    if m:
        g = m.groupdict()
        arch = {
            "config": g["config"],
            "params": int(g["params"].replace(",", "")),
            "L": int(g["L"]) if g["L"] else None,
            "H": int(g["H"]) if g["H"] else None,
            "D": int(g["D"]) if g["D"] else None,
            "ctx": int(g["ctx"]) if g["ctx"] else None,
            "dropout": float(g["drop"]) if g["drop"] else None,
        }
        # 新形式で L/H/D/ctx/dropout が揃わない時だけ、同じ [model] 行の legacy cfg を試す。
        if arch["ctx"] is None:
            cg = _parse_cfg_fields(model_line)
            if cg:
                arch.update(
                    L=int(cg["L"]), H=int(cg["H"]), D=int(cg["D"]),
                    ctx=int(cg["ctx"]), dropout=float(cg["dropout"]),
                )
        info["arch"] = arch
    for tm in _TRAIN_RE.finditer(text):
        info["traj"].append(
            {"iter": int(tm["iter"]), "train": float(tm["tr"]), "val": float(tm["va"])}
        )
    return info


def _parse_cfg_fields(model_line: str) -> dict[str, str] | None:
    """旧 ``cfg={...}`` 形式から architecture field を順不同で抜き出す。"""
    m = _CFG_BLOCK_RE.search(model_line)
    if not m:
        return None
    cfg_text = m.group("cfg")
    parsed: dict[str, str] = {}
    for key, pat in _CFG_FIELD_RES.items():
        fm = pat.search(cfg_text)
        if not fm:
            # all-or-nothing: 1 field でも欠けたら legacy fallback は使わない
            return None
        parsed[key] = fm.group(1)
    return parsed


def _gather(run_dir: Path) -> dict[str, Any] | None:
    verdict_path = run_dir / "verdict.json"
    if not verdict_path.exists():
        return None
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    log = _parse_log(run_dir.parent / f"{run_dir.name}_run.log")
    ckpt_arch = _load_checkpoint_arch(run_dir / "model.pt")
    arch = _merge_arch(log["arch"], ckpt_arch, fallback_config=verdict.get("config"))
    traj = log["traj"]
    best_val = min((t["val"] for t in traj), default=None)
    final_val = traj[-1]["val"] if traj else None
    overfit_gap = (
        round(final_val - best_val, 4) if (best_val is not None and final_val is not None) else None
    )
    return {
        "run": run_dir.name,
        "verdict": verdict,
        "arch": arch,
        "best_val": round(best_val, 4) if best_val is not None else None,
        "final_val": round(final_val, 4) if final_val is not None else None,
        "overfit_gap": overfit_gap,
        "traj": traj,
    }


def _load_checkpoint_arch(model_path: Path) -> dict[str, Any] | None:
    """Read authoritative arch metadata from a trusted local training checkpoint."""
    if not model_path.exists():
        return None
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully for mixed local artifacts
        print(
            f"[warn] {model_path}: checkpoint arch unavailable ({exc.__class__.__name__})",
            file=sys.stderr,
        )
        return None
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        return None
    return {
        "config": cfg.get("model_type"),
        "params": None,
        "L": cfg.get("n_layer"),
        "H": cfg.get("n_head"),
        "D": cfg.get("n_embd"),
        "ctx": cfg.get("block_size"),
        "dropout": cfg.get("dropout"),
    }


def _merge_arch(
    log_arch: dict[str, Any] | None,
    ckpt_arch: dict[str, Any] | None,
    *,
    fallback_config: Any,
) -> dict[str, Any] | None:
    """Prefer checkpoint config for architecture fields while preserving log-only metadata."""
    if log_arch is None:
        if ckpt_arch is None:
            return None
        merged = dict(ckpt_arch)
        if merged.get("config") in (None, "None"):
            merged["config"] = fallback_config
        return merged
    if ckpt_arch is None:
        merged = dict(log_arch)
        if merged.get("config") in (None, "None"):
            merged["config"] = fallback_config
        return merged
    merged = dict(log_arch)
    if merged.get("config") in (None, "None"):
        merged["config"] = ckpt_arch.get("config") or fallback_config
    for key in ("L", "H", "D", "ctx", "dropout"):
        if ckpt_arch.get(key) is not None:
            merged[key] = ckpt_arch[key]
    return merged


def _extra_corpus_count(rec: dict[str, Any]) -> int:
    extra = rec["verdict"].get("extra_corpus_files", [])
    return len(extra) if isinstance(extra, list) else 0


def _fmt_arch(rec: dict[str, Any]) -> str:
    a = rec["arch"]
    v = rec["verdict"]
    if a and a.get("ctx") is not None:
        return f"{a['config']} L{a['L']}H{a['H']}D{a['D']} ctx{a['ctx']} drop{a['dropout']}"
    # log が arch 行を持たない (eval 由来等) 場合は verdict から最低限
    return f"{v.get('config', '?')} (arch n/a)"


def _cell(value: Any) -> str:
    """Render optional table cells consistently."""
    return "?" if value is None else str(value)


def _gate_cell(value: Any) -> str:
    """Render tri-state gate cells as PASS / FAIL / ?."""
    if value is None:
        return "?"
    return "PASS" if value else "FAIL"


def _markdown_table(records: list[dict[str, Any]]) -> str:
    head = (
        "| run | extra corpora | arch | params | iters | eval tok | unigram PPL | model PPL | "
        "ratio | gate | degen | best_val | final_val | overfit gap |"
    )
    sep = "|" + "---|" * 14
    rows = [head, sep]
    for r in records:
        v = r["verdict"]
        params = r["arch"]["params"] if r["arch"] else v.get("n_params")
        params_cell = f"{params:,}" if params is not None else "?"
        rows.append(
            f"| {r['run']} | {_extra_corpus_count(r)} | {_fmt_arch(r)} | {params_cell} | {_cell(v.get('max_iters'))} | "
            f"{_cell(v.get('n_eval_tokens'))} | {_cell(v.get('unigram_ppl'))} | {_cell(v.get('model_ppl'))} | "
            f"{_cell(v.get('ratio_model_over_unigram'))} | "
            f"{_gate_cell(v.get('ppl_gate_pass'))} | {_cell(v.get('degenerate_sample'))} | "
            f"{_cell(r['best_val'])} | {_cell(r['final_val'])} | {_cell(r['overfit_gap'])} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P1 capability comparison aggregator")
    ap.add_argument("runs", nargs="*", default=_DEFAULT_RUNS, help="run dirs (with verdict.json)")
    ap.add_argument("--json", default=None, help="optional path to dump the gathered records")
    args = ap.parse_args(argv)

    records: list[dict[str, Any]] = []
    for run in args.runs:
        rec = _gather(Path(run))
        if rec is not None:
            records.append(rec)
        else:
            print(f"[skip] {run} (no verdict.json yet)")

    if not records:
        print("no runs found")
        return 1

    print("\n" + _markdown_table(records) + "\n")

    # 軌跡 (過学習診断)
    for r in records:
        if not r["traj"]:
            continue
        curve = "  ".join(f"{t['iter']}:{t['val']:.3f}" for t in r["traj"])
        print(f"[{r['run']}] val: {curve}")

    # headline: aozora smoke-best ratio vs 実 p1 ratio
    by_run = {r["run"]: r for r in records}
    base = by_run.get("lm_aozora_drop")
    p1 = by_run.get("lm_aozora_realp1")
    if base and p1:
        br = base["verdict"].get("ratio_model_over_unigram")
        pr = p1["verdict"].get("ratio_model_over_unigram")
        if br is None or pr is None:
            print(
                "\n[headline aozora] ratio_model_over_unigram missing; "
                "skip headline comparison."
            )
        else:
            verdict = "improved" if pr < br else "did NOT improve"
            print(
                f"\n[headline aozora] smoke-best ratio {br} vs 実p1 ratio {pr} -> "
                f"実p1 {verdict} held-out (lower=better). "
                f"extras(base={_extra_corpus_count(base)}, p1={_extra_corpus_count(p1)}). "
                f"caveat: 異なる eval 窓 (ctx) と過学習 gap を上表で要確認。"
            )

    if args.json:
        Path(args.json).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
