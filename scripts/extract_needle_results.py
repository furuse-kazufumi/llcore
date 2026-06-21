# SPDX-License-Identifier: Apache-2.0
"""Extract b2-ready long-context numbers from a ``nas_pareto.json`` offload result.

The GH Actions offload (tag ``needle-run-*``) writes ``proxy_v2`` into ``nas_pareto.json``.
This script reads that block with the *verified* schema and prints the exact values the
article ``docs/articles/drafts/b2-suppress-your-win.md`` needs, so integration is one
deterministic command instead of an ad-hoc one-liner that silently KeyErrors.

Verified schema (``src/llcore/runtime/eval_proxy.py``):
  * ``proxy_v2.context_sweep`` keys are **stringified** lengths (``"2048"``, not ``2048`` —
    ``build_proxy_v2_report`` does ``{str(k): v ...}``).
  * each value is ``{"mean", "ci_lo", "ci_hi", "p_worse", "pos_frac", "p_sign", "n_windows"}``;
    the paired Δnll the article cites is ``"mean"`` (bootstrap mean), CI = ``ci_lo``/``ci_hi``.
  * ``proxy_v2.needle`` is ``{"horizon": int|None, "by_depth": {...}}``; ``horizon`` is the
    shortest length where the genome fails retrieval while the all-softmax base succeeds, or
    ``None`` if it never fails within the swept lengths.

The output is intentionally honest-disclosure shaped: Δnll is reported WITH its CI, and the
needle line distinguishes "never failed (None)" from "failed at N tok".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt_sweep_row(length: str, row: dict[str, Any]) -> str:
    """Format one context-sweep length as ``L: mean (CI lo..hi, n=K)``."""
    mean = float(row["mean"])
    ci_lo = float(row["ci_lo"])
    ci_hi = float(row["ci_hi"])
    n = int(float(row["n_windows"]))
    return f"  {length:>5}: delta_nll {mean:+.4f}  (95% CI {ci_lo:+.4f}..{ci_hi:+.4f}, n={n})"


def _fmt_needle(needle: dict[str, Any] | None) -> str:
    """Format the needle horizon as an honest, article-ready sentence."""
    if needle is None:
        return "needle: (not present in this run)"
    horizon = needle.get("horizon")
    if horizon is None:
        return "needle: horizon=None - never failed retrieval within swept lengths (no broken horizon observed)"
    return f"needle: horizon={int(horizon)} tok - shortest length where retrieval breaks while all-softmax base still succeeds"


def extract(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the ``proxy_v2`` long-context block from a loaded ``nas_pareto.json``.

    Returns a plain dict with ``context_sweep`` (str-keyed) and ``needle`` (or ``None``).
    Raises ``KeyError`` if ``proxy_v2`` is absent (offload did not run the rigorous tier).
    """
    proxy = report["proxy_v2"]
    if not isinstance(proxy, dict):  # defensive: schema drift
        raise TypeError(f"proxy_v2 is not a dict: {type(proxy)!r}")
    sweep = proxy.get("context_sweep", {})
    needle = proxy.get("needle")
    return {"context_sweep": sweep, "needle": needle}


def render(block: dict[str, Any]) -> str:
    """Render the extracted block as human-readable, b2-integration-ready text."""
    sweep: dict[str, Any] = block["context_sweep"]
    needle: dict[str, Any] | None = block["needle"]
    lines: list[str] = ["context_sweep (delta_nll = 'mean'):"]
    # sort by integer length so 2048/4096 land after 1024
    for length in sorted(sweep, key=lambda k: int(k)):
        lines.append(_fmt_sweep_row(length, sweep[length]))
    lines.append(_fmt_needle(needle))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=Path("out/needle_offload/nas_pareto.json"),
        help="path to nas_pareto.json (default: out/needle_offload/nas_pareto.json)",
    )
    args = ap.parse_args()
    report = json.loads(args.json_path.read_text(encoding="utf-8"))
    block = extract(report)
    print(render(block))


if __name__ == "__main__":
    main()
