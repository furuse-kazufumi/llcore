# SPDX-License-Identifier: Apache-2.0
"""llcore.memory — the memory-efficiency toolkit (facade over verified primitives).

llcore's value is not the (tiny) model but the **memory-efficiency infrastructure
layer**: a small set of CPU-verified primitives for shrinking a char-LM's resident
footprint and working set *without silently losing capability*. This module is the
one import that re-exports those primitives and adds a single one-call report.

What it re-exports (each is the *same object* as its origin — no re-implementation):

- int8 weight-only quantization & streaming/mmap inference
  (:func:`quantize_per_channel_int8`, :class:`Int8Linear`, :func:`convert_linears_to_int8`,
  :func:`save_int8_checkpoint`, :func:`load_int8_model`, :func:`int8_footprint_bytes`)
  — ``docs/MEMORY_EFFICIENCY_FINDINGS.md`` (b)/(c)/(a').
- capability gates (:func:`held_out_top1_report`, :func:`passes_capability_gate`)
  — the *fail-closed* operational contribution: a footprint win is only accepted if the
  model retains the bulk of its exact next-token accuracy (a PPL-only gate can pass a
  badly-degraded low-bit model; see ``docs/MEMORY_EFFICIENCY_FINDINGS.md`` (b')).
- constant-state recurrence vs GPT KV growth (:class:`RecurrentLM`, :func:`gpt_kv_bytes`,
  :func:`constant_state_bytes`) — the structural memory axis (``... FINDINGS`` (0')/(a)).

New here: :class:`MemoryReport` + :func:`measure_memory`, which answer
"how much smaller, and at what capability cost?" in one call.

Honest scope: this is **packaging**, not a new algorithm — the quantization / mmap /
recurrence primitives are re-derivations of established techniques (llama.cpp / GGUF;
see ``docs/POSITIONING_VS_LLAMACPP.md``). The footprint numbers are *resident weight
bytes* (storage), not process RSS; retention is measured under simulated int8 (a
footprint/working-set optimization, not an int8-GEMM speedup). The design property worth
keeping: each primitive *scales up* on better hardware (int8 → real int8 GEMM on GPU,
mmap → shared page-cache residency on big RAM, constant state → long context without the
attention quadratic), so it is a foundation, not a CPU-only stopgap.

CLI: ``py -3.11 -m llcore.memory report <checkpoint> [--corpus-file F] [--json OUT]``.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from llcore.lm.compare import constant_state_bytes, gpt_kv_bytes
from llcore.lm.corpus import read_utf8_corpus_file
from llcore.lm.data import fetch_aozora_text, fetch_tinyshakespeare, train_val_split
from llcore.lm.eval import held_out_top1_report, passes_capability_gate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.quant import (
    Int8Linear,
    convert_linears_to_int8,
    int8_footprint_bytes,
    load_int8_model,
    quantize_per_channel_int8,
    save_int8_checkpoint,
)
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.tokenizer import CharTokenizer

__all__ = [
    # facade re-exports (verified primitives)
    "Int8Linear",
    "RecurrentConfig",
    "RecurrentLM",
    "constant_state_bytes",
    "convert_linears_to_int8",
    "gpt_kv_bytes",
    "held_out_top1_report",
    "int8_footprint_bytes",
    "load_int8_model",
    "passes_capability_gate",
    "quantize_per_channel_int8",
    "save_int8_checkpoint",
    # new in this module
    "MemoryReport",
    "measure_memory",
    "main",
]

DEFAULT_MIN_RETENTION = 0.97


@dataclass(frozen=True)
class MemoryReport:
    """A one-call memory report: footprint win + (optional) capability cost.

    ``fp32_bytes`` / ``int8_bytes`` are *resident weight bytes* (int8 body + per-row fp32
    scales for ``nn.Linear`` weights; fp32 for embeddings / LayerNorm / biases), as
    accounted by :func:`int8_footprint_bytes`. When held-out tokens are supplied,
    ``fp32_top1`` / ``int8_top1`` are teacher-forced next-token top-1 accuracies, and
    ``capability_gate_pass`` is the fail-closed verdict from :func:`passes_capability_gate`.
    The capability fields stay ``None`` when no evaluation data is given — absence is
    reported, never fabricated.
    """

    fp32_bytes: int
    int8_bytes: int
    compression_ratio: float
    percent_smaller: float
    min_retention: float
    fp32_top1: float | None = None
    int8_top1: float | None = None
    retention: float | None = None
    capability_gate_pass: bool | None = None
    n_eval_tokens: int | None = None
    kv_bytes_by_context: dict[int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        """Plain-dict view (JSON-serializable) for reports / artifacts.

        Note: ``kv_bytes_by_context`` keeps ``int`` keys here; ``json.dumps`` will
        render them as strings (JSON object keys are always strings).
        """
        return asdict(self)


def measure_memory(
    model: CharGPT,
    *,
    val_ids: torch.Tensor | None = None,
    block_size: int | None = None,
    batch_size: int = 32,
    min_retention: float = DEFAULT_MIN_RETENTION,
    context_lens: Sequence[int] | None = None,
) -> MemoryReport:
    """Measure ``model``'s memory profile across up to three axes.

    1. **Footprint** (always, read-only): int8 vs fp32 resident weight bytes via
       :func:`int8_footprint_bytes`.
    2. **Capability cost** (when ``val_ids`` given): the model is quantized on a *deep
       copy* (the caller's fp32 model is never mutated) and scored against itself —
       ``retention = int8_top1 / fp32_top1`` and the fail-closed
       :func:`passes_capability_gate` verdict at ``min_retention``. ``block_size``
       defaults to ``model.config.block_size``.
    3. **Structural growth** (when ``context_lens`` given): the GPT KV-cache bytes at
       each context length via :func:`gpt_kv_bytes` — this grows *linearly* with
       context, which is exactly what a constant-state recurrent model (re-exported
       :func:`constant_state_bytes`) avoids (its state is length-independent).
    """
    fp = int8_footprint_bytes(model)
    fp32_bytes = fp["fp32_bytes"]
    int8_bytes = fp["int8_bytes"]
    ratio = int8_bytes / fp32_bytes

    fp32_top1: float | None = None
    int8_top1: float | None = None
    retention: float | None = None
    gate: bool | None = None
    n_eval: int | None = None

    if val_ids is not None:
        bs = block_size if block_size is not None else model.config.block_size
        fp32_top1 = held_out_top1_report(model, val_ids, bs, batch_size)["top1_acc"]
        # Quantize a copy so the caller's fp32 model is untouched.
        int8_model = copy.deepcopy(model)
        convert_linears_to_int8(int8_model)
        int8_report = held_out_top1_report(int8_model, val_ids, bs, batch_size)
        int8_top1 = int8_report["top1_acc"]
        n_eval = int(int8_report["n_tokens"])
        retention = (int8_top1 / fp32_top1) if fp32_top1 > 0 else None
        gate = passes_capability_gate(int8_top1, fp32_top1, min_retention)

    kv_by_ctx: dict[int, int] | None = None
    if context_lens is not None:
        kv_by_ctx = {int(t): gpt_kv_bytes(model, int(t)) for t in context_lens}

    return MemoryReport(
        fp32_bytes=fp32_bytes,
        int8_bytes=int8_bytes,
        compression_ratio=ratio,
        percent_smaller=(1.0 - ratio) * 100.0,
        min_retention=min_retention,
        fp32_top1=fp32_top1,
        int8_top1=int8_top1,
        retention=retention,
        capability_gate_pass=gate,
        n_eval_tokens=n_eval,
        kv_bytes_by_context=kv_by_ctx,
    )


# --- CLI: py -3.11 -m llcore.memory report <checkpoint> ----------------------


def _load_fp32_checkpoint(path: Path) -> tuple[CharGPT, CharTokenizer]:
    """Load a dense fp32 char-LM checkpoint (config + state dict + tokenizer itos)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CharGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])


def _load_corpus_text(corpus: str | None, corpus_file: str | None) -> str:
    if corpus_file:
        return read_utf8_corpus_file(corpus_file)
    if corpus == "shakespeare":
        return fetch_tinyshakespeare()
    if corpus == "aozora":
        return fetch_aozora_text()
    raise ValueError("no corpus selected (pass --corpus or --corpus-file)")


def _print_report(report: MemoryReport, checkpoint: Path) -> None:
    print(f"checkpoint     : {checkpoint}")
    print(f"fp32 weights   : {report.fp32_bytes / 1e6:.2f} MB")
    print(
        f"int8 resident  : {report.int8_bytes / 1e6:.2f} MB  "
        f"(ratio {report.compression_ratio:.3f}, {report.percent_smaller:.1f}% smaller)"
    )
    if report.fp32_top1 is None or report.int8_top1 is None:
        print("capability     : (no corpus given; pass --corpus/--corpus-file for retention)")
        return
    if report.retention is not None:
        print(
            f"top-1 retention: {report.retention * 100:.1f}%  "
            f"(fp32 {report.fp32_top1:.4f} -> int8 {report.int8_top1:.4f}, "
            f"n={report.n_eval_tokens})"
        )
    else:
        print(
            f"top-1          : fp32 {report.fp32_top1:.4f} -> int8 {report.int8_top1:.4f} "
            f"(retention undefined; fp32 top-1 is 0)"
        )
    verdict = "PASS" if report.capability_gate_pass else "FAIL"
    print(f"capability gate: {verdict} (>= {report.min_retention * 100:.0f}% retention)")


def _print_kv_growth(report: MemoryReport) -> None:
    if not report.kv_bytes_by_context:
        return
    items = sorted(report.kv_bytes_by_context.items())
    lo_t, lo_b = items[0]
    hi_t, hi_b = items[-1]
    growth = hi_b / lo_b if lo_b else float("nan")
    print(
        f"KV cache       : T={lo_t} {lo_b / 1e6:.2f} MB -> T={hi_t} {hi_b / 1e6:.2f} MB "
        f"(x{growth:.1f}; grows linearly with context)"
    )
    print("                 (a constant-state recurrent model would stay flat here)")


def _should_promote(report: MemoryReport, *, force: bool) -> tuple[bool, str]:
    """Fail-closed decision: may we emit the int8 checkpoint for this report?

    The capability gate is treated as a *promotion gate* — this is the operational
    contribution llcore adds on top of the (re-derived) quantization primitives. We
    only promote a footprint win that is backed by measured capability retention:

    - ``--force`` overrides everything (operator override).
    - No capability evidence (no corpus given → ``capability_gate_pass is None``) →
      refuse: a footprint win with no retention measurement is unverified.
    - Gate failed (retention below ``min_retention``) → refuse.
    - Gate passed → promote.
    """
    if force:
        return True, "promotion forced (--force): writing despite the gate"
    if report.capability_gate_pass is None:
        return (
            False,
            "no capability evidence - pass --corpus/--corpus-file to measure retention "
            "before promoting (or --force to override)",
        )
    if report.capability_gate_pass is False:
        return (
            False,
            f"capability gate FAILED (retention below {report.min_retention:.2f}) - "
            "refusing to promote (or --force to override)",
        )
    return True, "capability gate passed — promoting"


def cmd_report(args: argparse.Namespace) -> int:
    model, tok = _load_fp32_checkpoint(Path(args.checkpoint))
    val_ids: torch.Tensor | None = None
    if args.corpus or args.corpus_file:
        text = _load_corpus_text(args.corpus, args.corpus_file)
        ids = torch.tensor(tok.encode_safe(text), dtype=torch.long)
        _, val_ids = train_val_split(ids, val_frac=args.val_frac)
    context_lens: list[int] | None = None
    if args.context_lens:
        context_lens = [int(x) for x in args.context_lens.split(",") if x.strip()]
    report = measure_memory(
        model, val_ids=val_ids, min_retention=args.min_retention, context_lens=context_lens
    )
    _print_report(report, Path(args.checkpoint))
    _print_kv_growth(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    if args.save_int8:
        ok, reason = _should_promote(report, force=args.force)
        if not ok:
            print(f"[save-int8] refused: {reason}", file=sys.stderr)
            return 2
        save_int8_checkpoint(model, Path(args.save_int8), tok.itos)
        print(f"[save-int8] {reason} -> wrote {args.save_int8}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="llcore.memory", description="memory-efficiency toolkit for the char-LM"
    )
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser(
        "report", help="footprint (+ optional capability retention) of an fp32 checkpoint"
    )
    r.add_argument("checkpoint")
    r.add_argument("--corpus", choices=["shakespeare", "aozora"], default=None)
    r.add_argument("--corpus-file", default=None, help="local UTF-8 corpus for retention")
    r.add_argument("--val-frac", type=float, default=0.1)
    r.add_argument("--min-retention", type=float, default=DEFAULT_MIN_RETENTION)
    r.add_argument(
        "--context-lens",
        default=None,
        help="comma-separated context lengths (e.g. 256,512,1024) to report GPT "
        "KV-cache growth (linear) vs constant-state recurrent (flat)",
    )
    r.add_argument("--json", default=None, help="also write the report as JSON to this path")
    r.add_argument(
        "--save-int8",
        default=None,
        help="emit an int8 checkpoint here, but only if the capability gate passes "
        "(fail-closed promotion gate)",
    )
    r.add_argument(
        "--force",
        action="store_true",
        help="emit the int8 checkpoint even if the gate fails or no corpus was given",
    )
    r.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - best-effort console setup
        pass
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
