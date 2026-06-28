# SPDX-License-Identifier: Apache-2.0
"""Tests for device wiring: ``llcore.lm.device`` helpers + CPU byte-identical guards.

The GPU migration (RTX 5090) needs the training/eval glue to run on CUDA. The models
already build state on the input's device (``init_state(..., device=)``); the gap is the
training/eval glue, which never moved batches off CPU. The wiring must keep the model and
all tensors it sees on ONE device while leaving the RNG/index sampling on CPU, so the random
stream — and therefore the CPU numbers — are byte-identical to before. These tests pin:

  1. ``resolve_device`` / ``model_device`` helpers (new code; red-first).
  2. The wired trainers/eval stay deterministic and ``.to("cpu")``-invariant (exercises the
     new ``.to(device)`` path with ``device == cpu``, a no-op that must not perturb results).

Cross-device (CPU vs CUDA) equality is verified Day-1 on the GPU box (migration plan §5);
this CPU-only suite proves the wiring did not change existing behavior.
"""
from __future__ import annotations

import torch


# --- resolve_device: spec string -> torch.device ----------------------------------------


def test_resolve_device_cpu() -> None:
    from llcore.lm.device import resolve_device

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_matches_availability() -> None:
    from llcore.lm.device import resolve_device

    dev = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected


def test_resolve_device_default_is_auto() -> None:
    from llcore.lm.device import resolve_device

    assert resolve_device() == resolve_device("auto")


def test_resolve_device_is_case_and_space_insensitive() -> None:
    from llcore.lm.device import resolve_device

    assert resolve_device("  CPU ") == torch.device("cpu")


# --- model_device: the device a model's parameters live on ------------------------------


def test_model_device_cpu_model() -> None:
    from llcore.lm.device import model_device
    from llcore.lm.recurrent import RecurrentConfig, RecurrentLM

    model = RecurrentLM(RecurrentConfig(vocab_size=8, block_size=16, n_layer=1, n_embd=16, state_size=16))
    assert model_device(model) == torch.device("cpu")
