# SPDX-License-Identifier: Apache-2.0
"""Device helpers shared by the training/eval glue and the experiment entry points.

The constant-state models already build their state on the input's device
(``init_state(..., device=)``), so a model that lives on CUDA and is fed CUDA inputs runs
end-to-end on the GPU. The two things that still need a single, shared implementation are:

- :func:`resolve_device` — turn a ``--device`` CLI string (``"auto"``/``"cpu"``/``"cuda"`` …)
  into a concrete :class:`torch.device`. ``"auto"`` is the backward-compatible default: it is
  ``cpu`` on the current laptop and ``cuda`` on the RTX 5090 box, so the same command runs on
  both without edits (migration plan §6).
- :func:`model_device` — the device a model's parameters live on, so the trainers/eval can move
  each freshly-sampled batch to the model without threading a ``device`` argument everywhere.

Keeping the RNG / index sampling on CPU and moving only the gathered batch means a run on CPU is
byte-identical to before (``.to("cpu")`` is a no-op), while a run on CUDA stays entirely on the GPU.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import torch


class HasParameters(Protocol):
    """Structural type for anything exposing ``parameters()`` (every ``nn.Module``)."""

    def parameters(self) -> Iterator[torch.Tensor]: ...


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device-spec string to a concrete :class:`torch.device`.

    ``"auto"`` (the default) selects CUDA when available and CPU otherwise — so a command
    written on the CPU laptop runs unchanged on the GPU box. Any other value is passed through
    to :class:`torch.device` (``"cpu"``, ``"cuda"``, ``"cuda:1"`` …). Case- and
    whitespace-insensitive.
    """
    s = spec.strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def model_device(model: HasParameters) -> torch.device:
    """Return the device of ``model``'s first parameter (``cpu`` if it has no parameters)."""
    for p in model.parameters():
        return p.device
    return torch.device("cpu")
