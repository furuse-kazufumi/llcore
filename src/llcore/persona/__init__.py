# SPDX-License-Identifier: Apache-2.0
"""Persona-indexed specialist priors (PoC 2b).

8 persona priors over kernel space (decay, mix, gate_str) — each persona biases
sampling toward a distinct region. 集団 = 8 persona × 4 individuals = 32.
"""
from llcore.persona.priors import (
    NUM_PERSONAS,
    PERSONA_LABELS,
    PERSONA_PRIORS,
    PersonaPrior,
    persona_sample_gene,
)

__all__ = [
    "NUM_PERSONAS",
    "PERSONA_LABELS",
    "PERSONA_PRIORS",
    "PersonaPrior",
    "persona_sample_gene",
]
