# SPDX-License-Identifier: Apache-2.0
"""More evolution DIRECTIONS — new Objectives for the Verified-Evolution skeleton.

Demonstrates the `Objective` extension point: adding a direction of evolution = adding a small
deterministic-fitness class. These plug into the UNCHANGED `evolvable_core.evolve()` with any
`VerifierBackend`, exactly like the rotation/benign objectives.

All deterministic (no per-eval RNG, per the Step-D noise-free lesson), all on the n=2 coupled
substrate, all R²-scored (project-standard, headroom). Each rewards a different dynamical
regime, so they exercise different certified regions:

- :class:`SetpointObjective`  — input-driven regulation to a target setpoint (control).
- :class:`DualRateDecayObjective` — autonomous decay with DIFFERENT per-channel rates (favours
  diagonal-ish dynamics → the inf-norm region).
- :class:`DelayedEchoObjective` — echo an input impulse after a delay (memory / leaky hold).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_components import CoupledGene, _r2, step


def _driven_response(gene: CoupledGene, s0: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Trajectory s[0..T] under an input SEQUENCE xs (shape (T, 2))."""
    s = np.asarray(s0, dtype=np.float64).copy()
    traj = [s.copy()]
    for k in range(xs.shape[0]):
        s = step(gene, s, xs[k])
        traj.append(s.copy())
    return np.stack(traj)


@dataclass(frozen=True)
class SetpointObjective:
    """Regulation: under a constant input, the state should rise to and hold a target setpoint.
    Target rises smoothly to the setpoint (so R² is well-posed)."""

    name: str = "setpoint"
    T: int = 30
    input_c: tuple = (0.6, -0.6)
    setpoint: tuple = (0.4, -0.3)
    rate: float = 0.8

    def _target(self) -> np.ndarray:
        ks = np.arange(self.T + 1)
        approach = (1.0 - self.rate ** ks)[:, None]   # 0 -> 1
        return approach * np.array(self.setpoint)[None, :]

    def fitness(self, gene: CoupledGene) -> float:
        xs = np.tile(np.array(self.input_c), (self.T, 1))
        traj = _driven_response(gene, np.zeros(2), xs)
        return _r2(traj, self._target())


@dataclass(frozen=True)
class DualRateDecayObjective:
    """Autonomous decay with DIFFERENT per-channel rates — rewards near-diagonal contracting
    dynamics (the inf-norm region), a contrast to the coupled/rotational objectives."""

    name: str = "dual_rate_decay"
    T: int = 30
    s0: tuple = (0.6, -0.5)
    rate0: float = 0.7
    rate1: float = 0.9

    def _target(self) -> np.ndarray:
        ks = np.arange(self.T + 1)
        return np.stack([self.s0[0] * self.rate0 ** ks, self.s0[1] * self.rate1 ** ks], axis=1)

    def fitness(self, gene: CoupledGene) -> float:
        traj = _driven_response(gene, np.array(self.s0), np.zeros((self.T, 2)))
        return _r2(traj, self._target())


@dataclass(frozen=True)
class DelayedEchoObjective:
    """Memory: an input impulse on channel 0 at t=0 should re-appear (echo) on channel 1 around
    a delay d, as a decaying bump — rewards leaky hold / delayed coupling dynamics."""

    name: str = "delayed_echo"
    T: int = 30
    delay: int = 6
    width: float = 2.5
    amp: float = 0.5

    def _inputs(self) -> np.ndarray:
        xs = np.zeros((self.T, 2))
        xs[0, 0] = 1.0
        return xs

    def _target(self) -> np.ndarray:
        ks = np.arange(self.T + 1)
        bump = self.amp * np.exp(-((ks - self.delay) ** 2) / (2.0 * self.width ** 2))
        return np.stack([np.zeros(self.T + 1), bump], axis=1)

    def fitness(self, gene: CoupledGene) -> float:
        traj = _driven_response(gene, np.zeros(2), self._inputs())
        return _r2(traj, self._target())


ALL_EXTRA = (SetpointObjective, DualRateDecayObjective, DelayedEchoObjective)


if __name__ == "__main__":
    from coupled_components import CoupledGeneCodec, make_verifier
    from evolvable_core import EvolveConfig, evolve
    codec = CoupledGeneCodec()
    cfg = EvolveConfig(pop_size=30, n_generations=40, resample_cap=20)
    for cls in ALL_EXTRA:
        obj = cls()
        r = evolve(codec, obj, make_verifier("sdp"), cfg, rng=np.random.default_rng(5))
        print(f"{obj.name:16s}: verified-evolve gen0={r.best_fitness_curve[0]:+.3f} -> "
              f"final={r.best_fitness_curve[-1]:+.3f}  (admitted divergent: see soundness)")
