# Verified Evolution skeleton (CPU) — llcore

> **進化の実現が最優先。一度実現すれば、これを骨組みとして進化方向の追加・機能拡張をするだけ。**
> (User directive, 2026-06-03.) This directory **realizes** llcore's founding thesis on CPU:
> *evolve the core dynamics of an AI substrate, with a verifier that never lets them break.*
> Once working, extending it = swap one of three plugins. Nothing else changes.

## What it is

A minimal, **verified** evolutionary loop over a neural-dynamics core. Each generation
mutates the dynamics gene; every child must pass a **fail-closed contraction verifier**
before it is admitted, so evolution can search aggressively yet the dynamical system is
provably never driven into divergence. The verifier backend is **SDP/Lyapunov** — the tool
the llcore CPU-Verification arc (Track A→D, `../CPU_VERIFICATION_RESEARCH_2026-06-02_VERDICT.md`)
proved correct (Z3/SMT is decorative for these invariants; SDP genuinely beats every fixed
induced norm).

## The skeleton (3 extension points)

`evolvable_core.py` defines three pluggable `Protocol`s and one GA loop:

```
GeneCodec        genotype (flat vector) <-> domain gene + random/clip/crossover/mutate
Objective        deterministic fitness (higher=better)  <-- the DIRECTION of evolution
VerifierBackend  certifies(gene) -> bool  (fail-closed admission gate)  <-- the SAFETY
evolve(codec, objective, verifier, cfg, rng) -> EvolveResult
```

`coupled_components.py` supplies the concrete plugins for the n=2 coupled RWKV-style map:
`CoupledGeneCodec`, three objectives (`RotationObjective` / `BenignDecayObjective` /
`NonNormalObjective`), and four verifier backends (`none` / `inf_norm` / `two_norm` / `sdp`),
the last three wrapping the Track C/D certifiers.

## How to extend (this is the whole point)

**Add an evolution direction** — implement `Objective`:

```python
@dataclass(frozen=True)
class MyTask:
    name: str = "my_task"
    def fitness(self, gene) -> float:        # deterministic, higher = better
        traj = _free_response(gene, np.array([0.4, 0.0]), 30)
        return _r2(traj, my_target)
evolve(CoupledGeneCodec(), MyTask(), make_verifier("sdp"))
```

**Extend the substrate** — implement `GeneCodec` (e.g. n=4 coupled, multi-kernel union,
a learning-rule gene). The GA and verifier interface are unchanged; only `dim`, `to_gene`,
and the operators change.

**Swap the verifier** — implement `VerifierBackend` (e.g. a JSR lower-bound gate for the
D4 residual, or a non-quadratic Lyapunov backend). `evolve` is agnostic to which sound
certificate you use — that is the verifier-backend plugin the arc called for.

## Run it

```bash
py -3.11 demo_evolve.py                 # realization demo: evolution climbs + 0 divergent admitted
py -3.11 -m pytest test_skeleton.py -q  # 12 TDD tests (determinism, soundness, improvement)
py -3.11 exp_runner.py exp1             # landscape attribution (mechanism evidence)
py -3.11 exp_runner.py exp2 --seeds 20  # gated evolution: G0-G5 verdicts
py -3.11 redteam.py all                 # adversarial lenses A-D
```

## Established findings (see `VERDICT.md`)

- Evolution **works** and is **verified**: best fitness climbs; the SDP gate admits **0**
  divergent genes while ungated evolution drifts 2–21 % into non-contraction (gate is
  load-bearing, G1/G2).
- A **better verifier pays off**: on the rotation task the conservative ∞-norm gate cripples
  evolution (it over-rejects the rotation region); the SDP/2-norm gate recovers it (G4),
  while a benign task shows no difference (G5 null — no generic-SDP artifact).

## Discipline

Additive research only — **src/ untouched**. Reuses Track C/D certifier modules. Seeds fixed
(paired/common-random-numbers). Pre-registration (`PREREGISTRATION.md`) before running;
adversarial red-team after; honest disclosure on every "too good" result.
