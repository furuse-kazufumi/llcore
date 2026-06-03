# VERDICT — Stage 3b: verifier backend plugin promoted to src/ (2026-06-03)

> Roadmap item #3: promote the arc's conclusion ("the right contraction verifier is
> SDP/Lyapunov, pluggable") from `research/` into the **production `src/`** as a backend
> plugin — additively, with **zero regression** to the existing suite.

## What landed (first src/ change of the arc)

`src/llcore/verifier/backends.py` (additive new module; existing modules untouched):

- `VerifierBackend` protocol — `certifies(gene) -> bool` + `name` + `available`.
- `ClosedFormScalarBackend` — the scalar RWKV-gene closed-form contraction (default; reuses
  `verify_lipschitz_contraction`; no new deps).
- `InfNormBackend` / `TwoNormBackend` — coupled-gene induced-norm contraction over the sound
  achievable-t box (numpy only).
- `SdpLyapunovBackend` — coupled-gene common quadratic Lyapunov LMI (**cvxpy, optional
  `[sdp]` extra**; `available=False` ⇒ fail-closed). 2-norm fast-path works even without cvxpy.
- `get_verifier_backend(name)` registry + `available_backends()` + `cvxpy_available()`.
  Exported from `llcore.verifier`.

Genes are **duck-typed** (a coupled gene exposes `.decay (n,)` / `.W (n,n)`; the backend
imports no research module), so it works for any coupled substrate today and a future src
coupled gene.

## Verification

- **255 passed** (the prior 248 + 7 new `tests/unit/test_backends.py`). **No regression** —
  the change is purely additive (new module + new exports). `src/` behaviour is otherwise
  unchanged (semver-safe: new API only).
- The **arc payoff shows up in the production API**: `inf_norm.certifies(ROT) is False`
  while `two_norm`/`sdp_lyapunov` admit the same rotational contraction — exactly the
  "∞-norm over-rejects, better verifier recovers" result, now a first-class src feature.
- Fail-closed verified: cvxpy-absent ⇒ `SdpLyapunovBackend.available is False` and a
  solver-needing gene is rejected (not raised); malformed genes ⇒ `False`.

## Bottom line

✅ The SDP/Lyapunov verifier is now a **production, pluggable backend** in `llcore.verifier`
(Stage 3b), cvxpy-optional and fail-closed, with the arc's headline distinction (inf vs
2-norm/SDP) covered by tests — and **0 regression** on the 248-test suite. The evolution loop
can now select its soundness gate by name from src.

Files: `src/llcore/verifier/backends.py`, `src/llcore/verifier/__init__.py` (+exports),
`tests/unit/test_backends.py`. No push. Next: #4 — new `Objective`s (evolution directions).
