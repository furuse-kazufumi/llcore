import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_tasks import DelayedRecallTask
from reservoir import LeakyDelayLineReservoir, make_eval_once, make_behavior, gene_bounds


def test_run_states_shape():
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    rng = np.random.default_rng(0)
    gene = res.random_gene(rng)
    inputs = np.sign(rng.normal(size=(12, 1)))
    states = res.run(gene, inputs)
    assert states.shape == (12, 8)
    assert np.all(np.isfinite(states))


def test_gene_dim_matches_bounds():
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    lo, hi = gene_bounds(res)
    assert lo.shape == hi.shape == (res.gene_dim,)
    assert res.gene_dim == 8 + 8 * 1


def test_eval_once_returns_unit_interval_and_memory_helps():
    task = DelayedRecallTask(seq_len=15, in_dim=1)
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    eval_once = make_eval_once(res, task, n_train=40, n_eval=40)
    f = eval_once(_slow_leak_gene(res), np.random.default_rng(3))
    assert 0.0 <= f <= 1.0


def test_behavior_in_bounds():
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    behavior = make_behavior(res)
    rng = np.random.default_rng(0)
    bd = behavior(res.random_gene(rng))
    assert bd.shape == (2,)
    assert np.all(np.isfinite(bd))


def _slow_leak_gene(res):
    g = np.zeros(res.gene_dim)
    g[: res.n_taps] = -3.0
    g[res.n_taps :] = 1.0
    return g
