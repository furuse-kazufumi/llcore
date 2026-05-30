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
    assert res.gene_dim == res.n_taps + res.n_taps * res.in_dim


def test_eval_once_returns_unit_interval():
    task = DelayedRecallTask(seq_len=15, in_dim=1)
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    eval_once = make_eval_once(res, task, n_train=40, n_eval=40)
    f = eval_once(_slow_leak_gene(res), np.random.default_rng(3))
    assert 0.0 <= f <= 1.0
    # honest observation: fast-leak でも DelayedRecall は高 R² (t=0 cue が力学系の
    # 固定点で保持される) → このタスクは記憶を強く要求しない兆候。C1 landscape で扱う。


def test_behavior_in_bounds():
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    behavior = make_behavior(res)
    rng = np.random.default_rng(0)
    bd = behavior(res.random_gene(rng))
    assert bd.shape == (2,)
    assert np.all(np.isfinite(bd))


def test_in_dim_2_run_and_bounds():
    res = LeakyDelayLineReservoir(n_taps=6, in_dim=2)
    assert res.gene_dim == res.n_taps + res.n_taps * res.in_dim
    lo, hi = gene_bounds(res)
    assert lo.shape == hi.shape == (res.gene_dim,)
    rng = np.random.default_rng(0)
    states = res.run(res.random_gene(rng), np.sign(rng.normal(size=(10, 2))))
    assert states.shape == (10, 6) and np.all(np.isfinite(states))


def _slow_leak_gene(res):
    g = np.zeros(res.gene_dim)
    g[: res.n_taps] = -3.0
    g[res.n_taps :] = 1.0
    return g
