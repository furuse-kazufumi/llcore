# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landscape_map import multimodality_report


def test_unimodal_reports_no_valley():
    target = np.array([0.5, 0.5])
    eval_once = lambda g, rng: float(np.exp(-np.sum((g - target) ** 2)))
    lo, hi = np.zeros(2), np.ones(2)
    rep = multimodality_report(eval_once, dim=2, bounds=(lo, hi), n_restarts=12,
                               n_evals=200, sigma=0.1, base_seed=0)
    assert rep["valley_fraction"] < 0.1


def test_report_keys():
    eval_once = lambda g, rng: float(-np.sum(g ** 2))
    lo, hi = -np.ones(2), np.ones(2)
    rep = multimodality_report(eval_once, dim=2, bounds=(lo, hi), n_restarts=6,
                               n_evals=100, sigma=0.1, base_seed=0)
    assert {"n_optima", "valley_fraction", "is_multimodal"} <= set(rep)
