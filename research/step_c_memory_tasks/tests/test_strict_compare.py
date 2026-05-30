import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strict_compare import strict_compare


def test_clear_win_passes():
    a = np.linspace(0.6, 0.9, 16)
    b = np.linspace(0.1, 0.3, 16)
    r = strict_compare(a, b, "map_elites", "random")
    assert r.passes
    assert r.diff > 0


def test_too_few_seeds_fails():
    a = np.array([0.9, 0.9, 0.9])
    b = np.array([0.1, 0.1, 0.1])
    r = strict_compare(a, b, "map_elites", "random")
    assert not r.passes  # n_seeds < 15


def test_tie_fails():
    a = np.full(16, 0.5)
    b = np.full(16, 0.5)
    r = strict_compare(a, b, "map_elites", "random")
    assert not r.passes
