import numpy as np
import pytest

from augmend import DropPlanes


def _dropped_rows(n, width, seed):
    x = np.ones((40, 20), np.float32)
    y = DropPlanes(axis=0, n=n, width=width)(x, rng=np.random.RandomState(seed))
    return np.where((y == 0).all(axis=1))[0]


def _runs(rows):
    return [len(r) for r in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)]


@pytest.mark.parametrize("width", (1, 3, 5))
@pytest.mark.parametrize("seed", range(5))
def test_drop_planes_single(width, seed):
    rows = _dropped_rows(n=1, width=width, seed=seed)
    assert _runs(rows) == [width]


@pytest.mark.parametrize("n,width", [(3, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("seed", range(5))
def test_drop_planes_multiple(n, width, seed):
    rows = _dropped_rows(n=n, width=width, seed=seed)
    # blocks may overlap, but every block has exactly `width` planes
    assert width <= len(rows) <= n * width
    assert all(r >= width for r in _runs(rows))
