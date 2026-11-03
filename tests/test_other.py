import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from augmend import DropPlanes, GaussianBlur
from augmend.transforms.other import _gaussian_filter


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


@pytest.mark.parametrize("shape", ((64,), (32, 32), (8, 16, 16), (16, 16, 3)))
@pytest.mark.parametrize("dtype", (np.uint8, np.float32, np.float64))
def test_gaussian_filter_matches_scipy(shape, dtype):
    rng = np.random.RandomState(0)
    x = (255 * rng.rand(*shape)).astype(dtype)
    sigmas = tuple(rng.uniform(0, 3, len(shape)))
    y = _gaussian_filter(x, sigmas, mode="reflect")
    assert y.dtype == x.dtype and y.flags["C_CONTIGUOUS"]
    assert np.array_equal(y, gaussian_filter(x, sigmas, mode="reflect"))


def test_gaussian_blur_scalar_amount():
    x = np.random.RandomState(0).rand(16, 16).astype(np.float32)
    assert np.allclose(GaussianBlur(amount=2)(x), gaussian_filter(x, 2))
