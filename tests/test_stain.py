import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")

from scipy.ndimage import gaussian_filter  # noqa: E402

from augmend import Augmend, HEStaining, Identity  # noqa: E402
from augmend.transforms.stain import estimate_stain_matrix  # noqa: E402

# Ruifrok & Johnston H&E stain vectors
M_TRUE = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])
M_TRUE = M_TRUE / np.linalg.norm(M_TRUE, axis=1, keepdims=True)


def _he_image(n=128, seed=0):
    rng = np.random.RandomState(seed)
    c = np.stack([gaussian_filter(rng.rand(n, n), 4) for _ in range(2)], -1)
    c = np.clip((c - c.mean()) * 30 + 0.5, 0, None) * np.array([1.2, 0.6])
    return np.round(255 * np.exp(-(c @ M_TRUE))).astype(np.uint8)


def test_estimate_stain_matrix():
    M = estimate_stain_matrix(_he_image().astype(np.float32) / 255)
    assert np.allclose(np.abs((M * M_TRUE).sum(axis=1)), 1, atol=1e-3)


@pytest.mark.parametrize("dtype", (np.uint8, np.float32, np.float64))
def test_dtype_and_shape(dtype):
    x = _he_image()
    x = x if dtype == np.uint8 else (x / 255).astype(dtype)
    y = HEStaining()(x, rng=np.random.RandomState(0))
    assert y.dtype == x.dtype and y.shape == x.shape
    scale = 255 if dtype == np.uint8 else 1
    assert np.abs(y.astype(np.float64) - x).mean() > 0.01 * scale


def test_uint8_and_float_agree():
    x = _he_image()
    y8 = HEStaining()(x, rng=np.random.RandomState(0))
    yf = HEStaining()(x.astype(np.float32) / 255, rng=np.random.RandomState(0))
    assert np.abs(y8 / 255 - yf).max() < 2 / 255


def test_reproducible():
    x = _he_image()
    a = HEStaining()(x, rng=np.random.RandomState(1))
    b = HEStaining()(x, rng=np.random.RandomState(1))
    c = HEStaining()(x, rng=np.random.RandomState(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_background_returned_unchanged():
    x = np.full((64, 64, 3), 240, np.uint8)
    assert np.array_equal(HEStaining()(x, rng=np.random.RandomState(0)), x)


@pytest.mark.parametrize("x", (np.zeros((32, 32, 4), np.uint8), np.zeros((32, 32, 3), np.uint16)))
def test_invalid_input(x):
    with pytest.raises(ValueError):
        HEStaining()(x)


def test_pipeline_with_labels_and_pickle():
    x = _he_image()
    y = (x[..., 0] < 128).astype(np.uint16)
    aug = Augmend()
    aug.add([HEStaining(), Identity()])
    aug2 = pickle.loads(pickle.dumps(aug))
    x1, y1 = aug([x, y], rng=np.random.RandomState(0))
    x2, y2 = aug2([x, y], rng=np.random.RandomState(0))
    assert np.array_equal(y1, y) and np.array_equal(x1, x2)
