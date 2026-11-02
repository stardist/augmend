import numpy as np
import pytest

from augmend import IntensityScaleShift


def test_invert_keeps_background():
    x = np.ones((16, 16), np.float32)
    x[4:8] = 0.2
    y = IntensityScaleShift(scale=(0.5, 2), shift=(0, 0), invert=True)(x, rng=np.random.RandomState(0))
    assert np.allclose(y[x == 1], 1)
    assert not np.allclose(y[4:8], 0.2)


@pytest.mark.parametrize("seed", range(3))
def test_invert_equals_transform_of_inverse(seed):
    x = np.random.RandomState(seed).rand(16, 16, 3).astype(np.float32)
    kwargs = dict(scale=(0.8, 1.2), shift=(-0.1, 0.1), axis=(0, 1))
    y = IntensityScaleShift(invert=True, **kwargs)(x, rng=np.random.RandomState(seed))
    y_ref = 1 - IntensityScaleShift(**kwargs)(1 - x, rng=np.random.RandomState(seed))
    assert np.allclose(y, y_ref)


def test_invert_rejects_integer_images():
    with pytest.raises(ValueError):
        IntensityScaleShift(invert=True)(np.zeros((8, 8), np.uint8))
