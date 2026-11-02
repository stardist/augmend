import numpy as np
import pytest

from augmend.transforms.elastic import transform_elastic


@pytest.mark.parametrize("order", (0, 1))
def test_elastic_workers(order):
    img = np.zeros((4, 32, 32, 32), np.uint16)
    img[1:-1, 8:-8, 8:-8] = 128

    kwargs = dict(axis=(1, 2, 3), grid=4, amount=5, order=order)
    out1 = transform_elastic(img, rng=np.random.RandomState(0), workers=1, **kwargs)
    out2 = transform_elastic(img, rng=np.random.RandomState(0), workers=4, **kwargs)

    assert np.allclose(out1, out2)
