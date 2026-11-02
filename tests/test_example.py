"""3D example from the README."""

import numpy as np

from augmend import Augmend, Elastic, FlipRot90


def test_example_3d():
    aug = Augmend()
    aug.add([FlipRot90(axis=(0, 1, 2)), FlipRot90(axis=(0, 1, 2))], probability=0.9)
    aug.add([Elastic(axis=(0, 1, 2), amount=5, order=1),
             Elastic(axis=(0, 1, 2), amount=5, order=0)], probability=0.9)

    x = np.zeros((64,) * 3, np.float32)
    Xs = np.meshgrid(*((np.arange(64),) * 3), indexing="ij")
    R = np.sqrt(np.sum([(X - c) ** 2 for X, c in zip(Xs, (40, 30, 25))], axis=0))
    x[R < 12] = 1.4
    y = np.zeros((64,) * 3, np.uint16)
    y[R < 12] = 200

    def data_gen():
        for _ in range(4):
            yield x, y

    res = tuple(aug.flow(data_gen()))

    assert len(res) == 4
    for x2, y2 in res:
        assert x2.shape == x.shape and y2.shape == y.shape
        assert y2.dtype == y.dtype
