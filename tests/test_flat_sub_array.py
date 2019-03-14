"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend.utils import _to_flat_sub_array, _from_flat_sub_array


def _test_single(shape, axis):
    x = np.random.rand(*shape)
    x_flat = _to_flat_sub_array(x, axis=axis)
    y = _from_flat_sub_array(x_flat, axis=axis, shape=shape)
    return np.allclose(x, y)


def test_flat_sub_array(n=100):
    for _ in range(n):
        shape = np.random.randint(11, 23, np.random.randint(2, 5))
        axis = tuple(np.random.choice(tuple(range(len(shape))), np.random.randint(1, len(shape) + 1), replace = False))
        s = _test_single(shape, axis)
        print("shape = %s\taxis = %s\n  %s"%(str(shape), str(axis), s))
        assert s


if __name__ == '__main__':
    test_flat_sub_array()
