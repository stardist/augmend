"""

mweigert@mpi-cbg.de
"""
import numpy as np
from augmend.utils import map_single_func_tree, map_trees, zip_trees, LeafTuple, pad_to_shape


def test_pad_to_shape():
    x = np.ones((4, 8), np.int32)
    y = pad_to_shape(x, (6, 6), cval=-1)
    assert y.shape == (6, 6) and y.dtype == x.dtype
    assert (y[1:5] == 1).all() and (y[[0, 5]] == -1).all()
    assert (pad_to_shape(x, (6, 6)) == np.where(y == -1, 0, y)).all()
    assert (pad_to_shape(x, (6, 6), mode="reflect", cval=-1) == 1).all()


def test_all():
    def mul(n):
        return lambda x: n*x

    x = [1, [2, 3], [3, 4, [5, 6]]]
    y = map_single_func_tree(mul(2), x)
    print(y)
    assert y == [2, [4, 6], [6, 8, [10, 12]]]


    x = [(1, 2), 3, (4,5) ]
    funcs = [(mul(2),mul(3)),mul(5),(mul(1),mul(2))]

    y = map_trees(funcs, x)
    print(y)
    assert y == [(2, 6), 15, (4,10) ]

    x = (('A', 'B'), 'C' ,['D','E'])
    y = ((1, 2), 3, (4,5))
    z = zip_trees(x, y)
    print(z)
    assert z == ((LeafTuple(('A', 1)), LeafTuple(('B', 2))), LeafTuple(('C', 3)), [LeafTuple(('D', 4)), LeafTuple(('E', 5))])

if __name__ == '__main__':

    test_all()