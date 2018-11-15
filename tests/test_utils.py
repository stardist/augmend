"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend.utils import map_single_func_tree, map_trees, zip_trees, LeafTuple


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