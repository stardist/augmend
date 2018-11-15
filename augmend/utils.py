"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from functools import partial

class LeafTuple(tuple):
    def __repr__(self):
        return "%s%s"%(self.__class__.__name__,  super().__repr__())

def _is_leaf_node(x):
    return isinstance(x, LeafTuple) or not isinstance(x, (tuple, list))

def _wrap_leaf_node(x):
    # to enable syntatic sugar: special case of trivial tree, wrap transform in list
    return (x,) if _is_leaf_node(x) else x

def map_single_func_tree(func, x):
    """
    applies func to all elements of x recursively
    (if the type of element is in iterate_types) such that the results has
    the same nested structure as x

    Example:
    ========

    func = lambda x: 2*x

    x = [1, [2, 3, (4,5)]]
    y = map_tree(func, x)

    print(x)
    print(y)

    [1, [2, 3, (4, 5)]]
    [2, [4, 6, (8, 10)]]

    """
    return func(x) if _is_leaf_node(x) \
        else type(x)(map(partial(map_single_func_tree, func), x))


def map_trees(func, x):
    """
    Applies a tree (nested list or tuples) of functions funcs to a
    corresponding tree of items xs.

    Example:
    ========

    funcs = [ ( lambda x: 2*x, lambda x: 3*x ), lambda x: 4*x ]
    xs = [ (1,2), 3  ]

    ys = map_trees(funcs, xs)

    print(xs)
    print(ys)

    >>>[(1, 2), 3]
    >>>[(2, 6), 12]

    """
    return func(x) if (_is_leaf_node(x) and _is_leaf_node(func)) \
        else type(x)(map(map_trees, func, x))



def zip_trees(*trees):
    """
    zip several trees (nested list or tuples)

    he leaf noes will be represented by a LeafTuple object (a subclass of tuple),
     such that it does not get interpreted as an inner node of the resulting tree

    Example:
    ========

    x = ( ('A','B') , 'C' )
    y = ( (1,2) , 3 )

    zip_trees(x,y)

    >>>((LeafTuple(('A', 1)), LeafTuple(('B', 2))), LeafTuple(('C', 3)))

    """
    assert all(len(trees[0]) == l for l in map(len, trees)), "all trees must have same size"

    return type(trees[0])(LeafTuple(t) if all(_is_leaf_node(_t) for _t in t) \
                     else zip_trees(*t) for t in zip(*trees))



# def zip_trees(*trees):
#     if all(_is_leaf_node(t) for t in trees):
#         return trees
#     else:
#         return tuple(zip_trees(*t) for t in zip(*trees))
