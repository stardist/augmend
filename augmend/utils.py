"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from functools import partial

def _get_global_rng():
    rng = np.random.RandomState()
    rng.set_state(np.random.get_state())
    return rng

class LeafTuple(tuple):
    """
    As we implicitely assume that a tuple or list cannot be a terminal leaf node,
    we have to add tuples as a different type - as a LeafTuple
    """
    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, super().__repr__())

def _is_leaf_node(x):
    """defines which nodes are considered a terminal leaf node
    """
    return isinstance(x, LeafTuple) or not isinstance(x, (tuple, list))

def _wrap_leaf_node(x):
    # to enable syntatic sugar: special case of trivial tree, wrap transform in list
    return (x,) if _is_leaf_node(x) else x

def _raise(e):
    raise e


def _all_of_type(iterable, t):
    try:
        for it in iterable:
            if not isinstance(it,t) and not _all_of_type(it,t):
                return False
        return True
    except TypeError:
        return False


def _normalized_weights(weights,n):
    if weights is None:
        weights = [1] * n
    len(weights) == n or _raise(ValueError("must be %d weights" % n))
    all(w>=0 for w in weights) or _raise(ValueError("all weights must be positive"))
    weights = np.asanyarray(weights)
    return weights / np.sum(weights)


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
    all_leaves = (_is_leaf_node(x) and _is_leaf_node(func))

    (all_leaves or len(func) == len(x)) or _raise(ValueError("tree structures not compatible %s" % str(x)))

    return func(x) if all_leaves \
        else type(x)(map(map_trees, func, x))


def zip_trees(*xs):
    """
    zip several trees (nested list or tuples)

    the leaf nodes will be represented by a LeafTuple object (a subclass of tuple),
     such that it does not get interpreted as an inner node of the resulting tree

    Example:
    ========

    x = ( ('A','B') , 'C' )
    y = ( (1,2) , 3 )

    zip_trees(x,y)

    >>>((LeafTuple(('A', 1)), LeafTuple(('B', 2))), LeafTuple(('C', 3)))

    """
    all(len(xs[0]) == l for l in map(len, xs)) or _raise(ValueError("tree structures not compatible %s" % str(xs)))

    return type(xs[0])(LeafTuple(t) if all(_is_leaf_node(_t) for _t in t) \
                           else zip_trees(*t) for t in zip(*xs))


def create_pattern(ndim=2, shape=None, dtype=np.float32):
    if shape is None:
        shape = (128,) * ndim

    x = np.zeros(shape, dtype)
    hs = (np.linspace(min(shape) // 4, 2*max(shape) // 3, len(shape))[::-1]).astype(int)
    w = min(shape) // 12

    for i, h in enumerate(hs):
        # ss = list(slice(None) for _ in shape)
        # ss[i] = slice(0,None,2*w)

        ss = list(slice(w//4, None, 2 * w) for _ in shape)
        ss[i] = slice(None)
        x[ss] = 128

    for i, h in enumerate(hs):
        ss = list(slice(_s//2 - w -_h//2, _s//2 + w-_h//2 ) for _s,_h in zip(shape, hs))
        ss[i] = slice(shape[i]//2-w -h//2 ,shape[i]//2+h//2)
        x[ss] = 256


    return x
