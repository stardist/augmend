"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import itertools
import six
from functools import partial

def _get_global_rng():
    return np.random.random.__self__

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
    if isinstance(e, six.string_types):
        e = ValueError(e)
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
    all(np.isscalar(w) and w>=0 for w in weights) or _raise(ValueError("all weights must be non-negative numbers"))
    weights = np.asanyarray(weights)
    np.sum(weights) > 0 or _raise(ValueError("not all weights can be 0"))
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


def flatten_tree(x):
    if _is_leaf_node(x):
        return (x,)
    else:
        return tuple(itertools.chain.from_iterable((_x,) if _is_leaf_node(_x) else flatten_tree(_x) for _x in x))

def _flatten_axis(ndim, axis=None):
    """ converts axis to a flatten tuple
    e.g.
    flatten_axis(3, axis = None) = (0,1,2)
    flatten_axis(4, axis = (-2,-1)) = (2,3)
    """

    # allow for e.g. axis = -1, axis = None, ...
    all_axis = np.arange(ndim)

    if axis is None:
        axis = tuple(all_axis)
    else:
        if np.isscalar(axis):
            axis = [axis, ]
        elif isinstance(axis, tuple):
            axis = list(axis)
        if max(axis) > max(all_axis):
            raise ValueError("axis = %s too large" % max(axis))
        axis = tuple(list(all_axis[axis]))
    return axis


def _to_flat_sub_array(arr, axis):
    axis = _flatten_axis(arr.ndim, axis)
    flat_axis = tuple(i for i in range(arr.ndim) if i not in axis)
    permute_axis = flat_axis + axis
    flat_shape = (-1,) + tuple(s for i, s in enumerate(arr.shape) if i in axis)
    arr_t = arr.transpose(permute_axis).reshape(flat_shape)
    return arr_t


def _from_flat_sub_array(arr, axis, shape):
    axis = _flatten_axis(len(shape), axis)
    flat_axis = tuple(i for i in range(len(shape)) if i not in axis)
    permute_axis = flat_axis + axis
    inv_permute_axis = tuple(permute_axis.index(i) for i in range(len(shape)))
    permute_shape = tuple(shape[p] for p in permute_axis)
    arr_t = arr.reshape(permute_shape)
    arr_t = arr_t.transpose(inv_permute_axis)
    return arr_t


def test_pattern(ndim=2, shape=None, dtype=np.float32):
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




def plot_augmented(transform, x, n = 4, rng = None, num=None, **kwargs):
    import matplotlib.pyplot as plt
    xs = flatten_tree(x)
    nx = len(xs)
    ys = tuple((x,) + tuple(transform(x, rng=rng) for _ in range(n)) for x in xs)
    titles = ("original",) + tuple("Augment_%s"%i for i in range(n))
    fig = plt.figure(num=num, figsize=(2+1*n,2+1*nx))
    fig.subplots_adjust(wspace = 0.05, hspace = 0.05, left = .05, top = 0.95, bottom = 0.05, right = 0.95)
    fig.clf()
    axs = fig.subplots(nx,n+1)
    if nx==1:
        axs = (axs,)

    for i,(axx, y) in enumerate(zip(axs,ys)):
        for ax,_y, t in zip(axx, y,titles):
            ax.imshow(_y, **kwargs)
            ax.axis('off')
            if i==0:
                ax.set_title(t, fontsize = 8)

    return fig