import numpy as np
from scipy import ndimage
import itertools
from .base import BaseTransform
from ..utils import _raise, _get_global_rng, _flatten_axis, _from_flat_sub_array, _to_flat_sub_array


def subgroup_permutations(ndim, axis=None):
    """
    iterate over the permutation subgroup of given axis
    """
    axis = _flatten_axis(ndim, axis)
    res = np.arange(ndim)
    for perm in itertools.permutations(axis):
        for a, p in zip(axis, perm):
            res[a] = p
        yield tuple(res)


def subgroup_flips(ndim, axis=None):
    """
    iterate over the product subgroup (False,True) of given axis
    """
    axis = _flatten_axis(ndim, axis)
    res = np.zeros(ndim, np.bool)
    for prod in itertools.product((False, True), repeat=len(axis)):

        for a, p in zip(axis, prod):
            res[a] = p
        yield tuple(res)


def transform_flip_rot90(img, rng=None, axis=None):
    """
    random augmentation of an array around axis
    """

    if rng is None or rng is np.random:
        rng = _get_global_rng()

    # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
    axis = _flatten_axis(img.ndim, axis)

    # list of all permutations
    perms = tuple(subgroup_permutations(img.ndim, axis))

    # list of all flips
    flips = tuple(subgroup_flips(img.ndim, axis))

    # random permutation and flip
    rand_perm_ind = rng.randint(len(perms))
    rand_flip_ind = rng.randint(len(flips))

    rand_perm = perms[rand_perm_ind]
    rand_flip = flips[rand_flip_ind]

    # first random permute
    augmented = img.transpose(rand_perm)

    # then random flip
    for axis, f in enumerate(rand_flip):
        if f:
            augmented = np.flip(augmented, axis)
    return augmented


def random_rotation_matrix(ndim=2, rng=np.random):
    """
    adapted from pg 11 of 

    Mezzadri, Francesco. 
    "How to generate random matrices from the classical compact groups." 
    arXiv preprint math-ph/0609050 (2006).

    """
    z = rng.randn(ndim, ndim)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = np.multiply(q, ph, q)
    # enforce parity
    q *= np.linalg.det(q)
    return q


def transform_rotation(img, rng=None, axis=None, offset=None, mode="constant", order=1):
    """
    random roation around axis
    """

    if rng is None:
        rng = np.random
    if offset is None:
        offset = tuple(s // 2 for s in img.shape)

    # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
    axis = _flatten_axis(img.ndim, axis)

    M_rot = random_rotation_matrix(len(axis))
    M = np.identity(img.ndim)
    M[np.ix_(np.array(axis), np.array(axis))] = M_rot

    # as scipy.ndimage applies the offset *after* the affine matrix...
    offset -= np.dot(M, offset)

    return ndimage.affine_transform(img, M, offset=offset, order=order, mode=mode)




class FlipRot90(BaseTransform):
    """
    flip and 90 degree rotation augmentation
    """

    def __init__(self, axis=None):
        """
        :param axis, tuple:
            the axis along which to flip and rotate
        """
        super().__init__(
            default_kwargs=dict(axis=axis),
            transform_func=transform_flip_rot90
        )
