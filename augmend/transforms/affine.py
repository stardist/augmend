import numpy as np
from scipy import ndimage
import warnings
import itertools
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy



from .base import BaseTransform
from ..utils import _raise, _get_global_rng, _flatten_axis, _from_flat_sub_array, _to_flat_sub_array, pad_to_shape

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



def transform_flip(img, rng=None, axis=None):
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

    # random flip
    rand_flip_ind = rng.randint(len(flips))

    rand_flip = flips[rand_flip_ind]

    # random flip
    for axis, f in enumerate(rand_flip):
        if f:
            img = np.flip(img, axis)
    return img


def random_rotation_matrix(ndim=2, rng=None):
    """
    adapted from pg 11 of

    Mezzadri, Francesco.
    "How to generate random matrices from the classical compact groups."
    arXiv preprint math-ph/0609050 (2006).

    """
    if rng is None:
        rng = _get_global_rng()

    z = rng.randn(ndim, ndim)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = np.multiply(q, ph, q)
    # enforce parity
    q *= np.linalg.det(q)
    return q


def transform_rotation(img, rng=None, axis=None, offset=None, mode="constant", order=1, workers = 1):
    """
    random rotation around axis
    """

    if rng is None or rng is np.random:
        rng = _get_global_rng()

    # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
    axis = _flatten_axis(img.ndim, axis)

    if offset is None:
        offset = tuple(s // 2 for s in np.array(img.shape)[np.array(axis)])

    if len(axis) < img.ndim:
        # flatten all axis that are not affected
        img_flattened = _to_flat_sub_array(img, axis)
        state = rng.get_state()

        def _func(x, rng):
            rng.set_state(state)
            return transform_rotation(x, rng=rng,
                                     axis=None, offset = offset, order=order,
                                      workers = 1)

        # copy rng, to be thread-safe
        rng_flattened = tuple(deepcopy(rng) for _ in img_flattened)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                res_flattened = np.stack(executor.map(_func, img_flattened, rng_flattened))
        else:
            res_flattened = np.stack(map(_func, img_flattened, rng_flattened))

        return _from_flat_sub_array(res_flattened, axis, img.shape)
    else:
        M_rot = random_rotation_matrix(len(axis), rng)
        M = np.identity(img.ndim)
        M[np.ix_(np.array(axis), np.array(axis))] = M_rot

        # as scipy.ndimage applies the offset *after* the affine matrix...
        offset -= np.dot(M, offset)

        return ndimage.affine_transform(img, M, offset=offset, order=order, mode=mode)



def transform_scale(img, rng=None, axis=None,  factors=(1,2), order=1, use_gpu=False):
    """
    scale tranformation
    :param img, ndarray:
        the nD image to deform
    :param rng:
        the random number generator to be used
    :param axis, tuple or callable:
        the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
    :param factors, float, tuple of floats of same length as axis, or callable:
        the maximal scales per axis.
    :param order, int or callable:
        the interpolation order (e.g. set order = 0 for nearest neighbor)
    :return ndarray:
        the deformed img/array

    Example:
    ========

    img = np.zeros((128,) * 2, np.float32)

    img[::16] = 128
    img[:,::16] = 128

    out = transform_scale(img, axis = 1, amounts=(1,2))


    """
    if use_gpu:
        from gputools import scale as zoom_gputools

    img = np.asanyarray(img)

    axis = _flatten_axis(img.ndim, axis)

    if np.isscalar(factors):
        factors = (factors,factors) * len(axis)

    if np.isscalar(factors[0]):
        factors = (factors,) * len(axis)

    factors = np.asanyarray(factors)

    if not img.ndim >= len(axis):
        raise ValueError("dimension of image (%s) < length of axis (%s)" % (img.ndim, len(axis)))

    if not len(axis) == len(factors):
        raise ValueError("length of axis (%s) != length of amount (%s)" % (len(axis), len(factors)))

    if rng is None or rng is np.random:
        rng = _get_global_rng()

    if len(axis) < img.ndim:
        # flatten all axis that are not affected
        img_flattened = _to_flat_sub_array(img, axis)
        state = rng.get_state()

        def _func(x, rng):
            rng.set_state(state)
            return transform_scale(x, rng=rng,
                                     axis=None, factors = factors, order=order, use_gpu = use_gpu)

        # copy rng, to be thread-safe
        rng_flattened = tuple(deepcopy(rng) for _ in img_flattened)

        res_flattened = np.stack(map(_func, img_flattened, rng_flattened))

        return _from_flat_sub_array(res_flattened, axis, img.shape)

    else:

        scale = tuple(rng.uniform(lower, upper) for lower, upper in factors)

        if use_gpu and img.ndim==3:
            # print("scaling by %s via gputools"%str(scale))
            inter = {
                0: "nearest",
                1:"linear"}
            res = zoom_gputools(img, scale, interpolation= inter[order])
            res = pad_to_shape(ndimage.zoom(img, scale, order=order), img.shape)
        else:
            # print("scaling by %s via scipy"%str(scale))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                res = pad_to_shape(ndimage.zoom(img, scale, order=order), img.shape)
        return res


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

class Flip(BaseTransform):
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
            transform_func=transform_flip
        )


class Rotate(BaseTransform):
    """
    flip and 90 degree rotation augmentation
    """

    def __init__(self, axis=None, workers = 1):
        """
        :param axis, tuple:
            the axis along which to flip and rotate
        """
        super().__init__(
            default_kwargs=dict(
                axis=axis,
                workers=workers
            ),
            transform_func=transform_rotation
        )



class Scale(BaseTransform):
    """
    scale augmentation
    """

    def __init__(self, axis=None, factors=2, order=1, use_gpu =False):
        """
        :param axis, tuple:
            the axis along which to flip and rotate
        """
        super().__init__(
            default_kwargs=dict(
                axis=axis,
                factors = factors,
                order=order,
                use_gpu  = use_gpu
            ),
            transform_func=transform_scale
        )

