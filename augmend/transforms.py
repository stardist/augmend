import numpy as np
from scipy.ndimage.interpolation import zoom, map_coordinates
import itertools
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
import copy


def flatten_axis(ndim, axis=None):
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
    axis = flatten_axis(arr.ndim, axis)
    flat_axis = tuple(i for i in range(arr.ndim) if i not in axis)
    permute_axis = flat_axis + axis
    flat_shape = (-1,) + tuple(s for i, s in enumerate(arr.shape) if i in axis)
    arr_t = arr.transpose(permute_axis).reshape(flat_shape)
    return arr_t


def _from_flat_sub_array(arr, axis, shape):
    axis = flatten_axis(len(shape), axis)
    flat_axis = tuple(i for i in range(len(shape)) if i not in axis)
    permute_axis = flat_axis + axis
    inv_permute_axis = tuple(permute_axis.index(i) for i in range(len(shape)))
    permute_shape = tuple(shape[p] for p in permute_axis)
    arr_t = arr.reshape(permute_shape)
    arr_t = arr_t.transpose(inv_permute_axis)
    return arr_t


#
# def transform_elastic(img, rng=None, axis=None, grid=4, amount=5, order=1):
#     """
#     elastic deformation of an n-dimensional image along the given axis
#
#     :param img, ndarray:
#         the nD image to deform
#     :param axis, tuple or callable:
#         the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
#     :param grid, int, tuple of ints of same length as axis, or callable:
#         the number of gridpoints per axis at which random deformation vectors are attached.
#     :param amount, float, tuple of floats of same length as axis, or callable:
#         the maximal pixel shift of deformations per axis.
#     :param order, int or callable:
#         the interpolation order (e.g. set order = 0 for nearest neighbor)
#     :param rng:
#         the random number generator to be used
#     :return ndarray:
#         the deformed img/array
#
#     Example:
#     ========
#
#     img = np.zeros((128,) * 2, np.float32)
#
#     img[::16] = 128
#     img[:,::16] = 128
#
#     out = transform_elastic(img, grid=5, amount=5)
#
#
#     """
#     img = np.asanyarray(img)
#
#     if callable(axis):
#         axis = axis(img)
#
#     axis = flatten_axis(img.ndim, axis)
#
#     if np.isscalar(grid):
#         grid = (grid,) * len(axis)
#     elif callable(grid):
#         grid = grid(img)
#
#     if np.isscalar(amount):
#         amount = (amount,) * len(axis)
#     elif callable(amount):
#         amount = amount(img)
#
#     if callable(order):
#         order = order(img)
#
#     grid = np.asanyarray(grid)
#     amount = np.asanyarray(amount)
#
#     if not img.ndim >= len(axis):
#         raise ValueError("dimension of image (%s) < length of axis (%s)" % (img.ndim, len(axis)))
#
#     if not len(axis) == len(grid):
#         raise ValueError("length of axis (%s) != length of grid (%s)" % (len(axis), len(grid)))
#
#     if not len(axis) == len(amount):
#         raise ValueError("length of axis (%s) != length of amount (%s)" % (len(axis), len(amount)))
#
#     if np.amin(grid) < 2:
#         raise ValueError("grid should be at least 2x2 (but is %s)" % str(grid))
#
#     if rng is None:
#         rng = np.random
#
#     grid_full = np.ones(img.ndim, np.int)
#     grid_full[np.array(axis)] = np.array(grid)
#
#     amount_full = np.zeros(img.ndim, np.float32)
#     amount_full[np.array(axis)] = np.array(amount)
#
#     dxs_coarse = list(a * rng.uniform(-1, 1, grid_full) for a in amount_full)
#
#     # make sure, the border dxs are pointing inwards, such that
#     # we dont have out-of-border pixel accesses
#
#     for ax in range(img.ndim):
#         ss = [slice(None) for i in range(img.ndim)]
#         ss[ax] = slice(0, 1)
#         dxs_coarse[ax][ss] *= np.sign(dxs_coarse[ax][ss])
#         ss[ax] = slice(-1, None)
#         dxs_coarse[ax][ss] *= -np.sign(dxs_coarse[ax][ss])
#
#     zoom_factor = tuple(s / g if i in axis else 1 for i, (s, g) in enumerate(zip(img.shape, grid_full)))
#
#     dxs = tuple(np.broadcast_to(zoom(dx, zoom_factor, order=1), img.shape) for dx in dxs_coarse)
#
#     Xs = np.meshgrid(*tuple(np.arange(s) for s in img.shape), indexing='ij')
#
#     indices = tuple(np.reshape(X + dx, (-1, 1)) for X, dx in zip(Xs, dxs))
#
#     return map_coordinates(img, indices, order=order).reshape(img.shape)


# FIXME: the one below is more flexible, as it allows to apply independent distortions
# FIXME: along the axis it is not applied to (e.g. batches)

def transform_elastic(img, rng=None, axis=None, grid=5, amount=5, order=1, workers=1):
    """
    elastic deformation of an n-dimensional image along the given axis

    :param img, ndarray:
        the nD image to deform
    :param rng:
        the random number generator to be used
    :param axis, tuple or callable:
        the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
    :param grid, int, tuple of ints of same length as axis, or callable:
        the number of gridpoints per axis at which random deformation vectors are attached.
    :param amount, float, tuple of floats of same length as axis, or callable:
        the maximal pixel shift of deformations per axis.
    :param order, int or callable:
        the interpolation order (e.g. set order = 0 for nearest neighbor)
    :param workers, int:
        if >1 uses multithreading with the given number of workers 

    :return ndarray:
        the deformed img/array

    Example:
    ========

    img = np.zeros((128,) * 2, np.float32)

    img[::16] = 128
    img[:,::16] = 128

    out = transform_elastic(img, grid=5, amount=5)


    """

    img = np.asanyarray(img)

    axis = flatten_axis(img.ndim, axis)

    if np.isscalar(grid):
        grid = (grid,) * len(axis)
    if np.isscalar(amount):
        amount = (amount,) * len(axis)

    grid = np.asanyarray(grid)
    amount = np.asanyarray(amount)

    if not img.ndim >= len(axis):
        raise ValueError("dimension of image (%s) < length of axis (%s)" % (img.ndim, len(axis)))

    if not len(axis) == len(grid):
        raise ValueError("length of axis (%s) != length of grid (%s)" % (len(axis), len(grid)))

    if not len(axis) == len(amount):
        raise ValueError("length of axis (%s) != length of amount (%s)" % (len(axis), len(amount)))

    if np.amin(grid) < 2:
        raise ValueError("grid should be at least 2x2 (but is %s)" % str(grid))

    if rng is None:
        rng = np.random

    if len(axis) < img.ndim:
        # flatten all axis that are not affected
        img_flattened = _to_flat_sub_array(img, axis)
        state = rng.get_state()

        def _func(x, rng):
            rng.set_state(state)
            return transform_elastic(x, rng=rng,
                                     axis=None, grid=grid, amount=amount, order=order,
                                     workers=1
                                     )

        # copy rng, to be thread-safe
        rng_flattened = tuple(copy.deepcopy(rng) for _ in img_flattened)

        if workers > 1:

            with ThreadPoolExecutor(max_workers=workers) as executor:
                res_flattened = np.stack(executor.map(_func, img_flattened, rng_flattened))
        else:
            res_flattened = np.stack(map(_func, img_flattened, rng_flattened))

        return _from_flat_sub_array(res_flattened, axis, img.shape)

    else:
        # print(np.sum(rng.get_state()[1]))
        dxs_coarse = list(a * rng.uniform(-1, 1, grid) for a in amount)

        # make sure, the border dxs are pointing inwards, such that
        # we dont have out-of-border pixel accesses

        for ax in range(img.ndim):
            ss = [slice(None) for i in range(img.ndim)]
            ss[ax] = slice(0, 1)
            dxs_coarse[ax][tuple(ss)] *= np.sign(dxs_coarse[ax][tuple(ss)])
            ss[ax] = slice(-1, None)
            dxs_coarse[ax][tuple(ss)] *= -np.sign(dxs_coarse[ax][tuple(ss)])

        zoom_factor = tuple(s / g if i in axis else 1 for i, (s, g) in enumerate(zip(img.shape, grid)))

        dxs = tuple(zoom(dx, zoom_factor, order=1) for dx in dxs_coarse)

        Xs = np.meshgrid(*tuple(np.arange(s) for s in img.shape), indexing='ij')

        indices = tuple(np.reshape(X + dx, (-1, 1)) for X, dx in zip(Xs, dxs))

        return map_coordinates(img, indices, order=order).reshape(img.shape)


def subgroup_permutations(ndim, axis=None):
    """
    iterate over the permutation subgroup of given axis
    """
    axis = flatten_axis(ndim, axis)
    res = np.arange(ndim)
    for perm in itertools.permutations(axis):
        for a, p in zip(axis, perm):
            res[a] = p
        yield tuple(res)


def subgroup_flips(ndim, axis=None):
    """
    iterate over the product subgroup (False,True) of given axis
    """
    axis = flatten_axis(ndim, axis)
    res = np.zeros(ndim, np.bool)
    for prod in itertools.product((False, True), repeat=len(axis)):

        for a, p in zip(axis, prod):
            res[a] = p
        yield tuple(res)


def transform_flip_rot(img, rng=None, axis=None):
    """
    random augmentation of an array around axis
    """

    if rng is None:
        rng = np.random

    # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
    axis = flatten_axis(img.ndim, axis)

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


############################################################################



class BaseTransform(object):
    """
    base class for an augmentation action
    """

    def __init__(self, default_kwargs, transform_func):
        self._default_kwargs = default_kwargs
        self._transform_func = transform_func

    def __call__(self, x, rng=np.random, **kwargs):
        kwargs = {**self._default_kwargs, **kwargs}
        return self._transform_func(x,
                                    rng=rng,
                                    **kwargs)

    def __add__(self, other):
        return Concatenate([self, other])

    def __repr__(self):
        return self.__class__.__name__
        # kwargs_str = '\n'.join(" = ".join(map(str, item)) for item in self._default_kwargs.items())
        # return "%s\n\ndefault arguments:\n%s" % (self.__class__.__name__, kwargs_str)


class Concatenate(BaseTransform):
    def __init__(self, transforms):
        super().__init__(
            default_kwargs=dict(),
            transform_func=lambda x, rng: reduce(lambda x, f: f(x, rng), transforms, x)
        )


class Identity(BaseTransform):
    """
    Do nothing
    """

    def __init__(self):
        super().__init__(
            default_kwargs=dict(),
            transform_func=lambda x, rng: x
        )


class AdditiveNoise(BaseTransform):
    """
    Add gaussian noise
    """

    def __init__(self, sigma=.1):
        super().__init__(
            default_kwargs=dict(sigma=sigma),
            transform_func=lambda x, rng, sigma: x + (sigma(x, rng) if callable(sigma) else sigma) * rng.normal(0, 1,
                                                                                                                x.shape)
        )


class CutOut(BaseTransform):
    """
    Cut parts of the image
    """

    def __init__(self, width=16):
        super().__init__(
            default_kwargs=dict(width=width),
            transform_func=self._cutout)

    def _cutout(self, x, rng, width=16):
        if np.isscalar(width):
            width = (width,) * x.ndim
        assert all(tuple(w <= s for w, s in zip(width, x.shape)))
        x0 = tuple(rng.randint(0, s - w - 1) for w, s in zip(width, x.shape))
        ss = tuple(slice(_x0, _x0 + w) for w, _x0 in zip(width, x0))
        y = x.copy()
        y[ss] = 0
        return y


class Elastic(BaseTransform):
    """
    elastic deformation of an n-dimensional image along the given axis

    :param axis, tuple:
        the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
    :param grid, int or tuple of ints of same length as axis:
        the number of gridpoints per axis at which random deformation vectors are attached.
    :param amount, float or tuple of floats of same length as axis:
        the maximal pixel shift of deformations per axis.
    :param order, int:
        the interpolation order (e.g. set order = 0 for nearest neighbor)
    :param rng:
        the random number generator to be used
    :return ndarray:
        the deformed img/array

    Example:
    ========

    img = np.zeros((128,) * 2, np.float32)

    img[::16] = 128
    img[:,::16] = 128

    aug_elastic = ElasticAugmentor(grid=5, amount=5)

    out = aug_elastic(img)

    """

    def __init__(self, axis=None, grid=5, amount=5, order=1):
        """

        :param axis, tuple:
            the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
        :param grid, int or tuple of ints of same length as axis:
            the number of gridpoints per axis at which random deformation vectors are attached.
        :param amount, float or tuple of floats of same length as axis:
            the maximal pixel shift of deformations per axis.
        :param order, int:
            the interpolation order (e.g. set order = 0 for nearest neighbor)
        """

        super().__init__(
            default_kwargs=dict(
                grid=grid,
                axis=axis,
                amount=amount,
                order=order),
            transform_func=transform_elastic
        )


class FlipRot(BaseTransform):
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
            transform_func=transform_flip_rot
        )
