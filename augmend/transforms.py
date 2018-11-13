import numpy as np
from scipy.ndimage.interpolation import zoom, map_coordinates
import itertools


def transform_elastic(img, grid=(5, 5), amount=5, order=1, random_generator  =None):
    img = np.asanyarray(img)

    if np.isscalar(grid):
        grid = (grid,) * img.ndim
    if np.isscalar(amount):
        amount = (amount,) * img.ndim

    grid = np.asanyarray(grid)
    amount = np.asanyarray(amount)


    assert img.ndim == len(grid) and len(grid) == len(amount)
    if np.amin(grid) < 2:
        raise ValueError("grid should be at least 2x2")

    if random_generator is None:
        random_generator = np.random

    dxs_coarse = list(a * random_generator.uniform(-1, 1, grid) for a in amount)


    # make sure, the border dxs are pointing inwards, such that
    # we dont have out-of-border pixel accesses

    for axis in range(img.ndim):
        ss = [slice(None) for i in range(img.ndim)]
        ss[axis] = slice(0, 1)
        dxs_coarse[axis][ss] *= np.sign(dxs_coarse[axis][ss])
        ss[axis] = slice(-1, None)
        dxs_coarse[axis][ss] *= -np.sign(dxs_coarse[axis][ss])


    zoom_factor = tuple(s / g for s, g in zip(img.shape, grid))

    dxs = tuple(zoom(dx, zoom_factor, order=1) for dx in dxs_coarse)


    Xs = np.meshgrid(*tuple(np.arange(s) for s in img.shape), indexing='ij')

    indices = tuple(np.reshape(X + dx, (-1, 1)) for X, dx in zip(Xs, dxs))

    return map_coordinates(img, indices, order=order).reshape(img.shape)


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
        axis = tuple(list(all_axis[axis]))
    return axis


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



def transform_flip_rot(img, axis=None, random_generator=None):
    """
    random augmentation of an array around axis 
    """

    if random_generator is None:
        random_generator = np.random

    # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
    axis = flatten_axis(img.ndim, axis)

    # list of all permutations
    perms = tuple(subgroup_permutations(img.ndim, axis))

    # list of all flips
    flips = tuple(subgroup_flips(img.ndim, axis))

    # random permutation and flip
    rand_perm_ind = random_generator.randint(len(perms))
    rand_flip_ind = random_generator.randint(len(flips))

    rand_perm = perms[rand_perm_ind]
    rand_flip = flips[rand_flip_ind]

    # first random permute
    augmented = img.transpose(rand_perm)

    # then random flip
    for axis, f in enumerate(rand_flip):
        if f:
            augmented = np.flip(augmented, axis)
    return augmented

