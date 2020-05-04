import os
import sys
import numpy as np
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from .base import BaseTransform
from ..utils import _raise, _validate_rng, _flatten_axis, _from_flat_sub_array, _to_flat_sub_array


def abspath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


def _zoom_and_transform_cpu(img, dxs_coarse, order):
    """
    img: is a ndarray of ndim dimension  
    dxs_coarse = (ndim, grid[0], grid[1],..., grid[ndim])
    """
    zoom_factor = tuple(s / g for s, g in zip(img.shape, dxs_coarse[0].shape))
    dxs = tuple(ndimage.zoom(dx, zoom_factor, order=1) for dx in dxs_coarse)
    Xs = np.meshgrid(*tuple(np.arange(s) for s in img.shape), indexing='ij')
    indices = tuple(np.reshape(X + dx, (-1, 1)) for X, dx in zip(Xs, dxs))
    return ndimage.map_coordinates(img, indices, order=order).reshape(img.shape)


def _zoom_and_transform_gpu(img, dxs_coarse, order):
    """
    img: is a ndarray of ndim dimension  
    dxs_coarse = (ndim, grid[0], grid[1],..., grid[ndim])
    """
    assert img.ndim in (2, 3)
    assert order in (0, 1)
    from gputools import OCLProgram, OCLImage, OCLArray
    #print("using gpu...")

    order_defines = {0: ["-D", "SAMPLERFILTER=CLK_FILTER_NEAREST"],
                     1: ["-D", "SAMPLERFILTER=CLK_FILTER_LINEAR"]}

    options_types = {np.uint8: ["-D", "TYPENAME=uchar", "-D", "READ_IMAGE=read_imageui"],
                     np.uint16: ["-D", "TYPENAME=short", "-D", "READ_IMAGE=read_imageui"],
                     np.int32: ["-D", "TYPENAME=int", "-D", "READ_IMAGE=read_imagei"],
                     np.float32: ["-D", "TYPENAME=float", "-D", "READ_IMAGE=read_imagef"]}

    dtype = img.dtype.type
    if not dtype in options_types:
        raise ValueError("type %s not supported! Available: %s" % (dtype, str(list(options_types.keys()))))

    img_im = OCLImage.from_array(img)
    dxs_im = tuple(OCLImage.from_array(dx) for dx in dxs_coarse)
    res_g = OCLArray.empty(img.shape, dtype)

    prog = OCLProgram(abspath("kernels/elastic.cl"),
                      build_options=order_defines[order] + options_types[dtype])

    prog.run_kernel("zoom_and_transform%s"%img.ndim,
                    res_g.shape[::-1], None,
                    img_im, *dxs_im, res_g.data)
    return res_g.get().astype(dtype, copy = False)
    #
    # zoom_factor = tuple(s / g for s, g in zip(img.shape, dxs_coarse[0].shape))
    # dxs = tuple(ndimage.zoom(dx, zoom_factor, order=1) for dx in dxs_coarse)
    # Xs = np.meshgrid(*tuple(np.arange(s) for s in img.shape), indexing='ij')
    # indices = tuple(np.reshape(X + dx, (-1, 1)) for X, dx in zip(Xs, dxs))
    # return ndimage.map_coordinates(img, indices, order=order).reshape(img.shape)
    #

def transform_elastic(img, rng=None, axis=None, grid=5, amount=5, order=1, workers=1, use_gpu=False):
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

    axis = _flatten_axis(img.ndim, axis)

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

    rng = _validate_rng(rng)

    if len(axis) < img.ndim:
        # flatten all axis that are not affected
        img_flattened = _to_flat_sub_array(img, axis)
        # state = rng.get_state()

        def _func(x, rng):
            # rng.set_state(state)
            # print(rng.uniform(0,1))
            return transform_elastic(x, rng=rng,
                                     axis=None, grid=grid, amount=amount, order=order,
                                     workers=1,
                                     use_gpu = use_gpu
                                     )

        # copy rng, to be thread-safe
        rng_flattened = tuple(deepcopy(rng) for _ in img_flattened)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                res_flattened = np.stack(tuple(executor.map(_func, img_flattened, rng_flattened)))
        else:
            res_flattened = np.stack(tuple(map(_func, img_flattened, rng_flattened)))

        return _from_flat_sub_array(res_flattened, axis, img.shape)

    else:
        # print(np.sum(rng.get_state()[1]))
        dxs_coarse = list((a * rng.uniform(-1, 1, grid)).astype(np.float32) for a in amount)
        # make sure, the border dxs are pointing inwards, such that
        # we dont have out-of-border pixel accesses

        for ax in range(img.ndim):
            ss = [slice(None) for i in range(img.ndim)]
            ss[ax] = slice(0, 1)
            dxs_coarse[ax][tuple(ss)] *= np.sign(dxs_coarse[ax][tuple(ss)])
            ss[ax] = slice(-1, None)
            dxs_coarse[ax][tuple(ss)] *= -np.sign(dxs_coarse[ax][tuple(ss)])

        if use_gpu and img.ndim in (2,3):
            res = _zoom_and_transform_gpu(img, dxs_coarse=dxs_coarse, order=order)
        else:
            res = _zoom_and_transform_cpu(img, dxs_coarse=dxs_coarse, order=order)

        return res


#############################################################################

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

    def __init__(self, axis=None, grid=5, amount=5, order=1, workers = 0, use_gpu = False):
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
                workers = workers,
                use_gpu = use_gpu,
                order=order),
            transform_func=transform_elastic
        )
