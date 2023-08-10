import numpy as np
from scipy import ndimage
import warnings
import itertools
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy


from .base import BaseTransform, Data
from ..utils import _raise, _validate_rng, _flatten_axis, _from_flat_sub_array, _to_flat_sub_array, pad_to_shape

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


def iter_fliprot90(img, axis = None, copy = True, return_mapping = False):
    """generates all fliprot90 versions of img

    img: np.ndarray
       The image to fliprot
    axis: tuple
       Axis over which to flip and rotate
    copy: boolean
       If true, create copies of the img
    return_mapping: boolean
       If true, return triples (y, f, f_inv) such that y = f(img) and img = f_inv(y)
    """

    # list of all permutations
    axis = _flatten_axis(img.ndim, axis)

    perms = tuple(subgroup_permutations(img.ndim, axis))
    # list of all flips
    flips = tuple(subgroup_flips(img.ndim, axis))

    perms_inv, flips_inv = perms[::-1], flips[::-1]

    def _inverse_perm(perm):
        a = tuple(range(img.ndim))
        return tuple(perm.index(_a) for _a in a)

    def _perm(perm,x):
        perm = tuple(perm[:x.ndim])
        perm = perm + tuple(range(len(perm),x.ndim))
        return x.transpose(perm)

    def _flip(flip,x):
        flip = tuple(flip[:x.ndim])
        flip = flip + tuple(False for _ in range(len(perm),x.ndim))
        for axis, f in enumerate(flip):
            x = np.flip(x, axis) if f else x
        return x

    def _apply(x, perm,flip,inverse=False):
        if inverse:
            perm = _inverse_perm(perm)
            return _perm(perm,_flip(flip,x))
        else:
            return _flip(flip,_perm(perm,x))

    for perm,flip in itertools.product(perms, flips):
        def get_map(perm,flip,inverse=False):
            def iter_map(x):
                return _apply(x,perm,flip,inverse=inverse)
            return iter_map
        f, f_inv = get_map(perm,flip), get_map(perm,flip,inverse = True)
        augmented = f(img).copy() if copy else f(img)
        if return_mapping:
            yield augmented, f, f_inv
        else:
            yield augmented



def random_rotation_matrix(ndim=2, rng=None):
    """
    adapted from pg 11 of

    Mezzadri, Francesco.
    "How to generate random matrices from the classical compact groups."
    arXiv preprint math-ph/0609050 (2006).

    """
    rng = _validate_rng(rng)
    z = rng.randn(ndim, ndim)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    q = np.multiply(q, ph, q)
    # enforce parity
    q *= np.linalg.det(q)
    return q



def transform_scale(img, rng=None, axis=None,  amount=(1,2), order=1, mode = "constant", use_gpu=False):
    """
    scale tranformation
    :param img, ndarray:
        the nD image to deform
    :param rng:
        the random number generator to be used
    :param axis, tuple or callable:
        the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
    :param amount, pair of float, or tuple of pairs  of same length as axis, or callable:
        the maximal scale amount (scale_min, scale_max) per axis.
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
    img = np.asanyarray(img)

    axis = _flatten_axis(img.ndim, axis)

    if np.isscalar(amount):
        amount = (amount,amount) * len(axis)

    if np.isscalar(amount[0]):
        amount = (amount,) * len(axis)

    amount = np.asanyarray(amount)

    if not img.ndim >= len(axis):
        raise ValueError("dimension of image (%s) < length of axis (%s)" % (img.ndim, len(axis)))

    if not len(axis) == len(amount):
        raise ValueError("length of axis (%s) != length of amount (%s)" % (len(axis), len(amount)))

    rng = _validate_rng(rng)

    if len(axis) < img.ndim:
        # flatten all axis that are not affected
        img_flattened = _to_flat_sub_array(img, axis)
        # state = rng.get_state()

        def _func(x, rng):
            # rng.set_state(state)
            return transform_scale(x, rng=rng,
                                   axis=None, amount=amount, order=order,
                                   mode=mode,
                                   use_gpu = use_gpu)

        # copy rng, to be thread-safe
        rng_flattened = tuple(deepcopy(rng) for _ in img_flattened)

        res_flattened = np.stack(tuple(map(_func, img_flattened, rng_flattened)))

        # ensure that rng was stepped once
        dummy = rng.uniform()
        
        return _from_flat_sub_array(res_flattened, axis, img.shape)

    else:

        scale = tuple(rng.uniform(lower, upper) for lower, upper in amount)

        if use_gpu:
            if not img.ndim==3:
                raise ValueError('use_gpu=True only supported for img.ndim==3')
            from gputools import scale as zoom_gputools            
            # print("scaling by %s via gputools"%str(scale))
            inter = {
                0: "nearest",
                1:"linear"}
            res = pad_to_shape(zoom_gputools(img, scale, interpolation= inter[order]), img.shape, mode=mode)
        else:
            # print("scaling by %s via scipy"%str(scale))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                res = pad_to_shape(ndimage.zoom(img, scale, order=order, mode = mode), img.shape,mode=mode)
        return res


def transform_isotropic_scale(img, rng=None, axis=None,  amount=(1,2), order=1, mode = "constant", use_gpu=False):
    """
    isotropic scale transformation
    :param img, ndarray:
        the nD image to deform
    :param rng:
        the random number generator to be used
    :param axis, tuple or callable:
        the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
    :param amount, pair of float
        the maximal scale amount (scale_min, scale_max) 
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

    img = np.asanyarray(img)
    axis = _flatten_axis(img.ndim, axis)
    if np.isscalar(amount):
        amount = (amount,amount) 
    amount = np.asanyarray(amount)

    if not img.ndim >= len(axis):
        raise ValueError("dimension of image (%s) < length of axis (%s)" % (img.ndim, len(axis)))

    if len(amount)!=2:
        raise ValueError("amount should be a tuple of length 2!")

    rng = _validate_rng(rng)

    if len(axis) < img.ndim:
        # flatten all axis that are not affected
        img_flattened = _to_flat_sub_array(img, axis)
        # state = rng.get_state()

        def _func(x, rng):
            # rng.set_state(state)
            return transform_isotropic_scale(x, rng=rng,
                                   axis=None, amount=amount, order=order,
                                   mode=mode,
                                   use_gpu = use_gpu)

        # copy rng, to be thread-safe
        rng_flattened = tuple(deepcopy(rng) for _ in img_flattened)

        res_flattened = np.stack(tuple(map(_func, img_flattened, rng_flattened)))

        # ensure that rng was stepped once
        dummy = rng.uniform()
        
        return _from_flat_sub_array(res_flattened, axis, img.shape)

    else:

        scale = rng.uniform(*amount)
        scale = (scale,)*len(axis)

        if use_gpu:
            if not img.ndim==3:
                raise ValueError('use_gpu=True only supported for img.ndim==3')
            from gputools import scale as zoom_gputools            
            # print("scaling by %s via gputools"%str(scale))
            inter = {
                0: "nearest",
                1:"linear"}
            res = pad_to_shape(zoom_gputools(img, scale, interpolation= inter[order]), img.shape, mode=mode)
        else:
            # print("scaling by %s via scipy"%str(scale))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                res = pad_to_shape(ndimage.zoom(img, scale, order=order, mode = mode), img.shape,mode=mode)
        return res



class Flip(BaseTransform):
    """
    flip augmentation
    """

    def __init__(self, axis=None):
        """
        :param axis, tuple:
            the axis along which to flip and rotate
        """
        super().__init__() 
        self.axis=axis


    def flip_perm(self, ndim:int , rng=None):
        """
        random augmentation of an array around axis
        """
        rng = _validate_rng(rng)
        # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
        axis = _flatten_axis(ndim, self.axis)
        # list of all flips
        flips = tuple(subgroup_flips(ndim, axis))
        # random flip
        rand_flip_ind = rng.randint(len(flips))
        rand_flip = flips[rand_flip_ind]

        return rand_flip, tuple(range(ndim))

    def transform_image(self, img, rng):
        rand_flip, rand_perm = self.flip_perm(img.ndim, rng)
        # first random permute
        augmented = img.transpose(rand_perm)
        # then random flip
        for axis, f in enumerate(rand_flip):
            if f:
                augmented = np.flip(augmented, axis)
        return augmented

    def transform_points(self, points, shape, rng):
        rand_flip, rand_perm = self.flip_perm(points.shape[-1], rng)
        # first random permute
        augmented = points[:,np.array(rand_perm)]
        # then random flip
        for axis, f in enumerate(rand_flip):
            if f:
                augmented[:,axis] = shape[axis] - augmented[:,axis] - 1
        return augmented


class FlipRot90(Flip):
    """
    flip and 90 degree rotation augmentation
    """

    def flip_perm(self, ndim:int , rng=None):
        """
        random augmentation of an array around axis
        """
        rng = _validate_rng(rng)
        # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
        axis = _flatten_axis(ndim, self.axis)
        # list of all permutations
        perms = tuple(subgroup_permutations(ndim, axis))
        # list of all flips
        flips = tuple(subgroup_flips(ndim, axis))
        # random permutation and flip
        rand_perm_ind = rng.randint(len(perms))
        rand_flip_ind = rng.randint(len(flips))
        rand_perm = perms[rand_perm_ind]
        rand_flip = flips[rand_flip_ind]
        return rand_flip, rand_perm 

class Rotate(BaseTransform):
    """
    random rotations 
    """
    def __init__(self, axis=None, order=1, mode = "reflect", offset=None, use_gpu=False, workers = 1):
        """
        :param axis, tuple:
            the axis along which to flip and rotate
        """
        super().__init__() 
        self.axis=axis,
        self.order = order,
        self.mode = mode,
        self.workers=workers
        self.offset=offset
        self.use_gpu=use_gpu
    
    def random_mat_offset(self, shape, rng):
        rng = _validate_rng(rng)
        ndim = len(shape)
        # flatten the axis, e.g. (-2,-1) -> (2,3) for the different array shapes
        axis = _flatten_axis(ndim, self.axis)
        if self.offset is None:
            offset = tuple(s // 2 for s in np.array(shape)[np.array(axis)])
        else: 
            offset = self.offset

        M_rot = random_rotation_matrix(len(axis), rng)
        M = np.identity(ndim)
        M[np.ix_(np.array(axis), np.array(axis))] = M_rot
        # as scipy.ndimage applies the offset *after* the affine matrix...
        offset -= np.dot(M, offset)

        return M, offset


    def transform_rotation(img, rng=None, axis=None, offset=None, mode="constant", order=1, use_gpu=False, workers = 1):
        """
        random rotation around axis
        """


        if len(axis) < img.ndim:
            # flatten all axis that are not affected
            img_flattened = _to_flat_sub_array(img, axis)
            def _func(x, rng):
                # rng.set_state(state)
                return transform_rotation(x, rng=rng,
                                        axis=None, offset = offset, order=order,
                                        mode=mode,
                                        workers = 1)

            # copy rng, to be thread-safe
            rng_flattened = tuple(deepcopy(rng) for _ in img_flattened)

            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    res_flattened = np.stack(tuple(executor.map(_func, img_flattened, rng_flattened)))
            else:
                res_flattened = np.stack(tuple(map(_func, img_flattened, rng_flattened)))

            return _from_flat_sub_array(res_flattened, axis, img.shape)
        else:

            if use_gpu:
                if not img.ndim==3 and mode=='constant':
                    raise ValueError('use_gpu=True only supported for img.ndim==3 and mode=="constant"')
                from gputools.transforms import affine
                inter = {0: "nearest",1:"linear"}
                res = affine(img, M, interpolation=inter[order])
            else:
                res = ndimage.affine_transform(img, M, offset=offset, order=order, mode=mode)
            return res


class Scale(BaseTransform):
    """
    scale augmentation
    """

    def __init__(self, axis=None, amount=(.5,2), order=1, mode="constant", use_gpu =False):
        """
        :param axis, tuple:
            the axis along which to flip and rotate
        """
        super().__init__(
            default_kwargs=dict(
                axis=axis,
                amount = amount,
                order=order,
                mode=mode,
                use_gpu  = use_gpu
            ),
            transform_func_array=transform_scale
        )



class IsotropicScale(BaseTransform):
    """
    scale augmentation
    """

    def __init__(self, axis=None, amount=(.5,2), order=1, mode="constant", use_gpu =False):
        """
        :param axis, tuple:
            the axis along which to flip and rotate
        """
        super().__init__(
            default_kwargs=dict(
                axis=axis,
                amount = amount,
                order=order,
                mode=mode,
                use_gpu  = use_gpu
            ),
            transform_func_array=transform_isotropic_scale
        )

        
