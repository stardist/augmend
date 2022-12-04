import numpy as np
import warnings
from scipy.ndimage.filters import gaussian_filter 
from .base import BaseTransform
from ..utils import _validate_rng, _flatten_axis



def blur(img, rng, amount = ((1,5),(2,4)), mode = "reflect", axis= None, use_gpu=False):
    """
    amount: scalar, or pair (min, max) or tuple of pairs ((min_1, max_1), (min_2, max_2),
    """
    img = np.asanyarray(img)
    axis = _flatten_axis(img.ndim, axis)
    # TODO: add proper error handling

    rng = _validate_rng(rng)

    if np.isscalar(amount):
        amount = (amount,amount) * len(axis)
    if np.isscalar(amount[0]) and len(amount)==2:
        amount = (amount,) * len(axis)

    if not (len(amount) == len(axis) and all(len(am)==2 for am in amount)):
        raise ValueError("misformed amount:  {amount}".format(amount=amount))
        
    amount = np.asanyarray(amount)
    assert len(axis) == len(amount)
    
    
    sigmas = tuple(rng.uniform(lower, upper) for lower, upper in amount)

    sigmas_all = [0]*img.ndim
    for ax,sig in zip(axis, sigmas):
        sigmas_all[ax] = sig

    if use_gpu:
        from gputools import blur
        if not mode=='constant':
            warnings.warn(f'mode {mode} not available when using GPU - switching to mode "constant" ')
        sigmas_all = tuple(max(1e-8,s) for s in sigmas_all)
        y = blur(img, sigmas_all)
    else:
        y = gaussian_filter(img, sigmas_all, mode=mode)
    return y



def drop_planes(x, rng, axis, width, n, val):
    rng = _validate_rng(rng)
    axis = _flatten_axis(x.ndim, axis)
    if np.isscalar(val):
        val = (val, val)
    ax = rng.choice(axis)
    
    if x.shape[ax]<width:
        raise ValueError(f'cannot drop {n} planes since shape of input {x.shape} along axis {ax} is too small')

    x2 = x.copy().astype(np.float32)

    ss = [slice(None)]*x.ndim
    for _ in range(n):
        start = rng.randint(0,x.shape[ax]-width+1)
        ss[ax] = slice(start, start+n)
        x2[tuple(ss)] = rng.uniform(*val)
    return x2.astype(x.dtype, copy=False)



def drop_edge_planes(x, rng, axis, width,  val):
    rng = _validate_rng(rng)
    axis = _flatten_axis(x.ndim, axis)
    if np.isscalar(val):
        val = (val, val)
    ax = rng.choice(axis)
    
    if x.shape[ax]<width:
        raise ValueError(f'cannot drop {n} planes since shape of input {x.shape} along axis {ax} is too small')

    x2 = x.copy().astype(np.float32)

    ss = [slice(None)]*x.ndim
    if rng.randint(0,2)==0:
        ss[ax] = slice(0,width)
    else:
        ss[ax] = slice(-width, None)

    x2[tuple(ss)] = rng.uniform(*val)
    return x2.astype(x.dtype, copy=False)




class GaussianBlur(BaseTransform):
    """
    Cut parts of the image
    """
    def __init__(self, amount= (1,4), mode=  "reflect", axis = None, use_gpu=False):
        super().__init__(
            default_kwargs=dict(amount=amount, mode=mode, axis=axis, use_gpu=use_gpu),
            transform_func_array=blur)


class CutOut(BaseTransform):
    """
    Cut parts of the image
    """
    def __init__(self, width=(10,16), n = (2,6), val = 0, axis = None):
        super().__init__(
            default_kwargs=dict(width=width, n=n, val=val, axis=axis),
            transform_func_array=self._cutout)

    def _cutout(self, img, rng, width=(10,16), n=(2,6), val = 0, axis = None):
        rng = _validate_rng(rng)

        axis = _flatten_axis(img.ndim, axis)
        
        if np.isscalar(width):
            width = ((width,width+1),) * len(axis)
        if np.isscalar(width[0]) and len(width)==2:
            width = (width,) * len(axis)
        if not (len(width) == len(axis) and all(len(ws)==2 for ws in width)):
            raise ValueError("misformed amount:  {width}".format(width=width))


        if np.isscalar(n):
            n = (n,n+1)
        if np.isscalar(val):
            val = (val,) * 2
            
        assert all(tuple(max(w) <= s for w, s in zip(width, img.shape)))
        assert len(axis) == len(width)

        y = img.copy()
        for _ in range(rng.randint(*n)):
            width_rand = tuple(rng.randint(*w) for w in width)
            xs = tuple(rng.randint(0, s - w - 1) for w, s in zip(width_rand, img.shape))
            ss = list(slice(None) for _ in range(img.ndim))
            for ax,x,w in zip(axis,xs,width_rand):
                ss[ax] = slice(x, x + w)
            y[tuple(ss)] = rng.uniform(*val)
            
        return y

                    
class DropPlanes(BaseTransform):
    """
    set planes along a random axis to given value (or random value from a given range of values)
    """
    def __init__(self, axis=None, n=1, width=1, val=0):
        """
        Parameters
        ----------
        axis : tuple or None
           possible axis to use for dropping (None if all)
        width : int 
           with of each plane to drop
        n    : int 
           number of planes to drop
        val  : float or (float, float)
           set plane to this value 
              
        """
        super().__init__(
            default_kwargs=dict(axis=axis, width=width, n=n, val=val),
            transform_func_array=drop_planes)

        
class DropEdgePlanes(BaseTransform):
    """
    set edge planes along a random axis to given value (or random value from a given range of values)
    """
    def __init__(self, axis=None, width=1, val=0):
        """
        Parameters
        ----------
        axis : tuple or None
           possible axis to use for dropping (None if all)
        width : int 
           with of each plane to drop
        val  : float or (float, float)
           set plane to this value 
              
        """
        super().__init__(
            default_kwargs=dict(axis=axis, width=width, val=val),
            transform_func_array=drop_edge_planes)


        
