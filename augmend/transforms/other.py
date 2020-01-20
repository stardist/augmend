import numpy as np
from scipy.ndimage.filters import gaussian_filter 
from .base import BaseTransform
from ..utils import _validate_rng, _flatten_axis

class CutOut(BaseTransform):
    """
    Cut parts of the image
    """
    def __init__(self, width=(10,16), n = (2,6), val = 0, axis = None):
        super().__init__(
            default_kwargs=dict(width=width, n=n, val=val, axis=axis),
            transform_func=self._cutout)

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
        for _ in range(np.random.randint(*n)):
            width_rand = tuple(np.random.randint(*w) for w in width)
            xs = tuple(rng.randint(0, s - w - 1) for w, s in zip(width_rand, img.shape))
            ss = list(slice(None) for _ in range(img.ndim))
            for ax,x,w in zip(axis,xs,width_rand):
                ss[ax] = slice(x, x + w)
            y[tuple(ss)] = np.random.uniform(*val)
            
        return y


def blur(img, rng, amount = ((1,5),(2,4)), mode = "reflect", axis= None):
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
    y = gaussian_filter(img, sigmas_all, mode=mode)
    return y
         
    
class GaussianBlur(BaseTransform):
    """
    Cut parts of the image
    """
    def __init__(self, amount= (1,4), mode=  "reflect", axis = None):
        super().__init__(
            default_kwargs=dict(amount=amount, mode=mode, axis=axis),
            transform_func=blur)

    
