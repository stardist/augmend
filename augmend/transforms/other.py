import numpy as np
from scipy.ndimage.filters import gaussian_filter 
from .base import BaseTransform
from ..utils import _validate_rng, _flatten_axis

class CutOut(BaseTransform):
    """
    Cut parts of the image
    """
    def __init__(self, width=16):
        super().__init__(
            default_kwargs=dict(width=width),
            transform_func=self._cutout)

    def _cutout(self, x, rng, width=16):
        rng = _validate_rng(rng)
        if np.isscalar(width):
            width = (width,) * x.ndim
        assert all(tuple(w <= s for w, s in zip(width, x.shape)))
        x0 = tuple(rng.randint(0, s - w - 1) for w, s in zip(width, x.shape))
        ss = tuple(slice(_x0, _x0 + w) for w, _x0 in zip(width, x0))
        y = x.copy()
        y[ss] = 0
        return y


def blur(img, rng, amount = ((1,5),(2,4)), mode = "reflect", axis= None):

    img = np.asanyarray(img)

    axis = _flatten_axis(img.ndim, axis)

    rng = _validate_rng(rng)

    if np.isscalar(amount):
        amount = (amount,amount) * len(axis)
    if np.isscalar(amount[0]):
        amount = (amount,) * len(axis)
    amount = np.asanyarray(amount)
    
    sigma = tuple(rng.uniform(lower, upper) for lower, upper in amount)
    y = gaussian_filter(img,sigma, mode=mode)
    return y
         
    
class GaussianBlur(BaseTransform):
    """
    Cut parts of the image
    """
    def __init__(self, amount= (1,4), mode=  "reflect", axis = None):
        super().__init__(
            default_kwargs=dict(amount=amount, mode=mode, axis=axis),
            transform_func=blur)

    
