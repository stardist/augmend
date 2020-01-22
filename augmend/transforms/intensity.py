import numpy as np
from .base import BaseTransform
from ..utils import _validate_rng, _flatten_axis


def additive_noise(x,rng, sigma):
    rng = _validate_rng(rng)

    if callable(sigma):
        return x + sigma(x, rng)
    else:
        if np.isscalar(sigma):
            sigma = (sigma,sigma) 
        assert len(sigma)==2
        noise = rng.uniform(*sigma)
        return x + noise*rng.normal(0, 1,x.shape) 

def intensity_scale_shift(x, rng, scale_range, shift_range, axis):
    rng = _validate_rng(rng)
    axis = _flatten_axis(x.ndim, axis)
    size = tuple(1 if i in axis else s for i,s in enumerate(x.shape))
    return x*rng.uniform(*scale_range, size = size)+rng.uniform(*shift_range, size = size)


                    
class AdditiveNoise(BaseTransform):
    """
    Add gaussian noise
    """
    def __init__(self, sigma=.1):
        super().__init__(
            default_kwargs=dict(sigma=sigma),
            transform_func=additive_noise)



class IntensityScaleShift(BaseTransform):
    """
    apply affine intensity shift
    """

    def __init__(self, scale_range=(.8,1.2), shift_range = (-.1,.1), axis=None):
        """
        all dimensions that are given by axis will be shifted simultaneously 
        axis = None  -> all 
        axis = (0,1) -> axis 2,3... will be shifted independently 
        """
        super().__init__(
            default_kwargs=dict(scale_range=scale_range, shift_range=shift_range, axis=axis),
            transform_func=intensity_scale_shift)



        
