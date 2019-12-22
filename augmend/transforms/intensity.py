import numpy as np
from .base import BaseTransform
from ..utils import _validate_rng


def additive_noise(x,rng, sigma):
    rng = _validate_rng(rng)


    if callable(sigma):
        return x + sigma(x, rng)
    else:
        if np.isscalar(sigma):
            sigma = (sigma,sigma) * len(axis)
        assert len(sigma)==2
        noise = rng.uniform(*sigma)
        return x + noise * rng.normal(0, 1,x.shape) 

def intensity_scale_shift(x, rng, scale_range, shift_range):
    rng = _validate_rng(rng)
    return x*rng.uniform(*scale_range)+rng.uniform(*shift_range)


                    
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

    def __init__(self, scale_range=(1.,1.), shift_range = (0,0)):
        super().__init__(
            default_kwargs=dict(scale_range = scale_range, shift_range = shift_range),
            transform_func=intensity_scale_shift)



        
