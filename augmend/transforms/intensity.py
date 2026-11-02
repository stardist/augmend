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
        return x + noise*rng.normal(0, 1,x.shape).astype(np.float32)

def intensity_scale_shift(x, rng, scale, shift, axis, invert=False):
    rng = _validate_rng(rng)
    axis = _flatten_axis(x.ndim, axis)
    size = tuple(1 if i in axis else s for i,s in enumerate(x.shape))
    s = rng.uniform(*scale, size = size).astype(np.float32)
    t = rng.uniform(*shift, size = size).astype(np.float32)
    if invert:
        if not np.issubdtype(np.asarray(x).dtype, np.floating):
            raise ValueError(f"invert=True expects float images in [0, 1], got dtype {np.asarray(x).dtype}")
        return 1 - ((1 - x) * s + t)
    return x * s + t

    
    
    
                    
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

    def __init__(self, scale=(.8,1.2), shift = (-.1,.1), axis=None, invert=False):
        """
        all dimensions that are given by axis will be shifted simultaneously 
        axis = None  -> all 
        axis = (0,1) -> axis 2,3... will be shifted independently 
        invert = True -> scale and shift 1 - x instead of x, for float images in [0, 1]
                         with bright background (e.g. brightfield, H&E), which then stays bright
        """
        super().__init__(
            default_kwargs=dict(scale=scale, shift=shift, axis=axis, invert=invert),
            transform_func=intensity_scale_shift)



