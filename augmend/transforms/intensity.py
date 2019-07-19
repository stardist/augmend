import numpy as np
from .base import BaseTransform


class AdditiveNoise(BaseTransform):
    """
    Add gaussian noise
    """

    def __init__(self, sigma=.1):
        super().__init__(
            default_kwargs=dict(sigma=sigma),
            transform_func=lambda x, rng, sigma: x + (sigma(x, rng) if callable(sigma) else sigma) * rng.normal(0, 1,
                                                                                                                x.shape))



class IntensityScaleShift(BaseTransform):
    """
    apply affine intensity shift
    """

    def __init__(self, scale_range=(1.,1.), shift_range = (0,0)):
        super().__init__(
            default_kwargs=dict(scale_range = scale_range, shift_range = shift_range),
            transform_func=lambda x, rng, scale_range, shift_range:
            rng.uniform(*scale_range)*x+rng.uniform(*shift_range))



        
