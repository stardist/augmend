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
                                                                                                                x.shape)
        )