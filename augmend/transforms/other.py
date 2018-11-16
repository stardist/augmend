import numpy as np
from .base import BaseTransform

class CutOut(BaseTransform):
    """
    Cut parts of the image
    """

    def __init__(self, width=16):
        super().__init__(
            default_kwargs=dict(width=width),
            transform_func=self._cutout)

    def _cutout(self, x, rng, width=16):
        if np.isscalar(width):
            width = (width,) * x.ndim
        assert all(tuple(w <= s for w, s in zip(width, x.shape)))
        x0 = tuple(rng.randint(0, s - w - 1) for w, s in zip(width, x.shape))
        ss = tuple(slice(_x0, _x0 + w) for w, _x0 in zip(width, x0))
        y = x.copy()
        y[ss] = 0
        return y

