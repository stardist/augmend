from multiprocessing.sharedctypes import Value
import numpy as np
from .base import BaseTransform
from ..utils import _validate_rng, _flatten_axis
from typing import List, Sequence, Union

def random_crop(x, rng, size, axis):
    rng = _validate_rng(rng)
    axis = _flatten_axis(x.ndim, axis)
    if not len(axis) == len(size):
        raise ValueError(f"Length of axis ({len(axis)}) and size ({len(size)}) should match!")

    full_size = list(x.shape)
    for a, s in zip(axis, size):
        full_size[a] = s
    if not all(s >= w for s, w in zip(x.shape, full_size)):
        raise ValueError(f"Input shape {x.shape} cannot be smaller than crop size {size}!")
    starts = tuple(None if i not in axis else rng.randint(0, s-w+1) for i, (s, w) in enumerate(zip(x.shape, full_size)))
    ends = tuple(None if i not in axis else start+w for i, (w, start) in enumerate(zip(full_size, starts)))
    slices = tuple(slice(start, end) for start, end in zip(starts, ends))
    return x[slices]

class RandomCrop(BaseTransform):
    """
    apply random crop
    """

    def __init__(self, size: Sequence[int]=(200,200), axis: Union[int, List[int]]=None):
        """
        all dimensions that are given by axis will be cropped to a dimension of given size
        axis = None  -> all 
        axis = (0,1) -> only first and second axis will be cropped
        """
        super().__init__(
            default_kwargs=dict(size=size, axis=axis),
            transform_func=random_crop)
