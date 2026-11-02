"""

mweigert@mpi-cbg.de
"""
import numpy as np
from abc import ABCMeta, abstractmethod

from .utils import _raise, _is_leaf_node, _wrap_leaf_node, _normalized_weights, _get_global_rng
from .transforms import TransformTree


class BaseAugmend(metaclass=ABCMeta):
    def __init__(self, *transforms, rng=None):
        """
        :param rng, np.random.RandomState or np.random.Generator:
            if None, numpy's global RandomState of the calling process is used at call time
        """
        self._rng = rng
        self._transforms = []

        for t in transforms:
            self.add(t)

    def add(self, transform):
        """
        :param transform:
            The transformation object to be applied
        """
        try:
            transform = TransformTree(transform)
        except ValueError:
            callable(transform) or _raise(ValueError("transform needs to be callable with signature (data, rng)"))
        self._transforms.append(transform)

    @property
    def _rng(self):
        # resolved on access, so that a pipeline sent to another process
        # (e.g. a spawned DataLoader worker) uses that process' global RNG
        return _get_global_rng() if self._rng_init is None else self._rng_init

    @_rng.setter
    def _rng(self, rng):
        self._rng_init = None if rng is np.random else rng

    @abstractmethod
    def _call(self, x, rng):
        pass

    def __call__(self, x, rng=None):
        """
        :param rng, np.random.RandomState or np.random.Generator:
            used for this call only (default: the pipeline's rng)
        """
        if rng is None:
            rng = self._rng
        # wrap
        wrapped = _is_leaf_node(x)
        x = _wrap_leaf_node(x)
        # actual computation
        x = self._call(x, rng)
        # unwrap
        return x[0] if (wrapped and len(x)==1 and _is_leaf_node(x[0])) else x

    def __repr__(self):
        return f"{self.__class__.__name__}{self._transforms}"
        # return "%s%s, w=%s"%(self.__class__.__name__,  self.transforms, self.weights)

    def __len__(self):
        return len(self._transforms)

    # def __getitem__(self, *args):
    #     return self._transforms.__getitem__(*args)



# TODO: rename to "Pipeline"?
class Augmend(BaseAugmend):
    """
    Main augmentation pipeline object

    Example:
    ========


    """

    def __init__(self, *transforms, probabilities=None, rng=None):
        """
        :param rng, random_number_generator:
        """
        super().__init__(rng=rng)
        self._probabilities = []
        if probabilities is None:
            probabilities = [1]*len(transforms)
        len(probabilities)==len(transforms) or _raise(ValueError())
        for t,p in zip(transforms,probabilities):
            self.add(t,p)

    def __repr__(self):
        return "\n".join(
            f"{i + 1} (p={p:.2f}): {t}"
            for i, (p, t) in enumerate(zip(self._probabilities, self._transforms)))

    def add(self, transform, probability=1.0):
        """
        :param transform:
            The transformation object to be applied
        :param probability, float:
            the probability which which to activate the augmentation (0<= p<= 1)
        """
        (np.isscalar(probability) and 0 <= probability <= 1) or _raise(ValueError())
        super().add(transform)
        self._probabilities.append(probability)

    def _call(self, x, rng):
        """apply augmentation chain to arrays/images
        """
        for trans, prob in zip(self._transforms, self._probabilities):
            if rng.uniform(0,1) <= prob:
                x = trans(x, rng=rng)
        return x


    def flow(self, iterable):
        """apply augmentation chain to an iterator of arrays/batches
        """
        return map(self, iterable)


    

class Choice(BaseAugmend):
    def __init__(self, *transforms, weights=None):
        super().__init__(*transforms)
        self._weights = _normalized_weights(weights,len(transforms))

    def _call(self, x, rng):
        trans = self._transforms[rng.choice(len(self),p=self._weights)]
        return trans(x, rng=rng)
