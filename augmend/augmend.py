"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from .utils import _raise, _is_leaf_node, _wrap_leaf_node, _normalized_weights, _get_global_rng
from .transforms import TransformTree

class BaseAugmend(object):
    def __init__(self, *transforms, rng=None):
        """
        :param rng, random_number_generator:
        """
        if rng is None or rng is np.random:
            rng = _get_global_rng()
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

    def __call__(self, x, rng=None):
        # TODO: do this properly
        raise NotImplementedError()

    def __repr__(self):
        return "%s%s"%(self.__class__.__name__,  self._transforms)
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
            map(lambda t: "%d (p=%.2f): %s" % (1+t[0], t[1], t[2]),
                zip(range(len(self)), self._probabilities, self._transforms)))

    def add(self, transform, probability=1.0):
        """
        :param transform:
            The transformation object to be applied
        :param probability, float:
            the probability which which to activate the augmentation (0<= p<= 1)
        """
        0 <= probability <= 1 or _raise(ValueError())
        super().add(transform)
        self._probabilities.append(probability)

    def __call__(self, x, rng=None):
        """apply augmentation chain to arrays/images
        """
        if rng is not None:
            self._rng = rng

        wrapped = _is_leaf_node(x)
        x = _wrap_leaf_node(x)

        for trans, prob in zip(self._transforms, self._probabilities):
            if self._rng.uniform(0,1) <= prob:
                x = trans(x, rng=self._rng)

        return x[0] if (wrapped and len(x)==1 and _is_leaf_node(x[0])) else x


    def flow(self, iterable):
        """apply augmentation chain to an iterator of arrays/batches
        """
        return map(self, iterable)



class Choice(BaseAugmend):
    def __init__(self, *transforms, weights=None):
        super().__init__(*transforms)
        self._weights = _normalized_weights(weights,len(transforms))

    def __call__(self, x, rng=None):
        if rng is not None:
            self._rng = rng
        trans = self._transforms[self._rng.choice(len(self),p=self._weights)]
        return trans(x, rng=self._rng)
