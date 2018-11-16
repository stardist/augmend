"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from functools import reduce
from .utils import map_single_func_tree, zip_trees, _is_leaf_node, _wrap_leaf_node
from .transforms import BaseTransform


class Augmend(object):
    """
    Main augmentation pipeline object

    Example:
    ========


    """

    def __init__(self, rng=None):
        """
        :param rng, random_number_generator:
        """
        if rng is None:
            rng = np.random
        self._rng = rng
        self._transforms = []

    def __repr__(self):
        return "\n".join(
            map(lambda t: "\n-------------\nprob = %s\n%s\n-------------\n " % (t[1], t[0]), self._transforms))

    def add(self, transform, probability=1.):
        """
        :param transform:
            The transformation object to be applied
        :param probability, float:
            the probability which which to activate the augmentation (0<= p<= 1)
        """
        # don't wrap generic functions
        if isinstance(transform,BaseTransform):
            transform = _wrap_leaf_node(transform)
        self._transforms.append((transform, probability))

    def __call__(self, x):
        """apply augmentation chain to arrays/images
        """
        wrapped = _is_leaf_node(x)
        x = _wrap_leaf_node(x)

        for trans, prob in self._transforms:
            if self._rng.uniform(0, 1) <= prob:

                if isinstance(trans,Branch):
                    trans = trans(rng=self._rng)

                rand_state = self._rng.get_state()

                def _apply(leaf):
                    _trans, _x = leaf
                    self._rng.set_state(rand_state)
                    return _trans(_x, rng=self._rng)

                if callable(trans):
                    # TODO: pass rng to callable trans? especially if callable is another Augment object?
                    x = trans(x)
                else:
                    x = map_single_func_tree(_apply, zip_trees(trans, x))
        return x[0] if wrapped else x


    def flow(self, iterable):
        """apply augmentation chain to an iterator of arrays/batches
        """
        return map(self, iterable)


# TODO: basically same functionality as Choice transform, needs refactoring/rethinking
class Branch(object):
    def __init__(self, *transforms, weights=None):
        # don't wrap generic functions
        self.transforms = tuple(map(
            lambda t: _wrap_leaf_node(t) if isinstance(t,BaseTransform) else t,
            transforms
        ))
        if weights is None:
            weights = [1]*len(transforms)
        assert len(weights)==len(transforms)
        weights = np.asanyarray(weights)
        weights = weights / np.sum(weights)
        self.weights = weights

    def __call__(self, rng=np.random):
        return self.transforms[rng.choice(len(self.transforms),p=self.weights)]

    # def __len__(self):
    #     return len(self.transforms)

    # def __getitem__(self, *args):
    #     return self.transforms.__getitem__(*args)

    def __repr__(self):
        return "%s%s"%(self.__class__.__name__,  self.transforms)
        # return "%s%s, w=%s"%(self.__class__.__name__,  self.transforms, self.weights)
