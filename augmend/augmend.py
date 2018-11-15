"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from functools import reduce
from .utils import map_single_func_tree, zip_trees


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
        self._transforms.append((transform, probability))

    def __call__(self, x):
        """apply augmentation chain to arrays/images
        """
        for trans, prob in self._transforms:
            if self._rng.uniform(0, 1) <= prob:
                rand_state = self._rng.get_state()

                def _apply(leaf):
                    _trans, _x = leaf
                    self._rng.set_state(rand_state)
                    return _trans(_x, rng=self._rng)

                x = map_single_func_tree(_apply, zip_trees(trans, x))
        return x


    def flow(self, iterable):
        """apply augmentation chain to an iterator of arrays/batches
        """
        return map(self, iterable)
