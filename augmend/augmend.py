"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from functools import reduce
from .utils import map_recursive
from .transforms import transform_elastic, transform_flip_rot


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
        self._augs = []

    def add(self, augmend_obj):
        self._augs.append(augmend_obj)

    def apply(self, img):
        """apply augmentation chain to a single array/image
        """
        if len(self._augs) == 0:
            raise ValueError("empty augmentation list (please add some before using!")
        return reduce(lambda x, f: f(x, self._rng), self._augs, img)

    def __call__(self, iterable):
        """apply augmentation chain to an iterator of arrays/batches
        """

        # loop over iterables and augment all not list/tuple items
        for it in iterable:
            rand_state = self._rng.get_state()
            def _func(img):
                self._rng.set_state(rand_state)
                return self.apply(img)

            yield map_recursive(_func, it)


class BaseAugmenter(object):
    """
    base class for an augmentation action 
    """

    # def __init__(self, *args, **kwargs):
    #     pass

    def __call__(self, img, rng=np.random):
        print("random: ", np.random.randint(0, 100))
        return img


class ElasticAugmenter(BaseAugmenter):
    """
    elastic deformation of an n-dimensional image along the given axis 
    
    :param axis, tuple:
        the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
    :param grid, int or tuple of ints of same length as axis:
        the number of gridpoints per axis at which random deformation vectors are attached.  
    :param amount, float or tuple of floats of same length as axis:
        the maximal pixel shift of deformations per axis.
    :param order, int:
        the interpolation order (e.g. set order = 0 for nearest neighbor)
    :param rng:
        the random number generator to be used
    :return ndarray: 
        the deformed img/array
        
    Example:
    ========

    img = np.zeros((128,) * 2, np.float32)

    img[::16] = 128
    img[:,::16] = 128

    aug_elastic = ElasticAugmentor(grid=5, amount=5)
    
    out = aug_elastic(img)
    
    

    """

    def __init__(self, p=1., axis=None, grid=5, amount=5, order=1):
        """
        
        :param p, float:
            the probability which which to activate the augmentation (0<= p<= 1)
        :param axis, tuple:
            the axis along which to deform e.g. axis = (1,2). Set axis = None if all axe should be used
        :param grid, int or tuple of ints of same length as axis:
            the number of gridpoints per axis at which random deformation vectors are attached.  
        :param amount, float or tuple of floats of same length as axis:
            the maximal pixel shift of deformations per axis.
        :param order, int:
            the interpolation order (e.g. set order = 0 for nearest neighbor)
        """

        self.probability = p
        self.grid = grid
        self.amount = amount
        self.order = order
        self.axis = axis

    def __call__(self, img, rng=np.random):
        if rng.uniform(0, 1) <= self.probability:
            return transform_elastic(img,
                                     axis = self.axis,
                                     grid=self.grid,
                                     amount=self.amount,
                                     order=self.order, rng=rng)
        return img


class FlipRotAugmenter(BaseAugmenter):
    """
    flip and rotation augmentation  
    """

    def __init__(self, p=1., axis=None):
        self.probability = p
        self.axis = axis

    def __call__(self, img, rng=np.random):
        if rng.uniform(0, 1) <= self.probability:
            return transform_flip_rot(img,
                                      axis=self.axis,
                                      random_generator=rng)
        return img
