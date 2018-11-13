"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from functools import reduce
from .utils import map_recursive
from .transforms import transform_elastic, transform_flip_rot


class Augmend(object):

    def __init__(self, random_generator = None):
        if random_generator is None:
            random_generator = np.random
        self._random_generator = random_generator
        self._augs = []


    def add(self, augmend_obj):
        self._augs.append(augmend_obj)


    def apply(self, img):
        """apply augmentation chain to a single array/image
        """
        if len(self._augs) == 0:
            raise ValueError("empty augmentation list (please add some before using!")
        return reduce(lambda x, f: f(x, self._random_generator), self._augs, img)
        
    def __call__(self, iterable):
        """apply augmentation chain to an iterator of arrays/batches
        """
        rand_state = self._random_generator.get_state()
        def _func(img):
            self._random_generator.set_state(rand_state)
            return self.apply(img)
        #loop over iterables and augment all not list/tuple items
        for it in iterable:
            yield map_recursive(_func,it)



class BaseAugmenter(object):
    """
    base class for an augmentation action 
    """
    # def __init__(self, *args, **kwargs):
    #     pass

    def __call__(self, img, random_generator = np.random):
        print("random: ", np.random.randint(0,100))
        return img


class ElasticAugmenter(BaseAugmenter):
    """
    elastic augmentation  
    """
    def __init__(self, p=1.,  grid=(5, 5), amount=5, order=1):
        self.probability = p
        self.grid = grid
        self.amount = amount
        self.order=order

    def __call__(self, img, random_generator = np.random):
        if random_generator.uniform(0,1)<=self.probability:
            return transform_elastic(img,
                                     grid=self.grid,
                                     amount=self.amount,
                                     order=self.order, random_generator=random_generator)
        return img



class FlipRotAugmenter(BaseAugmenter):
    """
    elastic augmentation  
    """
    def __init__(self, p=1.,  axis = (1,2)):
        self.probability = p
        self.axis = axis

    def __call__(self, img, random_generator = np.random):
        if random_generator.uniform(0,1)<=self.probability:
            return transform_flip_rot(img,
                                     axis = self.axis,
                                     random_generator=random_generator)
        return img





