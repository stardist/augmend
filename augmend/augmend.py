"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from six import add_metaclass
from abc import ABCMeta, abstractmethod

from .utils import _raise, _is_leaf_node, _wrap_leaf_node, _normalized_weights, _get_global_rng
from .transforms import TransformTree


@add_metaclass(ABCMeta)
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

    @abstractmethod
    def _call(self, x):
        pass

    def __call__(self, x, rng=None):
        if rng is not None:
            self._rng = rng
        # wrap
        wrapped = _is_leaf_node(x)
        x = _wrap_leaf_node(x)
        # actual computation
        x = self._call(x)
        # unwrap
        return x[0] if (wrapped and len(x)==1 and _is_leaf_node(x[0])) else x

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
        (np.isscalar(probability) and 0 <= probability <= 1) or _raise(ValueError())
        super().add(transform)
        self._probabilities.append(probability)

    def _call(self, x):
        """apply augmentation chain to arrays/images
        """
        for trans, prob in zip(self._transforms, self._probabilities):
            if self._rng.uniform(0,1) <= prob:
                x = trans(x, rng=self._rng)
        return x


    def flow(self, iterable):
        """apply augmentation chain to an iterator of arrays/batches
        """
        return map(self, iterable)


    def tf_map(self, *args):
        """provides a function to be used with tf.data pipelines 

        # Example: 

        aug = Augmend()
        aug.add([Elastic(axis=(0, 1), amount=5, order=1),
                Elastic(axis=(0, 1), amount=5, order=0)])
    
        dataset = tf.data.Dataset.from_tensor_slices((x,y))

        gen = dataset.map(aug.tf_map, num_parallel_calls=8).batch(16)
        """
        import tensorflow as tf 
        def _func(*args):
            return self(args)
        return tf.numpy_function(_func,list(args),tuple(a.dtype for a in args))
    

class Choice(BaseAugmend):
    def __init__(self, *transforms, weights=None):
        super().__init__(*transforms)
        self._weights = _normalized_weights(weights,len(transforms))

    def _call(self, x):
        trans = self._transforms[self._rng.choice(len(self),p=self._weights)]
        return trans(x, rng=self._rng)
