from dataclasses import dataclass
from typing import Union
import numpy as np
from functools import reduce
from ..utils import _raise, _normalized_weights, _all_of_type, map_single_func_tree, zip_trees, _validate_rng


class Data(object):
    def __init__(self, image, points=None) -> None:
        self.image = image
        self.points = points
        if self.points is not None:
            if not self.points.ndim==2:
                raise ValueError(f'Points should be of shape (n,d), e.g. (101,2)!')

class TransformTree(object):
    def __init__(self, tree):
        if isinstance(tree, BaseTransform):
            tree = (tree,)
        _all_of_type(tree, BaseTransform) or _raise(ValueError("not a tree of transforms"))
        self.tree = tree

    def __call__(self, x: Union[np.ndarray, Data], rng=np.random):
        # get a deterministic yet random new initial state
        rand_state = np.random.RandomState(rng.randint(0,2**31-1)).get_state()
        
        def _apply(leaf):
            trans, _x = leaf
            # make sure that every transform has its own RandomState
            rng = np.random.RandomState()
            rng.set_state(rand_state)
            return trans(_x, rng=rng)

        return map_single_func_tree(_apply, zip_trees(self.tree, x))

    def __repr__(self):
        # return "%s(%s)"%(self.__class__.__name__, str(self.tree))
        return "%s" % (str(self.tree[0] if len(self.tree) == 1 else self.tree))



class BaseTransform(object):
    """
    base class for an augmentation action
    """

    def __init__(self, default_kwargs=None):
        if default_kwargs is None: 
            default_kwargs = dict() 
        self._default_kwargs = default_kwargs
    
    def transform_image(self, img: np.ndarray, rng):
        raise NotImplementedError()

    def transform_points(self, points : np.ndarray, shape: tuple, rng):
        raise NotImplementedError()

    def __call__(self, x: Union[np.ndarray, Data], rng=None, **kwargs):

        rng = _validate_rng(rng) 

        rand_state = np.random.RandomState(rng.randint(0,2**31-1)).get_state()
        rng = np.random.RandomState()

        if isinstance(x, np.ndarray):
            x = Data(image=x)
            rng.set_state(rand_state)
            out = self.transform_image(x,rng=rng,**kwargs)
            return out.image

        else:         
            # elif not isinstance(x, Data):
            #     raise ValueError('Input to a transform should be either a ndarray or a augmend.Data object!')
            rng.set_state(rand_state)
            out_image = self.transform_image(x.image,rng=rng)

            if x.points is not None: 
                rng.set_state(rand_state)
                out_points = self.transform_points(x.points, x.image.shape, rng=rng)
            else: 
                out_points = None 

            return Data(image=out_image, points=out_points) 

    def __add__(self, other):
        return ConcatenateTransform([self, other])

    # TODO: when chaining more than two things together, how do divvy up probabilities?
    def __or__(self, other):
        # TODO: warning if weights not uniform for any of the (potential) Choice transforms self or other
        # print(self, other)
        trans_self = list(self.transforms) if isinstance(self, ChoiceTransform)  else [self]
        trans_other = list(other.transforms) if isinstance(other, ChoiceTransform) else [other]
        return ChoiceTransform(trans_self + trans_other)

    def __repr__(self):
        return self.__class__.__name__


class ConcatenateTransform(BaseTransform):
    def __init__(self, transforms):
        self.transforms = tuple(transforms)

    def __call__(self, x: Data, rng=None):
        return reduce(lambda x, f: f(x, rng), self.transforms, x)
        
    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, self.transforms)


class ChoiceTransform(BaseTransform):
    def __init__(self, transforms, weights=None):
        self.transforms = tuple(transforms)
        self.weights = _normalized_weights(weights, len(transforms))

    def __call__(self, x: Data, rng=None):
        return self.transforms[rng.choice(len(self.transforms), p=self.weights)](x, rng)

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, self.transforms)


class Identity(BaseTransform):
    """
    Do nothing
    """
    def transform_image(self, img: np.ndarray, rng):
        return img

    def transform_points(self, points : np.ndarray, shape: tuple, rng):
        return points

class Lambda(BaseTransform):
    def __init__(self, func_image=(lambda x, rng: x), func_points=(lambda x, shape, rng: x)):
        super().__init__(default_kwargs=dict()) 

        self._transform_func_image=func_image, 
        self._transform_func_points=func_points 

    def transform_image(self, img: np.ndarray, rng):
        return self._transform_func_image(img, rng)

    def transform_points(self, points : np.ndarray, shape: tuple, rng):
        return self._transform_func_points(points, shape, rng)
