from dataclasses import dataclass
from typing import Union
import numpy as np
from functools import reduce
from ..utils import _raise, _normalized_weights, _all_of_type, map_single_func_tree, zip_trees


@dataclass(frozen=True)
class Data(object):
    array: np.ndarray = None
    points: np.ndarray = None
    shape: tuple = None
    def __post_init__(self):
        if self.array is not None:
            # if self.shape is None: 
            #     # https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
            #     super().__setattr__('shape', self.array.shape)
            if self.array.shape != self.array.shape:
                raise ValueError(f'Shape of array and given shape should be the same!')

        if self.points is not None:
            if not self.points.ndim==2:
                raise ValueError(f'Points should be of shape (n,d), e.g. (101,2)!')

    @staticmethod
    def from_array(self, x: np.ndarray):
        return Data(array=x)



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

    def __init__(self, default_kwargs, transform_func):
        self._default_kwargs = default_kwargs
        self._transform_func = transform_func

    def __call__(self, x: Union[np.ndarray, Data], rng=None, **kwargs):
        if isinstance(x, np.ndarray):
            x = Data.from_array(x)
        elif not isinstance(x, Data):
            print(isinstance(x, Data))
            raise ValueError('Input to a transform should be either a ndarray or a augmend.Data object!')

        kwargs = {**self._default_kwargs, **kwargs}
        return self._transform_func(x,
                                    rng=rng,
                                    **kwargs)

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
        # kwargs_str = '\n'.join(" = ".join(map(str, item)) for item in self._default_kwargs.items())
        # return "%s\n\ndefault arguments:\n%s" % (self.__class__.__name__, kwargs_str)


class ConcatenateTransform(BaseTransform):
    def __init__(self, transforms):
        self.transforms = tuple(transforms)
        super().__init__(
            default_kwargs=dict(),
            transform_func=lambda x, rng: reduce(lambda x, f: f(x, rng), self.transforms, x)
        )

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, self.transforms)


class ChoiceTransform(BaseTransform):
    def __init__(self, transforms, weights=None):
        self.transforms = tuple(transforms)
        self.weights = _normalized_weights(weights, len(transforms))
        super().__init__(
            default_kwargs=dict(),
            transform_func=(
                lambda x, rng: self.transforms[rng.choice(len(self.transforms), p=self.weights)](x, rng)
            )
        )

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__, self.transforms)
        # return "%s%s, w=%s"%(self.__class__.__name__,  self.transforms, self.weights)


class Identity(BaseTransform):
    """
    Do nothing
    """

    def __init__(self):
        super().__init__(
            default_kwargs=dict(),
            transform_func=lambda x, rng: x
        )



class Lambda(BaseTransform):
    def __init__(self, func=lambda x, rng: x):
        super().__init__(
            default_kwargs=dict(),
            transform_func=func
        )
        
