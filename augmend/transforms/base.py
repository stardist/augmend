import numpy as np
from functools import reduce
from ..utils import _raise, _normalized_weights, _all_of_type, _validate_rng, map_single_func_tree, zip_trees



def _identity(x, rng=None):
    return x


class TransformTree:
    def __init__(self, tree):
        if isinstance(tree, BaseTransform):
            tree = (tree,)
        _all_of_type(tree, BaseTransform) or _raise(ValueError("not a tree of transforms"))
        self.tree = tree

    def __call__(self, x, rng=None):
        # every leaf transform gets its own RandomState with the same seed,
        # such that coupled transforms (e.g. for image/label pairs) match
        seed = _validate_rng(rng).randint(0, 2**31 - 1)

        def _apply(leaf):
            trans, _x = leaf
            return trans(_x, rng=np.random.RandomState(seed))

        return map_single_func_tree(_apply, zip_trees(self.tree, x))

    def __repr__(self):
        # return "%s(%s)"%(self.__class__.__name__, str(self.tree))
        return str(self.tree[0] if len(self.tree) == 1 else self.tree)


class BaseTransform:
    """
    base class for an augmentation action
    """

    def __init__(self, default_kwargs, transform_func):
        self._default_kwargs = default_kwargs
        self._transform_func = transform_func

    def __call__(self, x, rng=None, **kwargs):
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
            transform_func=self._concatenate
        )

    def _concatenate(self, x, rng=None):
        return reduce(lambda x, f: f(x, rng), self.transforms, x)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.transforms}"


class ChoiceTransform(BaseTransform):
    def __init__(self, transforms, weights=None):
        self.transforms = tuple(transforms)
        self.weights = _normalized_weights(weights, len(transforms))
        super().__init__(
            default_kwargs=dict(),
            transform_func=self._choose
        )

    def _choose(self, x, rng=None):
        rng = _validate_rng(rng)
        return self.transforms[rng.choice(len(self.transforms), p=self.weights)](x, rng)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.transforms}"
        # return "%s%s, w=%s"%(self.__class__.__name__,  self.transforms, self.weights)


class Identity(BaseTransform):
    """
    Do nothing
    """

    def __init__(self):
        super().__init__(
            default_kwargs=dict(),
            transform_func=_identity
        )



class Lambda(BaseTransform):
    """
    Apply func(x, rng). The pipeline can only be pickled (e.g. for DataLoader workers) if func can.
    """

    def __init__(self, func=None):
        super().__init__(
            default_kwargs=dict(),
            transform_func=_identity if func is None else func
        )
        
