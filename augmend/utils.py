"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from functools import partial

def map_recursive(func, x, iterate_types = (list, tuple)):
    """
    applies func to all elements of x recursively 
    (if the type of element is in iterate_types) such that the results has 
    the same nested structure as x
    
    Example:
    ========
    
    func = lambda x: 2*x
    
    x = [1, [2, 3, (4,5)]]
    y = map_recursive(func, x)
    
    print(x)
    print(y)
    
    [1, [2, 3, (4, 5)]]    
    [2, [4, 6, (8, 10)]]
    
    """
    if not isinstance(x, iterate_types):
        return func(x)
    else:
        return type(x)(map(partial(map_recursive, func),x))



