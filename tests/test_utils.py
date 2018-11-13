"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend.utils import map_recursive



if __name__ == '__main__':


    def foo(x):
        print("here: ",x)
        return 2*x


    x = [1, [2, 3],[3,4,[5,6]]]

    y = map_recursive(foo, x)

    print(x)
    print(y)

