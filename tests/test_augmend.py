"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend.augmend import Augmend, ElasticAugmenter, FlipRotAugmenter

if __name__ == '__main__':
    arr = np.zeros((100, 100), np.float32)
    arr[::8] = 1.
    arr[:,::8] = 1.
    arr[-20:-10,10:40] = .666
    arr[10:-10, 10:20] = .333

    aug = Augmend()
    aug.add(ElasticAugmenter(p=1., amount = 2, order = 0))
    aug.add(FlipRotAugmenter(p=1., axis = (0,1)))

    res = tuple(aug([arr, arr]))
