"""
Example of Readme
mweigert@mpi-cbg.de
"""

import numpy as np
from augmend import Augmend, Elastic, FlipRot90

# define augmentation pipeline
aug = Augmend()
aug.add([FlipRot90(axis=(0, 1, 2)),
         FlipRot90(axis=(0, 1, 2))],
        probability=0.9)

aug.add([Elastic(axis=(0, 1, 2), amount=5, order=1),
         Elastic(axis=(0, 1, 2), amount=5, order=0)],
        probability=0.9)


# example 3d image and label
x = np.zeros((100,) * 3, np.float32)
x[-20:, :20, :20] = 1.
x[30:40, -10:] = .8
Xs = np.meshgrid(*((np.arange(0, 100),) * 3), indexing="ij")
R = np.sqrt(np.sum([(X - c) ** 2 for X, c in zip(Xs, (70, 60, 50))], axis=0))
x[R < 20] = 1.4

y = np.zeros((100,) * 3, np.uint16)
y[R < 20] = 200

# a simple data generator that returns a single pair of (image, label)
def data_gen():
    for i in range(4):
        yield x, y


g = data_gen()

# apply augmentation to any generator that returns one or sevral arrays
# The same augmentation pipeline will be applied to all ndarrays within a single element of the generator
aug_gen = aug.flow(g)

# get the results as tuple
res = tuple(aug_gen)
