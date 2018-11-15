# Augme*nd*

Augmentation of n-dimensional arrays.


Currently implemented:

* flips and 90 degree rotations
* elastic deformation 

## Installation

`pip install  git+https://github.com/mpicbg-csbd/augmend`

## Usage

```python

import numpy as np
from augmend import Augmend, ElasticAugmenter, FlipRotAugmenter

# example 3d image and label
img = np.zeros((100,) * 3, np.float32)
img[-20:,:20, :20] = 1.
img[30:40, -10:] = .8
Xs = np.meshgrid(*((np.arange(0, 100),) * 3), indexing="ij")
R = np.sqrt(np.sum([(X - c) ** 2 for X, c in zip(Xs, (70, 60, 50))], axis=0))
img[R<20] = 1.4

lbl = np.zeros((100,)*3,np.uint16)
lbl[R<20] = 200


# define augmentation pipeline
aug = Augmend()
aug.add(ElasticAugmenter(p=1., axis = (0,1,2),amount = 5, order = lambda x: 0 if x.dtype.type == np.uint16 else 1))
aug.add(FlipRotAugmenter(p=1., axis = (1,2)))

# a simple data generator (might as well return several arrays, as for a supervised data generator) 
def data_gen():
    for i in range(4):
        yield img, lbl

g = data_gen()

# apply augmentation to any generator that returns one or sevral arrays
# The same augmentation pipeline will be applied to all ndarrays within a single element of the generator
aug_gen = aug(g)

# get the results as tuple
res = tuple(aug_gen)


```
Should result in the following output. From left to right: orginal and 4 augmented volumes. Top and bottom, image (`img`) and labels (`lbl`). 

![alt text](imgs/examples.png)





