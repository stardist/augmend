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
from augmend.augmend import Augmend, ElasticAugmenter, FlipRotAugmenter

img = np.zeros((100,) * 3, np.float32)
img[-20:,:20, :20] = 1.
img[30:40, -10:] = .8
Xs = np.meshgrid(*((np.arange(0, 100),) * 3), indexing="ij")
R = np.sqrt(np.sum([(X - c) ** 2 for X, c in zip(Xs, (70, 60, 50))], axis=0))
img[R<20] = 1.4


aug = Augmend()

aug.add(ElasticAugmenter(p=1., axis = (0,1,2),amount = 5, order = 1))
aug.add(FlipRotAugmenter(p=1., axis = (1,2)))

def data_gen():
	for i in range(4):
		yield img

aug_gen = aug(data_gen())

res = tuple(aug_gen)


```
Should result in 

![alt text](imgs/examples.png)





