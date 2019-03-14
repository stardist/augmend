# Augme*nd*

Augmentation of n-dimensional arrays.

![](imgs/augmerino.png)

Currently implemented:

* flips and 90 degree rotations
* elastic deformation 

## Installation

`pip install  git+https://github.com/mpicbg-csbd/augmend`

## Usage

### Basic augmentation pipeline (single images)

First instantiate an augmentation pipeline class and then populate it with augmentation transforms (e.g. fliprotations, elastic transforms, etc). 

```python
from augmend import Augmend           
from augmend import Elastic, FlipRot90, AdditiveNoise

# define augmentation pipeline
aug = Augmend()

# define transforms
aug.add(FlipRot90(axis = (0,1)), probability=1)
aug.add(Elastic(axis = (0,1)),probability=1)
aug.add(AdditiveNoise(sigma = 0.3),probability=1)
#...

```

Afterwards, it can be applied to an image `img` by simply calling `aug(img)`

```python 
import numpy as np 
import matplotlib.pyplot as plt 

# input
img = np.zeros((128, 128), np.float32)
img[::16] = 1 

# output
result = aug(img)


plt.subplot(121);plt.title("img");plt.imshow(img)
plt.subplot(122);plt.title("result");plt.imshow(result)

```
![alt text](imgs/example2d.png)

### Simultanously transforming several images

Often, one is given several input images `[X,Y,...]` that need to be transformed the same way e.g. image/label pairs for supervised learning). 
To that end, `Augmend.add` accepts a list of transforms, which then will be applied to each image in the input with the same random seed.

```python 
aug.add([FlipRot90(),FlipRot90()], probability=1)

[X2,Y2] = aug([X,Y])

```


### Augmenting in 3D

Should work the same way - in fact, almost all augmentations should accept nD arrays. The axis over which the transformation is applied can be typically set via the `axis` parameter in the transform object, e.g. `FlipRot90(axis = (0,1,2))`.

Example:


```python
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

# resulting volume
res = aug([x, y])
```

Should result in a similar output like this (From left to right: original and 4 augmented volumes. Top and bottom, image `x` and labels `y`).

![alt text](imgs/examples.png)


### Drop-in replacement for (e.g. keras) data generator 

In a supervised learning setting, one often constructs a data generator  that yeilds batches of array pairs 

```
# a simple data generator (might as well return several arrays, as for a supervised data generator) 
def data_gen():
    for i in range(4):
        yield x_batch, y_batch
```

`Augmend.flow` allows to wrap that generator into the augmented one, like so



```
aug = Augmend()

aug.add([FlipRot90(axis=(1, 2)),
         FlipRot90(axis=(1, 2))],
        probability=0.9)

aug_gen = aug.flow(data_gen)

# get the results as tuple
res = next(aug_gen)
```


### Transforming arrays on the GPU

Some transforms (e.g. `Elastic` and `Scale`) allow to use the GPU for the transformation (which can be a bottleneck) via the keyword `use_gpu`. This requires additionally the installation of [`gputools`](https://github.com/maweigert/gputools)




