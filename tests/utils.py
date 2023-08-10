import os
import numpy as np
from tifffile import imread
from skimage.measure import label
from pathlib import Path



def create_checkerboard(shape, step = 16):
    xs = tuple(np.arange(s) for s in shape)
    Xs = np.meshgrid(*xs, indexing = "ij")
    Xs_check1 = tuple(2*(X//step%2)-1 for X in Xs)

    u = 2+np.prod(np.stack(Xs_check1),axis = 0)
    mask = np.sum([(1+2*i)*x for i, x in enumerate(Xs)], axis = 0)<np.min(shape)//2

    u[mask] = 2

    img = (1000*u).astype(np.uint16)

    rng = np.random.RandomState(42)

    points = rng.choice(np.prod(shape), np.prod(shape)//500, replace = False)
    points = np.stack(np.unravel_index(points, shape), axis = 1)

    img[tuple(points.T)] = 2400
    return img, points
    


