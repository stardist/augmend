import os
import numpy as np
from tifffile import imread
from skimage.measure import label
from pathlib import Path



def create_checkerboard(shape, step = 32):
    xs = tuple(np.arange(s) for s in shape)
    Xs = np.meshgrid(*xs, indexing = "ij")
    Xs_check1 = tuple(2*(X//step%2)-1 for X in Xs)
    Xs_check2 = tuple(2*(X//(step//2)%2)-1 for X in Xs)

    u = 2+np.prod(np.stack(Xs_check1),axis = 0)
    u2 = 2+np.prod(np.stack(Xs_check2),axis = 0)
    mask = np.sum(Xs, axis = 0)<np.min(shape)//2

    # u[mask] = .5*u2[mask]
    u[mask] = 2

    return (1000*u).astype(np.uint16)
    


