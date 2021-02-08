"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from imageio import imread
from tifffile import imread
import matplotlib.pyplot as plt
from augmend import Augmend, Elastic, FlipRot90, Scale, Rotate, IntensityScaleShift, GaussianBlur, AdditiveNoise, CutOut, IsotropicScale, Identity
from collections import OrderedDict
from csbdeep.utils import normalize

if __name__ == '__main__':
    img = imread("example/data/img_9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32.tif")

    img = normalize(img)
    img[::16] = .7
    img[:,::16] = .7
    
    np.random.seed(22)

    cmap = "gray"
    
    transforms = OrderedDict(
        identity = Identity(),
        elastic = Elastic(),
        fliprot = FlipRot90(),
        scale = Scale(),
        isoscale = IsotropicScale(),
        rotate = Rotate(),
        intensity = IntensityScaleShift(scale = (0.4,2)),
        gaussian = GaussianBlur(), 
        noise = AdditiveNoise(sigma=.2),
        cutout = CutOut(width = (40,41))
        )

    fig = plt.figure(figsize=(10,4), num=1)
    fig.clf()
    axs = fig.subplots(1,2)
    for ax in axs:
        ax.axis("off")
    
    for i,(name, t) in enumerate(transforms.items()):
        print(name)
        axs[0].imshow(img, cmap = cmap, clim = (0,1))
        axs[0].set_title("Identity()")
        axs[1].imshow(t(img), cmap = cmap, clim = (0,1))
        axs[1].set_title(f"{t.__class__.__name__}()")
        fig.savefig(f"../imgs/example_{i:02d}_{name}.png")


    plt.show()
        


    

    
    
