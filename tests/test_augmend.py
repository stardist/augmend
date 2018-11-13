"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend.augmend import Augmend, ElasticAugmenter, FlipRotAugmenter


if __name__ == '__main__':
    img = np.zeros((100,) * 2, np.float32)

    #arr[30:-50,10:-10,10:20] =1
    img[:60, :20] = 1
    img[::8] = .8
    img[:, ::8] = .8

    aug = Augmend()
    aug.add(ElasticAugmenter(p=1.,amount = 5, order = 1))
    aug.add(FlipRotAugmenter(p=1.))

    res = tuple(aug([img] * 6))

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,len(res)+1,figsize = (10,3))
    fig.subplots_adjust(wspace = 0.04, hspace=0.04, left = 0.05, right=.95)

    for i, (ax,_d) in enumerate(zip(axs,(img,)+res)):
        ax.imshow(_d, vmin = 0, vmax = 1, cmap = "magma")
        ax.axis("off")

    plt.show()
