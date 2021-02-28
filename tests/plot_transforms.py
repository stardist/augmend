"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend.utils import create_test_pattern
from augmend import FlipRot90, Elastic, Identity, AdditiveNoise
from augmend import Augmend




if __name__ == '__main__':

    x = create_test_pattern()
    y = (200*(x>200)).astype(np.int)

    t = FlipRot()
    t = Elastic()+AdditiveNoise(sigma = 20)

    import matplotlib.pyplot as plt

    n = 4
    fig = plt.figure(num=1, figsize=(5, 4))
    fig.clf()
    fig.subplots_adjust(wspace = .1, hspace = 0, left= 0.05, bottom = 0.05, right = 0.95)
    axs = fig.subplots(1, n+1)

    np.random.seed(42)
    for i, ax in enumerate(axs):
        ax.cla()
        ax.imshow(t(x) if i>0 else x)
        ax.axis("off")
        ax.set_title("Original" if i==0 else "%s"%i, fontsize=6)

    plt.show()
