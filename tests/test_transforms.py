"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from augmend import FlipRot90, Elastic, Identity

if __name__ == '__main__':
    x = np.zeros((100, 111), np.float32)
    x[:60, :90][::10] = 1
    x[:60, :90][:, ::10] = 1.5


    transforms = (
        FlipRot90(axis=0),
        FlipRot90(axis=1),
        FlipRot90(axis=0) + Identity(),
    )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(transforms), 4, num=1)

    for t, ax in zip(transforms, axs):
        for _ax in ax:
            _ax.cla()
            _ax.imshow(t(x))
            _ax.axis("off")
            _ax.set_title(t, fontsize=6)

    plt.show()
