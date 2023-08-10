import numpy as np
from augmend import Data, Augmend, Identity, Flip, FlipRot90, AdditiveNoise
from utils import create_checkerboard 

if __name__ == '__main__':

    x, p = create_checkerboard((128,128))

    t = Identity() 
    t = Flip()


    d = t(Data(image=x, points=p))

    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(num=1)
    fig.clf()
    axs = fig.subplots(1,2)
    axs[0].imshow(x)
    axs[0].plot(*p.T[::-1], 'o',color='C1', alpha=.7)
    axs[1].imshow(d.image)
    axs[1].plot(*d.points.T[::-1], 'o',color='C1', alpha=.7)
    
    plt.show()
