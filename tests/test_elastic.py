import numpy as np
from augmend.transforms import Elastic
from augmend.utils import test_pattern, plot_augmented

from augmend import Elastic
import time

if __name__ == '__main__':
    img = test_pattern(ndim=2, shape = (128,128,128))

    t  = time.time()
    trans = Elastic(use_gpu=True,
                    # use_gpu=False,
                     order=0,
                     amount = 2, grid = 5)

    out = trans(img, rng = np.random.RandomState(0))
    out = out[out.shape[0]//2]
    print("%.2f ms" % (1000*(time.time() - t)))


    import matplotlib.pyplot as plt
    plt.figure(num=1)
    plt.imshow(out)
    plt.show()


    #
    # t = Elastic(axis=(0,1), grid=4, amount=5, order=0)
    #
    # fig = plot_augmented(t,(img,img>80, img), num=1, rng = np.random.RandomState(0))
    #
    # fig.show()