import numpy as np
from scipy.misc import ascent
from augmend.transforms import transform_elastic
from time import time

if __name__ == '__main__':

    img = ascent()

    img = np.zeros((128,) * 3, np.uint16)
    img = np.zeros((128,128,128), np.uint16)

    # img[::16] = 128
    # img[:,::16] = 128
    img[10:-10, 40:-40, 40:-40] = 128
    # img = np.ones((128,77))
    # img[::10] = 2
    # img[:,::10] = 3


    t = time()
    out1 = transform_elastic(img, axis=(1, 2), grid=4, amount=5, order=0,
                             rng=np.random.RandomState(0))
    print("%.4f s"%(time()-t))

    t = time()
    out2 = transform_elastic(img, axis=(1, 2), grid=4, amount=5, order=0,
                             rng=np.random.RandomState(0),
                             workers = 4)

    print("%.4f s" % (time() - t))

    assert np.allclose(out1, out2)