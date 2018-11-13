import numpy as np
from scipy.misc import ascent
from augmend.transforms import transform_elastic

if __name__ == '__main__':
    img = ascent()

    img = np.zeros((128,) * 3, np.uint16)

    # img[::16] = 128
    # img[:,::16] = 128
    img[40:-40, 40:-40, 40:-40] = 128
    # img = np.ones((128,77))
    # img[::10] = 2
    # img[:,::10] = 3


    out = transform_elastic(img, grid=11, amount=5, order=0, seed = 0)

    out2 = transform_elastic(img.astype(np.float32), grid=11, amount=5, order=1, seed = 0)
