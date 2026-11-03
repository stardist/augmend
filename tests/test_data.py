import numpy as np

from augmend import data


def test_example_images():
    x, y = data.test_image_nuclei_2d(return_mask=True)
    assert x.shape == y.shape == (512, 512) and x.dtype == np.uint16 and y.max() > 0
    assert np.array_equal(data.test_image_nuclei_2d(), x)
    he = data.test_image_he_2d()
    assert he.shape == (300, 500, 3) and he.dtype == np.uint8
