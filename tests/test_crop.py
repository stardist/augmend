import numpy as np
from augmend import RandomCrop
from augmend.utils import create_test_pattern
import pytest

def simple_demo():
    img = create_test_pattern()

    t = RandomCrop()
    out = t(img)

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(out)

@pytest.mark.parametrize(["shape", "axis", "size", "expected_shape"], [[(500, 512), None, (256, 253), (256, 253)], [(32, 33, 36), (1, 2), (33, 12), (32, 33, 12)]])
def test_shapes_random_crop(shape, axis, size, expected_shape):
    t = RandomCrop(size=size, axis=axis)
    img = np.empty(shape)
    out = t(img)
    assert out.shape == expected_shape

@pytest.mark.parametrize(["shape", "axis", "size"], [[(500, 512), (1,), (256, 253)], [(32, 33, 36), (1, 2), (34, 12)]])
def test_shapes_random_crop_fail(shape, axis, size):
    t = RandomCrop(size=size, axis=axis)
    img = np.empty(shape)
    with pytest.raises(ValueError):
        t(img)

if __name__ == '__main__':
    # simple_demo()
    test_shapes_random_crop((500, 512), None, (200, 203))