import numpy as np
from augmend import Augmend, RandomCrop
from augmend.utils import create_test_pattern
import pytest

def simple_demo():
    y = create_test_pattern()
    x = y.astype(np.float32)+60*np.random.uniform(0,1,y.shape)

    aug = Augmend() 
    aug.add([RandomCrop(size=(64,64)), RandomCrop(size=(64,64))])
    
    x2, y2 = aug([x,y])

    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    fig, axs = plt.subplots(2,2, num=1)
    
    titles = 'image', 'label'

    for _x, _ax, _t in zip((x,y), axs[0], titles):
        _ax.imshow(_x)
        _ax.set_title(_t)
    for _x, _ax, _t in zip((x2,y2), axs[1], titles):
        _ax.imshow(_x)
        _ax.set_title(_t)
    
    plt.tight_layout()
        
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

@pytest.mark.parametrize(["shape", "size"], [[(128, 128), (64, 64)], [(128, 365), (32, 58)], [(128, 127), (128, 127)]])
def test_matching_crops(shape, size):
    x = np.random.randint(0, 100, shape)
    augmenter = Augmend([RandomCrop(size=size), RandomCrop(size=size)])
    a, b = augmenter([x, x])
    assert np.allclose(a, b)

if __name__ == '__main__':
    simple_demo()