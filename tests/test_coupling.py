"""

mweigert@mpi-cbg.de
"""
import numpy as np
import pytest
from scipy import ndimage
from augmend import Augmend, Elastic, FlipRot90, Scale, Shift, Rotate, IntensityScaleShift, GaussianBlur, AdditiveNoise, CutOut, IsotropicScale, Identity, DropEdgePlanes, RandomCrop

def test_coupling():
    
    np.random.seed(22)
    
    img = np.random.uniform(0,1,(100,200)).astype(np.float32)
    

    transforms = dict(
        identity = Identity(),
        elastic = Elastic(),
        fliprot = FlipRot90(),
        scale = Scale(),
        isoscale = IsotropicScale(),
        rotate = Rotate(),
        intensity = IntensityScaleShift(scale = (0.4,2)),
        gaussian = GaussianBlur(), 
        noise = AdditiveNoise(sigma=.2),
        cutout = CutOut(width = (40,41)),
        dropedges = DropEdgePlanes(width = 20),
        randomcrop = RandomCrop(size=(64, 65))
        )

    for i,(name, t) in enumerate(transforms.items()):
        print(name)
        aug = Augmend()
        aug.add([t,t, [t,t]])

        for _ in range(50):
            x,y,(u,w) = aug([img,img, [img, img]])
            same = np.allclose(x,y) and np.allclose(y,u) and np.allclose(u,w)
            if not same:
                print(name)
            assert same

        
    

    
    

if __name__ == '__main__':
    pass


def test_cval():
    """image and labels of one pipeline can be padded differently"""
    rng = np.random.RandomState(0)
    labels = rng.randint(1, 100, (64, 64)).astype(np.int32)
    transforms = dict(
        scale = lambda **k: Scale(amount=(.5, .9), **k),
        isoscale = lambda **k: IsotropicScale(amount=(.5, .9), **k),
        rotate = lambda **k: Rotate(**k),
        shift = lambda **k: Shift(amount=(5, 10), **k),
        elastic = lambda **k: Elastic(amount=40, **k),
    )
    for name, t in transforms.items():
        aug = Augmend()
        aug.add([t(order=1, mode="reflect"), t(order=0, mode="constant", cval=-1)])
        x, y = aug((labels.astype(np.float32), labels), rng=np.random.RandomState(1))
        assert x.shape == y.shape == labels.shape and y.dtype == labels.dtype, name
        assert (y == -1).any() and set(np.unique(y)) <= set(np.unique(labels)) | {-1}, name
        assert x.min() >= labels.min(), name  # no zeros from padding
        # the default still fills with zeros
        y0 = t(order=0, mode="constant")(labels, rng=np.random.RandomState(1))
        y1 = t(order=0, mode="constant", cval=-1)(labels, rng=np.random.RandomState(1))
        np.testing.assert_array_equal(y0 == 0, y1 == -1, err_msg=name)
    # channels share the transform and the padding
    aug = Scale(axis=(0, 1), amount=(.5, .9), order=0, mode="constant", cval=-1)
    y = aug(np.stack([labels, labels], axis=-1), rng=np.random.RandomState(1))
    assert (y[..., 0] == -1).any()
    np.testing.assert_array_equal(y[..., 0], y[..., 1])


@pytest.mark.parametrize("transform", (Scale, IsotropicScale))
def test_scale_is_log_uniform(transform, monkeypatch):
    affine, scales = ndimage.affine_transform, []
    monkeypatch.setattr(ndimage, "affine_transform", lambda x, matrix, **k: (scales.append(1 / np.asarray(matrix)), affine(x, matrix, **k))[1])
    aug = transform(amount=(0.25, 4))
    rng = np.random.RandomState(0)
    for _ in range(2000):
        aug(np.zeros((4, 4), np.float32), rng=rng)
    s = np.array(scales)
    assert s.min() >= 0.25 and s.max() <= 4
    assert abs(np.median(np.log(s))) < 0.1  # a factor and its inverse are equally likely
    assert abs(np.mean(s < 0.5) - 0.25) < 0.03 and abs(np.mean(s > 2) - 0.25) < 0.03
    with pytest.raises(ValueError):
        transform(amount=(0, 2))(np.zeros((4, 4), np.float32))


def test_scale_linear_ramp():
    """order 1 reproduces a linear image exactly, scaled about the center"""
    yy, xx = np.meshgrid(np.arange(40.0), np.arange(30.0), indexing="ij")
    img = (2 * yy + 3 * xx).astype(np.float64)
    out = Scale(amount=((1.6, 1.6), (0.8, 0.8)), order=1)(img, rng=np.random.RandomState(0))
    cy, cx = 19.5, 14.5
    expected = 2 * (cy + (yy - cy) / 1.6) + 3 * (cx + (xx - cx) / 0.8)
    inside = (np.abs(xx - cx) / 0.8 <= cx)  # samples within the input (the shrunk axis is padded)
    np.testing.assert_allclose(out[inside], expected[inside], atol=1e-9)


def test_scale_isotropic_factor():
    """the shared factor multiplies the per-axis ones, drawn with isotropic_probability"""
    affine, scales = ndimage.affine_transform, []
    img = np.zeros((8, 8, 3), np.float32)
    rng = np.random.RandomState(0)
    try:
        ndimage.affine_transform = lambda x, matrix, **k: (scales.append(1 / np.asarray(matrix)), affine(x, matrix, **k))[1]
        for _ in range(400):
            Scale(axis=(0, 1), amount=((1, 1), (2, 2)), isotropic=(3, 3), isotropic_probability=0.5)(img, rng=rng)
    finally:
        ndimage.affine_transform = affine
    s = np.round(np.array(scales), 6)
    assert np.all(s[:, 2] == 1)  # the channel axis is not scaled
    enlarged = s[:, 0] == 3
    assert set(map(tuple, s[enlarged])) == {(3, 6, 1)} and set(map(tuple, s[~enlarged])) == {(1, 2, 1)}
    assert abs(enlarged.mean() - 0.5) < 0.08
    # IsotropicScale is the shared factor alone
    a = IsotropicScale(axis=(0, 1), amount=(0.5, 2))(np.random.RandomState(1).uniform(size=(16, 16)), rng=np.random.RandomState(2))
    b = Scale(axis=(0, 1), amount=1, isotropic=(0.5, 2))(np.random.RandomState(1).uniform(size=(16, 16)), rng=np.random.RandomState(2))
    np.testing.assert_array_equal(a, b)
    for bad in (dict(isotropic=(0, 2)), dict(isotropic=(1, 2), isotropic_probability=1.5)):
        with pytest.raises(ValueError):
            Scale(**bad)(img)
