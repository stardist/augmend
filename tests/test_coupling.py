"""

mweigert@mpi-cbg.de
"""
import numpy as np
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
