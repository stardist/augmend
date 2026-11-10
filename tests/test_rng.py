import pickle

import numpy as np
import pytest

import augmend as A

# checksums of outputs produced by augmend 0.1.3 with the same seeds (see _checksum); Scale and IsotropicScale (one
# affine interpolation and the draws of transform_scale) from 0.2.0
REFERENCE = {
    "FlipRot90": (2619.8278, 3550.5581),
    "Flip": (2619.8278, 3550.5581),
    "Rotate": (2478.8058, 3898.7692),
    "Scale": (1997.7349, 3330.7765),  # log-uniform scale
    "IsotropicScale": (1779.053, 2984.5487),  # log-uniform scale
    "Shift": (1617.1118, 2750.0489),
    "Elastic": (2757.2672, 4521.8608),
    "CutOut": (2607.571, 4319.3833),
    "GaussianBlur": (2619.8278, 4311.8314),
    "DropPlanes": (2581.7701, 4279.3022),
    "DropEdgePlanes": (2619.2435, 4334.3832),
    "AdditiveNoise": (2626.4461, 4348.1016),
    "IntensityScaleShift": (3021.7011, 4987.7980),  # log-uniform scale
    "RandomCrop": (458.2208, 746.7831),
}
REFERENCE_PIPELINE = [
    ((2542.6039, 4120.2919), (2545.7995, 4125.829)),  # log-uniform Scale
    ((2511.9749, 3645.2794), (2512.6429, 3645.5309)),
    ((2374.3567, 3915.1511), (2375.5208, 3917.0396)),  # log-uniform Scale
]


def _image():
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    return (np.sin(yy / 3.0) + np.cos(xx / 5.0) + 0.01 * yy * xx).astype(np.float32)


def _checksum(y):
    y = np.asarray(y, np.float64)
    w = np.linspace(1, 2, y.size).reshape(y.shape)
    return (float(y.sum()), float((w * y).sum()))


def _transforms():
    return {
        "FlipRot90": A.FlipRot90(),
        "Flip": A.Flip(),
        "Rotate": A.Rotate(),
        "Scale": A.Scale(),
        "IsotropicScale": A.IsotropicScale(),
        "Shift": A.Shift(),
        "Elastic": A.Elastic(grid=4, amount=3),
        "CutOut": A.CutOut(width=(3, 6), n=(1, 3)),
        "GaussianBlur": A.GaussianBlur(),
        "DropPlanes": A.DropPlanes(),
        "DropEdgePlanes": A.DropEdgePlanes(),
        "AdditiveNoise": A.AdditiveNoise(sigma=0.5),
        "IntensityScaleShift": A.IntensityScaleShift(),
        "RandomCrop": A.RandomCrop(size=(16, 16)),
    }


def _pipeline(rng=None):
    aug = A.Augmend(rng=rng)
    aug.add([A.Elastic(grid=4, amount=3), A.Elastic(grid=4, amount=3, order=0)])
    aug.add([A.FlipRot90() + A.AdditiveNoise(sigma=0.3)] * 2, probability=0.7)
    aug.add([A.Rotate() | A.Scale()] * 2)
    return aug


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_reproducible_transform(name):
    aug = A.Augmend(_transforms()[name], rng=np.random.RandomState(42))
    assert _checksum(aug(_image())) == pytest.approx(REFERENCE[name], rel=1e-5)


def test_reproducible_pipeline():
    aug = _pipeline(np.random.RandomState(7))
    for ref in REFERENCE_PIPELINE:
        x, y = aug([_image(), _image()])
        assert _checksum(x) + _checksum(y) == pytest.approx(ref[0] + ref[1], rel=1e-5)


def test_global_rng_resolved_at_call_time():
    aug = _pipeline()
    np.random.seed(0)
    a = aug([_image(), _image()])
    np.random.seed(0)
    b = aug([_image(), _image()])
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_call_rng_does_not_replace_pipeline_rng():
    aug = _pipeline()
    aug([_image(), _image()], rng=np.random.RandomState(0))
    assert aug._rng is np.random.mtrand._rand


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_generator_rng(seed):
    aug = A.Augmend()
    aug.add([A.Elastic(grid=4, amount=3), A.Elastic(grid=4, amount=3)])
    aug.add([A.FlipRot90() | A.Rotate()] * 2)
    x, y = aug([_image(), _image()], rng=np.random.default_rng(seed))
    assert np.allclose(x, y)
    x2, y2 = aug([_image(), _image()], rng=np.random.default_rng(seed))
    assert np.array_equal(x, x2)


def test_generator_rng_direct_transform():
    t = A.Elastic(grid=4, amount=3)
    a = t(_image(), rng=np.random.default_rng(0))
    b = t(_image(), rng=np.random.default_rng(0))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", sorted(REFERENCE) + ["Identity", "Lambda", "concat", "choice"])
def test_pickle(name):
    extra = {
        "Identity": A.Identity(),
        "Lambda": A.Lambda(),
        "concat": A.FlipRot90() + A.Elastic(grid=4, amount=3),
        "choice": A.Flip() | A.Rotate(),
    }
    t = extra[name] if name in extra else _transforms()[name]
    aug = A.Augmend([t, t])
    aug2 = pickle.loads(pickle.dumps(aug))
    a = aug([_image(), _image()], rng=np.random.RandomState(3))
    b = aug2([_image(), _image()], rng=np.random.RandomState(3))
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
