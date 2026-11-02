import pickle

import numpy as np
import pytest

from augmend import Augmend, ColorJitter, Identity, IntensityScaleShift


def test_invert_keeps_background():
    x = np.ones((16, 16), np.float32)
    x[4:8] = 0.2
    y = IntensityScaleShift(scale=(0.5, 2), shift=(0, 0), invert=True)(x, rng=np.random.RandomState(0))
    assert np.allclose(y[x == 1], 1)
    assert not np.allclose(y[4:8], 0.2)


@pytest.mark.parametrize("seed", range(3))
def test_invert_equals_transform_of_inverse(seed):
    x = np.random.RandomState(seed).rand(16, 16, 3).astype(np.float32)
    kwargs = dict(scale=(0.8, 1.2), shift=(-0.1, 0.1), axis=(0, 1))
    y = IntensityScaleShift(invert=True, **kwargs)(x, rng=np.random.RandomState(seed))
    y_ref = 1 - IntensityScaleShift(**kwargs)(1 - x, rng=np.random.RandomState(seed))
    assert np.allclose(y, y_ref)


def test_invert_rejects_integer_images():
    with pytest.raises(ValueError):
        IntensityScaleShift(invert=True)(np.zeros((8, 8), np.uint8))


def _rgb(seed=0):
    x = np.random.RandomState(seed).rand(32, 40, 3).astype(np.float32)
    x[:4] = 0.5  # gray
    x[4:6] = 0  # black
    return x


def test_color_jitter_matches_torchvision():
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torchvision.transforms.functional")
    from augmend.transforms.intensity import color_jitter

    x = _rgb()
    rng = np.random.RandomState(0)
    for _ in range(20):
        fb, fc, fs, fh = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5), rng.uniform(0, 2), rng.uniform(-0.5, 0.5)
        t = torch.from_numpy(np.moveaxis(x, -1, 0).copy())
        t = F.adjust_hue(F.adjust_saturation(F.adjust_contrast(F.adjust_brightness(t, fb), fc), fs), fh)
        y = color_jitter(x, None, (fb, fb), (fc, fc), (fs, fs), (fh, fh))
        assert np.allclose(y, np.moveaxis(t.numpy(), 0, -1), atol=1e-5)


def test_color_jitter_identity():
    x = _rgb()
    y = ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)(x, rng=np.random.RandomState(0))
    assert np.allclose(x, y, atol=1e-6)


@pytest.mark.parametrize("seed", range(3))
def test_color_jitter_hue_saturation_keep_gray(seed):
    x = np.repeat(np.random.RandomState(seed).rand(16, 16, 1), 3, axis=-1).astype(np.float32)
    y = ColorJitter(brightness=0, contrast=0, saturation=0.5, hue=0.5)(x, rng=np.random.RandomState(seed))
    # grayscale weights (as in torchvision) sum to 0.9999
    assert np.allclose(x, y, atol=1e-4)


@pytest.mark.parametrize("dtype", (np.uint8, np.float32, np.float64))
def test_color_jitter_dtype(dtype):
    x = _rgb()
    x = (255 * x).astype(np.uint8) if dtype == np.uint8 else x.astype(dtype)
    y = ColorJitter()(x, rng=np.random.RandomState(0))
    assert y.dtype == x.dtype and y.shape == x.shape
    a = ColorJitter()(x, rng=np.random.RandomState(1))
    b = ColorJitter()(x, rng=np.random.RandomState(1))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("x", (np.zeros((8, 8, 4), np.float32), np.zeros((8, 8, 3), np.uint16)))
def test_color_jitter_invalid_input(x):
    with pytest.raises(ValueError):
        ColorJitter()(x)


@pytest.mark.parametrize("kwargs", (dict(brightness=-0.1), dict(hue=0.6), dict(contrast=(1.2, 0.8))))
def test_color_jitter_invalid_params(kwargs):
    with pytest.raises(ValueError):
        ColorJitter(**kwargs)


def test_color_jitter_pipeline_with_labels():
    x = _rgb()
    lbl = (x[..., 0] > 0.5).astype(np.uint16)
    aug = Augmend()
    aug.add([ColorJitter(hue=0.1), Identity()])
    aug2 = pickle.loads(pickle.dumps(aug))
    x1, l1 = aug([x, lbl], rng=np.random.RandomState(0))
    x2, _ = aug2([x, lbl], rng=np.random.RandomState(0))
    assert np.array_equal(l1, lbl) and np.array_equal(x1, x2) and not np.allclose(x1, x)
