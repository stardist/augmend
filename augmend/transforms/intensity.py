import numpy as np
from .base import BaseTransform
from ..utils import _validate_rng, _flatten_axis


def additive_noise(x,rng, sigma):
    rng = _validate_rng(rng)

    if callable(sigma):
        return x + sigma(x, rng)
    else:
        if np.isscalar(sigma):
            sigma = (sigma,sigma) 
        assert len(sigma)==2
        noise = rng.uniform(*sigma)
        return x + noise*rng.normal(0, 1,x.shape).astype(np.float32)

def intensity_scale_shift(x, rng, scale, shift, axis, invert=False):
    rng = _validate_rng(rng)
    axis = _flatten_axis(x.ndim, axis)
    size = tuple(1 if i in axis else s for i,s in enumerate(x.shape))
    s = rng.uniform(*scale, size = size).astype(np.float32)
    t = rng.uniform(*shift, size = size).astype(np.float32)
    if invert:
        if not np.issubdtype(np.asarray(x).dtype, np.floating):
            raise ValueError(f"invert=True expects float images in [0, 1], got dtype {np.asarray(x).dtype}")
        return 1 - ((1 - x) * s + t)
    return x * s + t

    
    
    
                    
class AdditiveNoise(BaseTransform):
    """
    Add gaussian noise
    """
    def __init__(self, sigma=.1):
        super().__init__(
            default_kwargs=dict(sigma=sigma),
            transform_func=additive_noise)



class IntensityScaleShift(BaseTransform):
    """
    apply affine intensity shift
    """

    def __init__(self, scale=(.8,1.2), shift = (-.1,.1), axis=None, invert=False):
        """
        all dimensions that are given by axis will be shifted simultaneously 
        axis = None  -> all 
        axis = (0,1) -> axis 2,3... will be shifted independently 
        invert = True -> scale and shift 1 - x instead of x, for float images in [0, 1]
                         with bright background (e.g. brightfield, H&E), which then stays bright
        """
        super().__init__(
            default_kwargs=dict(scale=scale, shift=shift, axis=axis, invert=invert),
            transform_func=intensity_scale_shift)





# ITU-R 601-2 luma, as used by torchvision/PIL for grayscale conversion
_RGB_TO_GRAY = np.array([0.2989, 0.587, 0.114], np.float32)


def _factor_range(value, name, center=1.0, bounds=(0, np.inf)):
    if np.isscalar(value):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        value = (max(center - value, bounds[0]), center + value)
    if not (len(value) == 2 and bounds[0] <= value[0] <= value[1] <= bounds[1]):
        raise ValueError(f"{name} must be a non-negative scalar or a pair (min, max) within {bounds}, got {value}")
    return tuple(float(v) for v in value)


def _rotate_hue(rgb, dh):
    """Rotate the HSV hue of float RGB values (..., 3) by dh (fraction of a full turn)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = np.maximum(np.maximum(r, g), b)
    c = v - np.minimum(np.minimum(r, g), b)
    cs = np.where(c > 0, c, 1)
    # hue in sextants [0, 6)
    h = np.where(v == r, (g - b) / cs, np.where(v == g, (b - r) / cs + 2, (r - g) / cs + 4))
    h[c == 0] = 0
    h += 6 * dh
    # HSV -> RGB: channel value v - c * clip(min(k, 4 - k), 0, 1) with k = (n + h) mod 6, n = 5, 3, 1
    out = np.empty_like(rgb)
    for i, n in enumerate((5, 3, 1)):
        k = h + n
        np.mod(k, 6, out=k)
        np.minimum(k, 4 - k, out=k)
        np.clip(k, 0, 1, out=k)
        k *= c
        np.subtract(v, k, out=out[..., i])
    return out


def color_jitter(x, rng, brightness, contrast, saturation, hue):
    rng = _validate_rng(rng)
    x = np.asarray(x)
    if x.shape[-1] != 3:
        raise ValueError(f"ColorJitter expects RGB images with channels last, got shape {x.shape}")
    if x.dtype == np.uint8:
        y = x.astype(np.float32) / 255
    elif np.issubdtype(x.dtype, np.floating):
        y = x
    else:
        raise ValueError(f"ColorJitter expects uint8 or float (in [0, 1]) images, got dtype {x.dtype}")

    # always draw all factors, so the random stream does not depend on the parameters
    f_brightness = rng.uniform(*brightness)
    f_contrast = rng.uniform(*contrast)
    f_saturation = rng.uniform(*saturation)
    f_hue = rng.uniform(*hue)

    y = np.clip(f_brightness * y, 0, 1)
    mean = np.mean(y @ _RGB_TO_GRAY)
    y = np.clip(f_contrast * y + (1 - f_contrast) * mean, 0, 1)
    gray = (y @ _RGB_TO_GRAY)[..., None]
    y = np.clip(f_saturation * y + (1 - f_saturation) * gray, 0, 1)
    if f_hue != 0:
        y = _rotate_hue(y, f_hue)

    if x.dtype == np.uint8:
        return np.round(255 * y).astype(np.uint8)
    return y.astype(x.dtype, copy=False)


class ColorJitter(BaseTransform):
    """
    Random brightness, contrast, saturation and hue changes of RGB images
    (channels last, uint8 or float in [0, 1]), applied in this order.
    Follows the definitions of torchvision's ColorJitter; all leading axes are treated as one image.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02):
        """
        :param brightness, float or (min, max):
            multiply by a factor in [max(0, 1 - brightness), 1 + brightness] (or [min, max])
        :param contrast, float or (min, max):
            blend with the mean gray value by a factor in [max(0, 1 - contrast), 1 + contrast]
        :param saturation, float or (min, max):
            blend with the grayscale image by a factor in [max(0, 1 - saturation), 1 + saturation]
        :param hue, float or (min, max):
            rotate hue by a value in [-hue, hue] (fraction of a full turn, at most 0.5)
        """
        super().__init__(
            default_kwargs=dict(
                brightness=_factor_range(brightness, "brightness"),
                contrast=_factor_range(contrast, "contrast"),
                saturation=_factor_range(saturation, "saturation"),
                hue=_factor_range(hue, "hue", center=0, bounds=(-0.5, 0.5)),
            ),
            transform_func=color_jitter)
