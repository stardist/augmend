"""
H&E stain augmentation: perturb stain vectors and stain amounts estimated via NMF of optical densities.
"""
import warnings

import numpy as np

from .base import BaseTransform
from ..utils import _validate_rng

# linear sRGB -> CIE Y (D65)
_RGB_TO_Y = np.array([0.212671, 0.715160, 0.072169])
# fewer tissue pixels than this: stain matrix is not estimated and the input is returned unchanged
_MIN_TISSUE_SAMPLES = 32


def _lightness(rgb):
    """CIE L* in [0, 100] of sRGB values in [0, 1] with shape (..., 3)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = lin @ _RGB_TO_Y
    f = np.where(y > 0.008856, np.cbrt(y), 7.787 * y + 16 / 116)
    return 116 * f - 16


def _to_unit_range(x):
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255
    if np.issubdtype(x.dtype, np.floating):
        return np.clip(x, 0, 1)
    raise ValueError(f"HEStaining expects uint8 or float (in [0, 1]) images, got dtype {x.dtype}")


def estimate_stain_matrix(rgb, n_pixels=4096, tissue_threshold=78.4, l1_reg=0.001):
    """
    Estimate the H&E stain matrix of an RGB image via non-negative matrix factorization
    of its optical densities, density = stains @ M with M of shape (2, 3).

    :param rgb, ndarray:
        RGB image in [0, 1] with shape (..., 3)
    :param n_pixels, int:
        number of randomly chosen pixels (fixed seed) used for fitting
    :param tissue_threshold, float:
        only pixels with CIE lightness L* (0-100) below this value are used for fitting
    :param l1_reg, float:
        sparsity regularization of the stain amounts
    :return ndarray or None:
        stain matrix (hematoxylin first) with unit-norm rows, or None if there is too little tissue
    """
    try:
        from sklearn.decomposition import NMF
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as e:
        raise ImportError('HEStaining requires scikit-learn: pip install "augmend[he]"') from e

    rgb = rgb.reshape(-1, 3)
    if len(rgb) > n_pixels:
        # fixed seed: the estimate depends only on the image, and the augmentation rng is not consumed
        rgb = rgb[np.random.RandomState(0).randint(0, len(rgb), n_pixels)]
    values = -np.log(np.clip(rgb[_lightness(rgb) < tissue_threshold], 1 / 255, 1))
    values = np.maximum(values, 1e-6)
    if len(values) < _MIN_TISSUE_SAMPLES:
        return None

    model = NMF(n_components=2, init="random", random_state=0, alpha_W=l1_reg, alpha_H=0, l1_ratio=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(values)

    M = model.components_
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    if np.any(norm == 0):
        return None
    M = M / norm
    # hematoxylin absorbs more red than eosin
    if M[0, 0] < M[1, 0]:
        M = M[::-1]
    return M


def he_staining(x, rng=None, amount_matrix=0.15, amount_stains=0.4, n_pixels=4096, tissue_threshold=78.4):
    rng = _validate_rng(rng)
    x = np.asarray(x)
    if x.shape[-1] != 3:
        raise ValueError(f"HEStaining expects RGB images with channels last, got shape {x.shape}")
    rgb = _to_unit_range(x)

    # draw first, so the random stream does not depend on the image content
    d_matrix = rng.uniform(-1, 1, (2, 3))
    d_stains = rng.uniform(-1, 1, 2)

    M = estimate_stain_matrix(rgb, n_pixels=n_pixels, tissue_threshold=tissue_threshold)
    if M is None:
        return x

    density = np.maximum(-np.log(np.clip(rgb, 1 / 255, 1)), 1e-6).reshape(-1, 3)
    stains = density @ np.linalg.pinv(M)
    stains = np.maximum(stains * (1 + amount_stains * d_stains), 0)
    M = np.maximum(M + amount_matrix * d_matrix, 0)
    out = np.clip(np.exp(-(stains @ M)), 0, 1).reshape(x.shape)

    if x.dtype == np.uint8:
        return np.round(255 * out).astype(np.uint8)
    return out.astype(x.dtype, copy=False)


class HEStaining(BaseTransform):
    """
    H&E stain augmentation of RGB images (channels last, uint8 or float in [0, 1]).

    Estimates the hematoxylin and eosin stain vectors via NMF of the optical densities,
    then randomly perturbs the stain vectors (by +- amount_matrix) and the stain amounts
    (by a factor 1 +- amount_stains). Images with too little tissue are returned unchanged.
    All leading axes are treated as one image. Requires scikit-learn (pip install "augmend[he]").

    Example: aug.add([HEStaining(), Identity()]) for (image, label) pairs
    """

    def __init__(self, amount_matrix=0.15, amount_stains=0.4, n_pixels=4096, tissue_threshold=78.4):
        """
        :param amount_matrix, float:
            maximal absolute perturbation of the (unit-norm) stain vectors
        :param amount_stains, float:
            maximal relative perturbation of the stain amounts
        :param n_pixels, int:
            number of randomly chosen pixels used to estimate the stain vectors
        :param tissue_threshold, float:
            pixels with CIE lightness L* (0-100) below this value count as tissue
        """
        super().__init__(
            default_kwargs=dict(amount_matrix=amount_matrix, amount_stains=amount_stains,
                                n_pixels=n_pixels, tissue_threshold=tissue_threshold),
            transform_func=he_staining)
