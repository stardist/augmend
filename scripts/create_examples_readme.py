"""
Create the example images of the "Available augmentations" section in Readme.md
(input followed by random outputs).
"""
import argparse
from pathlib import Path

import numpy as np
from imageio.v3 import imwrite

from augmend import (
    AdditiveNoise,
    ColorJitter,
    CutOut,
    DropEdgePlanes,
    DropPlanes,
    Elastic,
    Flip,
    FlipRot90,
    GaussianBlur,
    HEStaining,
    IntensityScaleShift,
    IsotropicScale,
    RandomCrop,
    Rotate,
    Scale,
    Shift,
)
from augmend.data import test_image_he_2d, test_image_nuclei_2d

GRAY = {
    "fliprot90": FlipRot90(),
    "flip": Flip(),
    "rotate": Rotate(),
    "scale": Scale(amount=(0.5, 2)),
    "isotropic_scale": IsotropicScale(amount=(0.5, 2)),
    "shift": Shift(amount=(-30, 30)),
    "elastic": Elastic(grid=5, amount=8),
    "random_crop": RandomCrop(size=(96, 96)),
    "intensity_scale_shift": IntensityScaleShift(scale=(0.5, 2), shift=(-0.2, 0.2)),
    "additive_noise": AdditiveNoise(sigma=0.1),
    "gaussian_blur": GaussianBlur(amount=(1, 4)),
    "cutout": CutOut(width=(10, 30), n=(2, 6)),
    "drop_planes": DropPlanes(n=3, width=6),
    "drop_edge_planes": DropEdgePlanes(width=30),
}

RGB = {
    "color_jitter": ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    "he_staining": HEStaining(),
    "intensity_scale_shift_invert": IntensityScaleShift(scale=(0.5, 1.5), shift=(-0.1, 0.1), axis=(0, 1), invert=True),
}


def nuclei_image(size=192):
    x = test_image_nuclei_2d()[300 : 300 + size, 32 : 32 + size].astype(np.float32)
    lo, hi = np.percentile(x, (1, 99.8))
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    # grid lines make geometric transforms visible
    x[::16] = 0.7
    x[:, ::16] = 0.7
    return x


def he_image(size=192):
    return test_image_he_2d()[50 : 50 + size, 150 : 150 + size].astype(np.float32) / 255


def strip(x, outputs, gap=6):
    """Input followed by outputs on a white canvas, outputs smaller than the input are centered."""
    h, w = x.shape[:2]
    canvas = np.ones((h, (len(outputs) + 1) * (w + gap) - gap) + x.shape[2:], np.float32)
    for i, y in enumerate([x, *outputs]):
        y = np.clip(y, 0, 1)
        dy, dx = (h - y.shape[0]) // 2, (w - y.shape[1]) // 2
        x0 = i * (w + gap) + dx
        canvas[dy : dy + y.shape[0], x0 : x0 + y.shape[1]] = y
    return np.round(255 * canvas).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--outdir", type=Path, default=Path(__file__).parents[1] / "imgs" / "examples")
    parser.add_argument("-n", "--n-samples", type=int, default=3)
    parser.add_argument("-s", "--seed", type=int, default=1)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for transforms, x in ((GRAY, nuclei_image()), (RGB, he_image())):
        for name, t in transforms.items():
            rng = np.random.RandomState(args.seed)
            outputs = [t(x, rng=rng) for _ in range(args.n_samples)]
            imwrite(args.outdir / f"{name}.png", strip(x, outputs))
