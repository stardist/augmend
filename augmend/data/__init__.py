"""Example images (same as in stardist.data), stored as compressed numpy arrays."""
from pathlib import Path

import numpy as np

_IMAGES = Path(__file__).parent / "images"


def test_image_nuclei_2d(return_mask=False):
    """Fluorescence microscopy image (uint16, 512x512) and label mask from the 2018 Kaggle DSB challenge.

    Caicedo et al. "Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl." Nature Methods 16.12
    """
    with np.load(_IMAGES / "nuclei_2d.npz") as f:
        img, mask = f["image"], f["mask"]
    return (img, mask) if return_mask else img


def test_image_he_2d():
    """H&E stained RGB image (uint8, 300x500x3) from the Cancer Imaging Archive, https://www.cancerimagingarchive.net"""
    with np.load(_IMAGES / "he_2d.npz") as f:
        return f["image"]
