import multiprocessing

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

import augmend as A  # noqa: E402


class _Dataset(Dataset):
    def __init__(self, aug):
        self.aug = aug

    def __len__(self):
        return 8

    def __getitem__(self, i):
        x = np.zeros((4, 4), np.float32)
        return torch.from_numpy(np.ascontiguousarray(self.aug(x)))


def _aug():
    return A.Augmend(A.AdditiveNoise(sigma=1.0), A.FlipRot90() + A.Identity(), A.Flip() | A.Identity())


@pytest.mark.parametrize("ctx", multiprocessing.get_all_start_methods())
def test_dataloader_workers_differ(ctx):
    loader = DataLoader(_Dataset(_aug()), batch_size=4, num_workers=2, multiprocessing_context=ctx)
    epochs = [[b[:, 0, 0].tolist() for b in loader] for _ in range(2)]
    # two batches per epoch, one from each worker
    assert epochs[0][0] != epochs[0][1]
    assert epochs[0] != epochs[1]
