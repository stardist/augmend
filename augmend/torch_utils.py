import torch
from torch.utils.data import Dataset
from collections.abc import Iterable

def map_iter(func, *args, iter_over=(tuple, list)):
    cls=type(args[0])
    if isinstance(args[0], Iterable) and issubclass(cls, iter_over):
        return cls(tuple(map_iter(func,*elem,iter_over=iter_over) for elem in zip(*args)))
    else:
        return func(*args)

def from_tensor(x):
    return x.cpu().numpy()

def to_tensor(x, device=None):
    x = torch.tensor(x.copy())
    if device is not None:
        x = x.to(device)
    return x

class _AugDataWrapper(Dataset):
    def __init__(self, aug, dataset):
        self._dataset = dataset
        self.aug = aug

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        x = map_iter(from_tensor, self._dataset[idx])
        devices = map_iter(lambda x: x.device, self._dataset[idx])
        x = self.aug(x)
        x = map_iter(lambda x, dev: to_tensor(x, device=dev), x, devices)
        return x
