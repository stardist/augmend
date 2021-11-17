import numpy as np
from augmend import Augmend, Elastic, FlipRot90
from augmend.utils import create_test_pattern
import torch 
from torch.utils.data import TensorDataset

def test_torch_data():
    y = create_test_pattern(n_samples=16, shape=(512,512), grid_w=(3,10)).astype(np.int16)
    x = (y + 50*np.random.normal(0,1,y.shape)).astype(np.float32)

    data = TensorDataset(torch.tensor(x),torch.tensor(y))
    
    # define augmentation pipeline
    aug = Augmend()
    
    aug.add([FlipRot90(axis=(0, 1)),
             FlipRot90(axis=(0, 1))])

    aug.add([Elastic(axis=(0, 1), amount=5, order=1),
             Elastic(axis=(0, 1), amount=5, order=0)])
    
    data = aug.torch_wrap(data)
    return data


if __name__ == '__main__':


    data = test_torch_data()
    
    x2, y2 = data[0]

    import matplotlib.pyplot as plt


    plt.ion()

    plt.subplot(1,2,1);
    plt.imshow(x2);
    
    plt.subplot(1,2,2);
    plt.imshow(y2)

    
