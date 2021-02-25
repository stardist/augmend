import numpy as np
from augmend import Augmend, Identity, FlipRot90, Elastic, Scale, AdditiveNoise
from concurrent.futures import ThreadPoolExecutor
import argparse
import pytest

@pytest.mark.parametrize('ndim', (2,3))
@pytest.mark.parametrize('seed', (2,42,67,128))
@pytest.mark.parametrize('n_samples', (4,48))
def test_multithreaded(ndim, seed, n_samples):
    np.random.seed(seed)

    def _create_stack():
        x = np.random.randint(0,100,(32,)*ndim).astype(np.uint16)
        return x
        
    x = np.stack([_create_stack() for _ in range(n_samples)])

    aug = Augmend()
    for c in (Identity, FlipRot90, Elastic, Scale, AdditiveNoise):
        aug.add([c(),c()], probability=np.random.uniform(0.5,1))

    np.random.seed(seed)
    a1 = tuple(aug([_x1,_x2]) for _x1,_x2 in zip(x,x) )
    
    np.random.seed(seed)
    with ThreadPoolExecutor(max_workers=16) as e:
        a2 = tuple(e.map(aug, zip(x,x)))

    #ensure that they transformed pairs are equal
    equal_single = all(tuple(all(tuple(np.allclose(*p) for p in zip(*a))) for a in a1))
    equal_multi  = all(tuple(all(tuple(np.allclose(*p) for p in zip(*a))) for a in a2))
    
    print("single -> ", equal_single)
    print("multi  -> ", equal_multi)

    assert equal_single
    assert equal_multi

    
        
        

if __name__ == '__main__':

    test_multithreaded(ndim=2, seed=32,n_samples=32)
        
