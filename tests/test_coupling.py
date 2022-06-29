"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend import Augmend, Elastic, FlipRot90, Scale, Rotate, IntensityScaleShift, GaussianBlur, AdditiveNoise, CutOut, IsotropicScale, Identity, DropEdgePlanes, RandomCrop

def test_coupling():
    
    np.random.seed(22)
    
    img = np.random.uniform(0,1,(100,200)).astype(np.float32)
    

    transforms = dict(
        identity = Identity(),
        elastic = Elastic(),
        fliprot = FlipRot90(),
        scale = Scale(),
        isoscale = IsotropicScale(),
        rotate = Rotate(),
        intensity = IntensityScaleShift(scale = (0.4,2)),
        gaussian = GaussianBlur(), 
        noise = AdditiveNoise(sigma=.2),
        cutout = CutOut(width = (40,41)),
        dropedges = DropEdgePlanes(width = 20),
        randomcrop = RandomCrop(size=(64, 65))
        )

    for i,(name, t) in enumerate(transforms.items()):
        print(name)
        aug = Augmend()
        aug.add([t,t, [t,t]])

        for _ in range(50):
            x,y,(u,w) = aug([img,img, [img, img]])
            same = np.allclose(x,y) and np.allclose(y,u) and np.allclose(u,w)
            if not same:
                print(name)
            assert same

        
    

    
    

if __name__ == '__main__':
    pass
