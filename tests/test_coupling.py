"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend import Augmend, Elastic, FlipRot90, Scale, Rotate, IntensityScaleShift, GaussianBlur, AdditiveNoise, CutOut, IsotropicScale, Identity, DropEdgePlanes

if __name__ == '__main__':

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
        dropedges = DropEdgePlanes(width = 20)
        )

    for i,(name, t) in enumerate(transforms.items()):
        print(name)
        aug = Augmend()
        aug.add([t,t])

        for _ in range(50):
            x,y = aug([img,img])
            same = np.allclose(x,y)
            if not same:
                print(name)
            assert same


    

    
    
