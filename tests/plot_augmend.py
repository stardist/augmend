"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend import Augmend, Elastic, FlipRot

if __name__ == '__main__':
    img = np.zeros((100,) * 3, np.float32)
    img[-20:,:20, :20] = 1.
    img[30:40, -10:] = .8
    Xs = np.meshgrid(*((np.arange(0, 100),) * 3), indexing="ij")
    R = np.sqrt(np.sum([(X - c) ** 2 for X, c in zip(Xs, (70, 60, 50))], axis=0))
    img[R<20] = 1.4

    lbl = np.zeros((100,)*3,np.uint16)
    lbl[R<20] = 200


    def data_gen():
        for i in range(4):
            yield img, lbl


    g = data_gen()


    aug = Augmend()
    aug.add(Elastic(p=1., axis=(0, 1, 2),
                    amount=5,
                    order=lambda x: 0 if x.dtype.type == np.uint16 else 1))

    aug.add(FlipRot(p=1., axis = (1, 2)))

    aug_gen = aug(g)

    res = tuple(aug_gen)


    from spimagine import volshow, Quaternion, TransformData

    t = TransformData(quatRot = Quaternion(0.876446026198029,0.16438093880493296,-0.44472712022471556,-0.0838990973937582), zoom = 1.0,
                             dataPos = 0,
                             minVal = 1e-06,
                             maxVal = 7622.243595903105,
                             gamma= 0.5075,
                             translate = np.array([-0.01742376,  0.1809034 , -0.01120263]),
                             bounds = np.array([-1.,  1., -1.,  1., -1.,  1.]),
                             isBox = True,
                             isIso = False,
                             alphaPow = 0.055,
                             isSlice = False,
                             slicePos = 0,
                             sliceDim = 0
                             )
    res_all = ((img, lbl),) + res


    for i, _imgs in enumerate(zip(*res_all)):

        w = volshow(_imgs)
        w.resize(800, 600)
        w.set_colormap("hot" if i==0 else "coolwarm")
        w.glWidget.set_background_mode_black(False)
        w.transform.fromTransformData(t)

        for j in range(len(_imgs)):
            w.transform.setPos(j)
            w.glWidget.refresh()
            w.saveFrame("_plots/out%d_%03d.png"%(i,j))

