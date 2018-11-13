"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from augmend.augmend import Augmend, ElasticAugmenter, FlipRotAugmenter

if __name__ == '__main__':
    img = np.zeros((100,) * 3, np.float32)

    img[-20:,:20, :20] = 1.
    img[30:40, -10:] = .8

    Xs = np.meshgrid(*((np.arange(0, 100),) * 3), indexing="ij")
    R = np.sqrt(np.sum([(X - c) ** 2 for X, c in zip(Xs, (70, 60, 50))], axis=0))

    img[R<20] = 1.4


    aug = Augmend()
    aug.add(ElasticAugmenter(p=1., axis = (0,1,2),amount = 5, order = 1))
    aug.add(FlipRotAugmenter(p=1., axis = (1,2)))

    res = tuple(aug([img] * 4))

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
    w = volshow((img,)+res)
    w.resize(800, 600)
    w.set_colormap("hot")
    w.glWidget.set_background_mode_black(False)
    w.transform.fromTransformData(t)
    w.saveFrame("_plots/out_%03d.png" % 0)
    for i in range(len(res)+1):
        w.transform.setPos(i)
        w.glWidget.refresh()
        w.saveFrame("_plots/out_%03d.png"%i)

