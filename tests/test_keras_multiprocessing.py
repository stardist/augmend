"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from keras.layers import Input
from keras.models import Model

from augmend import Augmend, Elastic

if __name__ == '__main__':

    N = 64
    X = np.zeros((10,N,N,N,1))

    inp = Input(X.shape[1:])
    model = Model(inp, inp)
    model.compile(loss="mse", optimizer="adam")

    def gen():
        while True:
            yield X[:1],X[:1]

    aug = Augmend()
    aug.add([Elastic(axis = (1,2,3)), Elastic(axis = (1,2,3))])

    train_gen = gen()
    train_gen = aug.flow(train_gen)


    model.fit_generator(train_gen, epochs = 2, steps_per_epoch=30,
                        use_multiprocessing= True,
                        workers=8
                        )