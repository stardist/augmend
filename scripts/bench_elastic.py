import numpy as np
from augmend import Elastic
from time import time
import argparse

def bench(shape, axis = None, use_gpu = False, workers=1, niter=10):
    d = np.empty(shape, np.float32)

    f = Elastic(axis = axis, use_gpu=use_gpu)

    # warm up
    f(d)

    t = time()
    for _ in range(niter):
        f(d)
    t = (time()-t)/niter
    return t




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--number', type  =int, default = 6,
                        help='maximal length of dimension 2^n')

    args = parser.parse_args()

    ns = 2**np.arange(3,args.number+1)

    t1 = np.array(tuple(bench((n,)*3,use_gpu=False, niter=1) for n in ns))
    t2 = np.array(tuple(bench((n,) * 3, use_gpu=True) for n in ns))


    import thesis
    import matplotlib
    import matplotlib.pyplot as plt
    plt.ion()
    thesis.setup(usetex = False)
    fig = thesis.figure(.8, num=1)
    fig.clf()
    ax = fig.add_axes((0.2,.2,0.7,.7))
    # ax.plot(ns,1000*t1, label = "CPU")
    # ax.plot(ns, 1000*t2, label="GPU")
    # ax.set_ylabel("time (ms)")
    ax.plot(ns,1./t1, label = "CPU (single thread)")
    ax.plot(ns, 1./t2, label="GPU (Titan X)")
    ax.set_ylabel("throughput (volumes/s)")
    
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("cube side (px)")
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title("Elastic deformation of cube volumes (float32)")
    plt.grid()
    plt.legend()
    plt.show()
