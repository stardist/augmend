import numpy as np
import tensorflow as tf
from augmend import Augmend, Elastic, FlipRot90
from augmend.utils import create_test_pattern


if __name__ == '__main__':



    y = create_test_pattern(n_samples=256, shape=(512,512), grid_w=(3,10))
    x = (y + 50*np.random.normal(0,1,y.shape)).astype(np.float32)


    n_workers=tf.data.AUTOTUNE
    
    # define augmentation pipeline
    aug = Augmend()
    aug.add([FlipRot90(axis=(0, 1)),
             FlipRot90(axis=(0, 1))])

    aug.add([Elastic(axis=(0, 1), amount=5, order=1),
             Elastic(axis=(0, 1), amount=5, order=0)])
    
    dataset = tf.data.Dataset.from_tensor_slices((x,y))

    gen = dataset.map(aug.tf_map,num_parallel_calls=n_workers).batch(8)


    from time import time

    count = 0
    t = time()
    for x2,y2 in gen:
        count += len(x2)
        
    t = time()-t

    print(f"time to fetch/augment {count} image pairs of shape {x[0].shape} and {n_workers} thread(s): {t:.2f}s")



    # import matplotlib.pyplot as plt
    # plt.ion()
    # fig, axs = plt.subplots(2,len(x2), num=1)
    # for i,(_x,_y) in enumerate(zip(x2,y2)):
    #     axs[0][i].imshow(_x)
    #     axs[1][i].imshow(_y)
    #     axs[0][i].axis('off')
    #     axs[1][i].axis('off')
