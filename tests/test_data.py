import numpy as np 
from augmend import Data, FlipRot90


if __name__ == "__main__":
    img = np.zeros((100,100))
    img[::8] = 1


    t = FlipRot90() 


    x = Data(array=img)

    t(x)