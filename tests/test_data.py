import numpy as np 
from augmend import Identity, FlipRot90, Data 

np.random.seed(10)

if __name__ == "__main__":
    img = np.zeros((100,100), np.uint8)
    img[:50:8] = 2
    img[:,:50:8] = 2
    img[:50:8,:50:8] = 1


    p = np.stack(np.where(img==1),1)


    t = FlipRot90() 
    t = Identity()

    x = img
    x = Data(array=img, points=p)

    y = t(x)