import numpy as np


def sigmoid(z):
    g = np.zeros(z.size)  # 首先建立一个零向量，因为传入的z也是向量
    
    g=1/(1+np.exp(-z))

    return g
