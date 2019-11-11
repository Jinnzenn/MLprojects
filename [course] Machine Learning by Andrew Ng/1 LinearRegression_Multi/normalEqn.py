import numpy as np
from numpy.linalg import inv

def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    theta = inv(X.T.dot(X)).dot(X.T).dot(y)  # 矩阵求逆

    return theta
