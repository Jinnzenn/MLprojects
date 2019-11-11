import numpy as np


def poly_features(X, p):
   
    X_poly=X[:]#第一列为原始输入特征
     #第2到p列 是原始输入特征的平方到p次方
    for i in range(2,p+1):
        X_poly=np.c_[X_poly,X**i]

    return X_poly