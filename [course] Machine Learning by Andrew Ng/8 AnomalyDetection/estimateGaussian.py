import numpy as np


def estimate_gaussian(X):
   
    m, n = X.shape #样本数m 特征数n
    
    #每个特征的均值和方差
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    #对特征矩阵的每一列求均值和方差
    mu=np.mean(X,axis=0)
    sigma2=np.var(X,axis=0)
    return mu, sigma2
