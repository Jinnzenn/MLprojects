import numpy as np


def estimate_gaussian(X):
   
    m, n = X.shape #样本数m 特征数n
    

    mu = np.zeros(n)
    sigma2 = np.zeros(n)

  
    mu=np.mean(X,axis=0) #计算每个特征的均值
    sigma2=np.var(X,axis=0) #对于基于单元高斯分布的异常检测模型 直接计算每个特征的方差 返回一维数组
    #sigma2=(1/m)*(X.T.dot(X)) #对于基于多元高斯分布的异常检测模型 计算协方差矩阵
    return mu, sigma2
