import numpy as np


def feature_normalize(X):
   
    n = X.shape[1]  # shape[1]返回特征矩阵列数 即特征数
    X_norm = X     #初始化特征缩放后的特征矩阵
    mu = np.zeros(n)  #初始化每一列特征的均值
    sigma = np.zeros(n)  #初始化每一列特征的标准差

    mu = np.mean(X, axis=0) #对每一列求均值 axis = 0 表示按照列方向执行方法
    sigma = np.std(X, axis=0) # 对每一列求标准差
    X_norm=(X_norm-mu)/sigma # broadcast功能 每一列减去该列的均值/该列的标准差

    return X_norm, mu, sigma
