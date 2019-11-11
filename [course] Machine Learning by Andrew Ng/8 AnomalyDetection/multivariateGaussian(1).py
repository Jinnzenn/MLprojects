import numpy as np


def multivariate_gaussian(X, mu, sigma2):
    k = mu.size  #特征数
    
    #如果是基于单元高斯分布的模型  将其sigma2转换为对角矩阵 作为协方差矩阵 代入多元高斯分布公式
    #此时单元模型和多元模型是等价的
    #如果是基于多元高斯分布的模型 直接将计算的协方差矩阵sigma2代入多元高斯分布公式
    if sigma2.ndim == 1 or (sigma2.ndim == 2 and (sigma2.shape[1] == 1 or sigma2.shape[0] == 1)):
        sigma2 = np.diag(sigma2)
    
    x = X - mu #对原始特征矩阵进行均值规范化
    p = (2 * np.pi) ** (-k / 2) * np.linalg.det(sigma2) ** (-0.5) * np.exp(-0.5*np.sum(np.dot(x, np.linalg.pinv(sigma2)) * x, axis=1))

    return p
