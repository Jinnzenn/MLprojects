import numpy as np


def feature_normalize(X):
    mu = np.mean(X, 0)  # 求特征矩阵每一列的均值
    sigma = np.std(X, 0, ddof=1)  # 求特征矩阵每一列的标准差
    X_norm = (X - mu) / sigma  # 对特征矩阵每一列进行缩放

    return X_norm, mu, sigma
