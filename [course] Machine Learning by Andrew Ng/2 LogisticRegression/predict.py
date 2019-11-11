import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0] #样本数 shape函数 0表示行数 1表示列数

    p = np.zeros(m) #每个样本预测的标签

    p = sigmoid(X.dot(theta))  #每个样本属于正类的概率
    p[p >= 0.5] = 1  #概率大于等于0.5 认为属于正类 标签为1 否则为0
    p[p < 0.5] = 0  # 这句语法比较特别
    return p
