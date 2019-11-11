import numpy as np


def h(X,theta):  #线性回归假设函数
    return X.dot(theta)  # 线性代数里的点乘
    
def compute_cost(X, y, theta):
    
    m = y.size #样本数
    cost = 0   #代价函数值，有必要首先赋值为0
    hypo = h(X,theta)  #得到假设函数值  (m,)
    
    cost = (hypo-y).dot(hypo-y)/(2*m)  #计算代价函数值，比较巧妙的用法，计算每一项后再叠加

    return cost

