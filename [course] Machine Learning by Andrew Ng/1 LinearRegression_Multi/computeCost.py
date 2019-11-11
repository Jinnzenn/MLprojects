import numpy as np


def h(X: object, theta: object) -> object:  #线性回归假设函数
    return X.dot(theta)  # 点乘，对应项相乘
    
def compute_cost(X, y, theta):
    
    m = y.size #样本数
    cost = 0   #代价函数值
    myh=h(X,theta)  #得到假设函数值 (m,),向量化表述
    
    cost=(myh-y).dot(myh-y)/(2*m)  #计算代价函数值

    return cost

