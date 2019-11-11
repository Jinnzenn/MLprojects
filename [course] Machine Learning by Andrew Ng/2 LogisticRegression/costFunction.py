import numpy as np
from sigmoid import *


def h(theta,X): #假设函数
    return sigmoid(np.dot(X,theta))  # 不仅直接相乘，还多了sigmoid()

def cost_function(theta, X, y):
    m = y.size #样本数

   
    cost = 0  # 实例化
    grad = np.zeros(theta.shape)
    
    myh=h(theta,X)  #得到假设函数值
    term1=-y.dot(np.log(myh))
    term2=(1-y).dot(np.log(1-myh))
    cost=(term1-term2)/m  # 代价函数
    
    grad=(myh-y).dot(X)/m  # 梯度，偏微分推导过程略

    return cost, grad
