import numpy as np
from sigmoid import *

def h(theta,X): #假设函数
    return sigmoid(X.dot(theta))

def cost_function_reg(theta, X, y, lmd):
    m = y.size

    cost = 0
    grad = np.zeros(theta.shape)
    
    myh=h(theta,X) #假设函数值
    term1=-y.dot(np.log(myh))
    term2=(1-y).dot(np.log(1-myh))
    term3=(lmd/(2*m))*(theta[1:].dot(theta[1:])) #不惩罚第一项 therta^2
    cost=(term1-term2)/m+term3

    grad=(myh-y).dot(X)/m
    grad[1:]+=(lmd/m)*theta[1:]  # 不惩罚第一个？

    return cost, grad
