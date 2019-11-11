import numpy as np


def h(theta,x):  #假设函数
    return x.dot(theta)

def linear_reg_cost_function(theta, x, y, lmd):
    
    m = y.size #训练样本数

   
    cost = 0
    grad = np.zeros(theta.shape)

    myh=h(theta,x) #假设函数值
    cost=(myh-y).dot(myh-y)/(2*m)+theta[1:].dot(theta[1:])*(lmd/(2*m)) #注意不惩罚第一个参数
    
    grad=(myh-y).dot(x)/m
    grad[1:]+=(lmd/m)*theta[1:]

    return cost, grad
