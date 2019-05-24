import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    
    m = y.size #样本数 
    J_history = np.zeros(num_iters)  #每一次迭代都有一个代价函数值

    for i in range(0, num_iters):   #num_iters次迭代优化
        theta=theta-(alpha/m)*(h(X,theta)-y).dot(X)
        J_history[i] = compute_cost(X, y, theta) #用每一次迭代产生的参数 来计算代价函数值

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #


        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
    