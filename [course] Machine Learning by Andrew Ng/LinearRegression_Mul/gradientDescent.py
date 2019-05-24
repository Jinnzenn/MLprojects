import numpy as np
from computeCost import *




def gradient_descent_multi(X, y, theta, alpha, num_iters):
   
    m = y.size  #样本数
    J_history=np.zeros(num_iters)

    for i in range(0, num_iters):   #num_iters次迭代优化
        theta=theta-(alpha/m)*((h(X,theta)-y).dot(X))  # 每一项都减去误差项,误差项是代价函数的偏导数
        J_history[i] = compute_cost(X, y, theta) #用每一次迭代产生的参数 来计算代价函数值

    return theta, J_history