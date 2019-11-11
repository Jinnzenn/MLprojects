import numpy as np


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmd):
    #将一维向量 转为矩阵形式
    X = params[0:num_movies * num_features].reshape((num_movies, num_features)) 
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    cost = 0
    X_grad = np.zeros(X.shape)  #存放电影特征向量梯度的矩阵
    theta_grad = np.zeros(theta.shape) #存放用户喜好向量梯度的矩阵


    term1=(X.dot(theta.T)-Y)*(X.dot(theta.T)-Y)
    term1=np.sum(term1*R)/2    #只考虑有评分的数据
    term2=(lmd/2)*((X.dot(X.T)).diagonal().sum()+(theta.dot(theta.T)).diagonal().sum())#正则化惩罚项
    cost=term1+term2
    
    X_grad=((X.dot(theta.T)-Y)*R).dot(theta)+lmd*X
    theta_grad=((X.dot(theta.T)-Y)*R).T.dot(X)+lmd*theta
    grad = np.concatenate((X_grad.flatten(), theta_grad.flatten())) #把所有参数的梯度放在一个一维向量中

    return cost, grad
