import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def validation_curve(X, y, Xval, yval):
    # 尝试不同的lambda值
    lambda_vec = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # 每设置一个lambda值，进行训练，返回此时的训练误差和验证误差
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

    i=0
    for lmd in lambda_vec:
        theta=tlr.train_linear_reg(X, y, lmd)
        error_train[i]=lrcf.linear_reg_cost_function(theta, X, y,0)[0]#注意计算误差时lmd=0
        error_val[i]=lrcf.linear_reg_cost_function(theta, Xval, yval,0)[0]#注意计算误差时lmd=0
        i+=1
   
    return lambda_vec, error_train, error_val
