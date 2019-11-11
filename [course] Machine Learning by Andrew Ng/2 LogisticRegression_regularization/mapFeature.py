import numpy as np
# 特征映射程序，

def map_feature(x1, x2): #生成新的多项式特征
    degree = 6

    x1 = x1.reshape((x1.size, 1))
    x2 = x2.reshape((x2.size, 1))
    result = np.ones(x1[:, 0].shape) #result初始为一个列向量 值全为1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            result = np.c_[result, (x1**(i-j)) * (x2**j)]  #不断拼接新的列 扩充特征矩阵 **表示次方

    return result
