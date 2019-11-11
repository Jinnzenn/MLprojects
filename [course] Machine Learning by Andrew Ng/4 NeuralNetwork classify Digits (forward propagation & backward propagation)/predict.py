import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    
    # theta1:25*401 输入层多一个偏置项
    # theta2:10*26  隐藏层多一个偏置项
    m = x.shape[0]  # 样本数
    num_labels = theta2.shape[0]  # 类别数

    x=np.c_[np.ones(m),x]  # 增加一列1   x：5000*401
    p = np.zeros(m)
    z1=x.dot(theta1.T)  # z1:5000*25
    a1=sigmoid(z1)    # a1:5000*25
    a1=np.c_[np.ones(m),a1]  # 增加一列1 a1:5000*26
    z2=a1.dot(theta2.T)  # z2:5000*10
    a2=sigmoid(z2)   # a2:5000*10
    
    p=np.argmax(a2,axis=1) #输出层的10个单元 第一个对应数字1...第十个对应数字0
 
    p+=1  #最大位置+1 即为预测的标签
    return p


