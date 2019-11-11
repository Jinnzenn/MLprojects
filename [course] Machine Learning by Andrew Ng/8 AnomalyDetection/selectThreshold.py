import numpy as np


def select_threshold(yval, pval):
    f1 = 0 #f1-score

    best_eps = 0  #最好的阈值参数
    best_f1 = 0#最好的阈值参数对应的f1-score
    y=np.zeros(yval.size) #存放预测值
    for epsilon in np.linspace(np.min(pval), np.max(pval), num=1001): #尝试不同的阈值
        
        #y=1 异常 ；y=0正常
        y[pval<epsilon]=1  #预测为异常的样本
        tp=np.sum([yval[x] for x in range(len(y)) if y[x]])  #true positive
        
        precision=tp/np.sum(y) #查准率
        recall=tp/np.sum(yval)   #召回率
        
        f1=2*precision*recall/(precision+recall)  #f1-score

        if f1 > best_f1: #得到对应最高f1-score的阈值
            best_f1 = f1
            best_eps = epsilon
        y=np.zeros(yval.size)

    return best_eps, best_f1
