import numpy as np


def email_features(word_indices):
    
    n = 1899 #词汇表中词的数量  特征向量大小

  
    features = np.zeros(n + 1) #n+1 单词序号从1开始 不+1的话 出现序号为1899的单词 就会越界
    
    features[word_indices]=1 

    features=features[1:]
    return features
