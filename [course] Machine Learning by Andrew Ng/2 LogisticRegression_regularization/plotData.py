import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    postive=X[y==1] #取出正样本
    negtive=X[y==0] #取出负样本
    
    plt.scatter(postive[:,0],postive[:,1],marker='x',c='red',label='y=1')
    plt.scatter(negtive[:,0],negtive[:,1],marker='o',c='blue',label='y=0')

