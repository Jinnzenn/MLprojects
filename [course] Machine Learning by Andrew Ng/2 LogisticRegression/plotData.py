import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    postive=X[y==1]  #分离正样本
    negtive=X[y==0]  #分离负样本
    
    plt.scatter(postive[:,0],postive[:,1],marker='+',c='red',label='Admitted') # 绘制散点图专用，画出正样本
    plt.scatter(negtive[:,0],negtive[:,1],marker='o',c='blue',label='Not Admitted') #画出负样本
    plt.axis([30, 100, 30, 100])  #设置x,y轴的取值范围
    plt.legend(['Admitted', 'Not admitted'], loc=1)  #设置图例
    plt.xlabel('Exam 1 score')   #x轴标题  考试1成绩
    plt.ylabel('Exam 2 score')   #y轴标题  考试2成绩
    plt.show()
