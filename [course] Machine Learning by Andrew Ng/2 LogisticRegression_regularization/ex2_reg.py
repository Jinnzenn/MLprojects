import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plotData import *
import costFunctionReg as cfr
import plotDecisionBoundary as pdb
import predict as predict
import mapFeature as mf

plt.ion()

data = np.loadtxt('ex2data2.txt', delimiter=',') #加载txt格式训练数据集 每一行用','分隔 
X = data[:, 0:2]  #前两列是原始输入特征（2）
y = data[:, 2]  #最后一列是标签 0/1

plot_data(X, y)  #可视化训练集

plt.xlabel('Microchip Test 1') 
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])#图例
plt.show()
input('Program paused. Press ENTER to continue')

'''第1部分 增加新的多项式特征，计算逻辑回归(正则化)代价函数和梯度'''
X = mf.map_feature(X[:, 0], X[:, 1])

initial_theta = np.zeros(X.shape[1])

lmd = 1 #正则化惩罚项系数

# 计算参数为0时的代价函数值和梯度
cost, grad = cfr.cost_function_reg(initial_theta, X, y, lmd)

#与期望值比较 验证正确性
np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})
print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

input('Program paused. Press ENTER to continue')

test_theta = np.ones(X.shape[1])
# 计算参数非0（1）时的代价函数值和梯度
cost, grad = cfr.cost_function_reg(test_theta, X, y, lmd)
#与期望值比较 验证正确性
print('Cost at test theta: {}'.format(cost))
print('Expected cost (approx): 2.13')
print('Gradient at test theta - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.3460\n 0.0851\n 0.1185\n 0.1506\n 0.0159')

input('Program paused. Press ENTER to continue')


'''第2部分 尝试不同的惩罚系数[0,1,10,100],分别利用高级优化算法求解最优参数，分别计算训练好的分类器在训练集上的准确率，
并画出决策边界
 '''

initial_theta = np.zeros(X.shape[1])

lmd = 10

def cost_func(t):
    return cfr.cost_function_reg(t, X, y, lmd)[0]

def grad_func(t):
    return cfr.cost_function_reg(t, X, y, lmd)[1]

theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=initial_theta, maxiter=400, full_output=True, disp=False)

print('Plotting decision boundary ...')
pdb.plot_decision_boundary(theta, X, y)
plt.title('lambda = {}'.format(lmd))

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()

p = predict.predict(theta, X)

print('Train Accuracy: {:0.4f}'.format(np.mean(y == p) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

input('ex2_reg Finished. Press ENTER to exit')

