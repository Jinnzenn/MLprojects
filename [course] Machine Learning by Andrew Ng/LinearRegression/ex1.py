import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from computeCost import *
from gradientDescent import *
from plotData import *

'''第1部分 可视化训练集'''

# 1.1 读取数据文件
# numpy.loadtxt(fname, dtype=, comments='#', delimiter=None, converters=None,
# skiprows=0, usecols=None, unpack=False, ndmin=0)
# fname,文件名；dtype,数据类型；comments，行开头为#时跳过；delimiter，指明数据间的分隔符号；
# skeprows,跳过开头的几行；uscols，指定列；unpack，是否输出为向量；converters:指定列使用函数处理，比如converter={0:function}
print('Plotting Data...')
data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))#加载txt格式的数据集 每一行以","分隔，
X = data[:, 0]   #输入变量 第一列
y = data[:, 1]   #输出变量 第二列
m = y.size     #样本数

# 1.2 可视化数据集
# plt.ion() 函数意为interactive on，把绘图显示模式由block转为interactive模式，即使在脚本中遇到plt.show()，代码还是会继续执行。
# figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
plt.ion()
plt.figure(0)  # 该语句绘制了一张图片
plot_data(X, y)  # 该语句在图片上绘制散点图，函数原型另行编写，内部含有plt.show()

input('Program paused. Press ENTER to continue')


'''第2部分 梯度下降法'''
print('Running Gradient Descent...')

X = np.c_[np.ones(m), X]  # 输入特征矩阵 前面增加一列1 方便矩阵运算。m是前面求得的样本数
theta = np.zeros(2)  # 初始化两个参数为0  theta_0和theta_1


iterations = 1500  #设置梯度下降迭代次数
alpha = 0.01      #设置学习率

# 计算最开始的代价函数值  并与期望值比较 验证程序正确性
print('Initial cost : ' + str(compute_cost(X, y, theta)) + ' (This value should be about 32.07)')

#使用梯度下降法求解线性回归 返回最优参数 以及每一步迭代后的代价函数值
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent: ' + str(theta.reshape(2)))


# 在数据集上绘制出拟合的直线
# plt.legend()用于显示图例
plt.figure(0)
line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')  # 拟合的直线
plot_data(X[:,1], y)  # 绘制数据集散点图，X[:,1]为输入值，数组所有第1行的第1个数据（第0列为另外加的一列1），y为输出值
plt.legend(handles=[line1])

input('Program paused. Press ENTER to continue')

# 用训练好的参数 预测人口为3.5*1000时 收益为多少  并与期望值比较 验证程序正确性
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
# 用训练好的参数 预测人口为7*1000时 收益为多少  并与期望值比较 验证程序正确性
predict2 = np.dot(np.array([1, 7]), theta)  # 1*theta_0 + 7*theta_1
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))

input('Program paused. Press ENTER to continue')

'''第3部分 可视化代价函数'''
print('Visualizing J(theta0, theta1) ...')


# 以下代码建立网格搜索
theta0_vals = np.linspace(-10, 10, 100)  # 用来创建等差数列 首位元素数值、末位元素数值和数列元素个数
theta1_vals = np.linspace(-1, 4, 100)

xs, ys = np.meshgrid(theta0_vals, theta1_vals)  # 生成网格采样点
J_vals = np.zeros(xs.shape)

# Fill out J_vals
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)

J_vals = np.transpose(J_vals)  # 转置

fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

plt.figure(2)
lvls = np.logspace(-2, 3, 20)
plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())
plt.plot(theta[0], theta[1], c='r', marker="x")
plt.show()
input('ex1 Finished. Press ENTER to exit')
