import matplotlib.pyplot as plt  # matplotlib.pylot这里并未使用，所以pycharm没有高亮语句
import numpy as np
import scipy.optimize as opt  # 引入高级优化方法
from plotData import *
import costFunction as cf  # 这一句是可以将同一文件目录下的文件简化引入的
import plotDecisionBoundary as pdb
import predict as predict
from sigmoid import *



data = np.loadtxt('ex2data1.txt', delimiter=',') #读取txt文件 每一行以','分隔
X = data[:, 0:2] #前两列为原始输入特征   分别两门考试的成绩
y = data[:, 2]   #第三列是输出变量(标签)  二分类 0/1  1代表通过 0代表未通过

'''第1部分 可视化训练数据集'''
print('Plotting Data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

# 可视化数据集，以下语句将打印到同一张图片中
plt.ion()  # 打开交互模式 interactive on
plot_data(X, y)  # 内部有plt.figure()语句


input('Program paused. Press ENTER to continue')

'''第2部分 计算代价函数和梯度'''


(m, n) = X.shape # m行数=样本数 n列数=原始输入特征数 python的语言特性，可以一次有序地返回多个参数


X = np.c_[np.ones(m), X] #特征矩阵X前加一列1  方便矩阵运算 'c_'表示col，按列合并，'r_'表示row，按行合并

#初始化模型参数为0
initial_theta = np.zeros(n + 1)

# 计算逻辑回归的代价函数和梯度
cost, grad = cf.cost_function(initial_theta, X, y)  # 调用cf文件中的cost_function函数,设定代价函数和梯度计算公式

np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})  # 设置输出格式
# 与期望值进行比较 验证程序的正确性
print('Cost at initial theta (zeros): {}'.format(cost))  # 0参数下的代价函数值
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n{}'.format(grad))  # 0参数下的梯度值
print('Expected gradients (approx): \n-0.1000\n-12.0092\n-11.2628')

# 用非零参数值计算代价函数和梯度
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = cf.cost_function(test_theta, X, y)
#与期望值进行比较 验证程序的正确性
print('Cost at test theta (zeros): {}'.format(cost))  # 非0参数下的代价函数值；语法说明：{}：标记格式化参数 format(变量名）
print('Expected cost (approx): 0.218')
print('Gradient at test theta: \n{}'.format(grad))
print('Expected gradients (approx): \n0.043\n2.566\n2.647')  # 非0参数下的代价函数值

input('Program paused. Press ENTER to continue')


'''第3部分 用高级优化方法fmin_bfgs求解最优参数'''

# 可以把高级优化想像成梯度下降法 只不过不用人工设置学习率
'''
    fmin_bfgs优化函数 第一个参数是计算代价cost的函数 第二个参数是计算梯度grad的函数 参数x0传入初始化的theta值
    maxiter设置最大迭代优化次数
'''


def cost_func(t):  # 单独写一个计算代价的函数  返回cost_function的第0个参数 便于传入给opt.fmin_bfgs
    return cf.cost_function(t, X, y)[0]


def grad_func(t):  # 单独写一个计算梯度的函数 返回梯度值
    return cf.cost_function(t, X, y)[1]


# 运行高级优化方法
theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=initial_theta, maxiter=400, full_output=True, disp = False)

# 打印最优的代价函数值和参数值  与期望值比较 验证正确性
print('Cost at theta found by fmin: {:0.4f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: \n{}'.format(theta))
print('Expected Theta (approx): \n-25.161\n0.206\n0.201')

# 画出决策边界
pdb.plot_decision_boundary(theta, X, y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
input('Program paused. Press ENTER to continue')

'''第4部分 用训练好的分类器进行预测，并计算分类器在训练集上的准确率'''

#假设一个学生 考试1成绩45 考试2成绩85  预测他通过的概率
prob = sigmoid(np.array([1, 45, 85]).dot(theta))  # np.array([a,b,c...])
#与期望值进行比较 验证正确性
print('For a student with scores 45 and 85, we predict an admission probability of {:0.4f}'.format(prob))  # float，精度为小数点后4位
print('Expected value : 0.775 +/- 0.002')

# 计算分类器在训练集上的准确率
p = predict.predict(theta, X)  # 调用predict.py中的predict预测函数
#与期望值进行比较 验证正确性
print('Train accuracy: {}'.format(np.mean(y == p) * 100))  # mean()函数会自动求平均值
print('Expected accuracy (approx): 89.0')

input('ex2 Finished. Press ENTER to exit')
