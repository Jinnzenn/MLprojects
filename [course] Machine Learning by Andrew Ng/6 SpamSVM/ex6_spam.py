import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm

import processEmail as pe
import emailFeatures as ef

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

'''第1部分 邮件预处理'''
#提取邮件中的单词 并得到这些单词在词汇表中的序号 存储在数组中
print('Preprocessing sample email (emailSample1.txt) ...')

file_contents = open('emailSample1.txt', 'r').read()  # 读入示例邮件内容
word_indices = pe.process_email(file_contents)  # 对邮件内容进行预处理，得到邮件中各单词在词汇表中的序号

print('Word Indices: ')
print(word_indices)

input('Program paused. Press ENTER to continue')

'''第2部分 邮件特征提取'''

#将邮件内容转换为向量 基于之前提取的邮件单词在词汇表中的序号，自此单个邮件处理完成
print('Extracting Features from sample email (emailSample1.txt) ... ')

features = ef.email_features(word_indices)  # 出现词汇对应的编号为1，否则为0

print('Length of feature vector: {}'.format(features.size))  # 特征向量大小
print('Number of non-zero entries: {}'.format(np.flatnonzero(features).size))  # 向量中非0项的数量

input('Program paused. Press ENTER to continue')

'''第3部分 训练线性SVM进行垃圾邮件分类'''

data = scio.loadmat('spamTrain.mat')  # 加载矩阵格式的邮件数据集
X = data['X']  # 输入特征矩阵
y = data['y'].flatten()  # 标签 并转换为一维数组 0/1

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes)')

c = 0.1
clf = svm.SVC(c, kernel='linear')
clf.fit(X, y)  # 训练svm

p = clf.predict(X) # 在训练集上进行预测

print('Training Accuracy: {}'.format(np.mean(p == y) * 100)) # 训练集上的准确率

'''第4部分 在测试集上测试线性svm分类器的准确率'''
# 加载测试集
data = scio.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

print('Evaluating the trained linear SVM on a test set ...')

p = clf.predict(Xtest)  # 在测试集上的预测结果

print('Test Accuracy: {}'.format(np.mean(p == ytest) * 100))  # 测试集上的准确率

input('Program paused. Press ENTER to continue')


'''第5部分 通过训练好的分类器 打印权重最高的前15个词 邮件中出现这些词更容易是垃圾邮件'''

vocab_list = pe.get_vocab_list() #得到词汇表 存在字典中
indices = np.argsort(clf.coef_).flatten()[::-1] #对权重序号进行从大到小排序 并返回
print(indices)

for i in range(15): #打印权重最大的前15个词 及其对应的权重 
    print('{} ({:0.6f})'.format(vocab_list[indices[i]], clf.coef_.flatten()[indices[i]]))

input('ex6_spam Finished. Press ENTER to exit')
