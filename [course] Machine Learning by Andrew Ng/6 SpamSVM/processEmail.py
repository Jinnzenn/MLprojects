import numpy as np
import re
import nltk, nltk.stem.porter


def process_email(email_contents):
    vocab_list = get_vocab_list()  # 得到词汇字典

    word_indices = np.array([], dtype=np.int64)  # 用来存储邮件中的单词在词汇标中的序号
    word_list=[]
    # 邮件预处理
   
    email_contents = email_contents.lower() # 将邮件内容转换为小写
    # 去除所有html标签
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # 把邮件内容中任何数字替换为number
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # 把邮件中任何以http或https开头的url 替换为httpadddr
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # 把邮件中任何邮箱地址替换为 emailaddr
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # 把邮件中的$符号 替换为dollar
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # 对邮件分词

    print('==== Processed Email ====')

    stemmer = nltk.stem.porter.PorterStemmer()

    # print('email contents : {}'.format(email_contents))

    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)

    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)

        if len(token) < 1:  # 跳过单字
            continue
        if token in vocab_list.values():
            word_list.append(list(vocab_list.keys())[list(vocab_list.values()).index(token)])
        
        print(token)

    print('==================')
    word_indices = np.array(word_list,dtype=np.int64)

    return word_indices


def get_vocab_list():  #得到词汇表
    vocab_dict = {}   #以字典形式获取，实例化
    with open('vocab.txt') as f:  #打开txt格式的词汇表，这个是预先给好的
        for line in f:
            (val, key) = line.split()  #读取每一行的键和值
            vocab_dict[int(val)] = key #存放到字典中

    return vocab_dict
