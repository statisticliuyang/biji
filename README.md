# README.md

## 推荐系统

### 阅读文献

*Smooth neighborhood recommender system*

*Scalable Collaborative Ranking for Personalized Prediction*

### 文献详述 -  Smooth neighborhood recommender system

#### 1. 简介：

**什么是推荐系统**

**主要方法与思路**

**当前研究的方向和问题**

#### 2. 正则化隐含因子模型

**模型定义**

**模型缺陷**

#### 改进模型：光滑邻域正则化隐含因子模型

**模型定义**

**对比之前模型的优势分析**

**模型求解**

**理论均方误差**

#### 3. 数值仿真与实例分析

**实例分析**

**数值仿真**

#### 4. 我的任务

### 文献详解 -  Scalable Collaborative Ranking for Personalized Prediction

## NLP-文本分类

### 文献阅读

*Natural language processing for ehr-based computational phenotyping*

*A survey of text classification algorithms. In Mining text data*

*Convolutional neural networks for sentence classification*

*Graph Convolutional Networks for Text Classification*

### 中文新闻分类

#### 经典分类器

##### 1. 预处理

提取标签以及分词：

```python
#writing for text anlysis using NB by liuyang 2020-06-07 to 2020-06-12

#先升级一下Anaconda下python的包
#pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#pip install -U numpy
#pip install -U scikit-learn
#pip install -U sklearn

import pandas as pd
import pkuseg  #cut chinese words (or import jieba)



#set the stopwords

stopwords = pd.read_csv('cn_stopwords.txt', encoding='utf8', names=['stopword'], index_col=False)
stop_list = stopwords['stopword'].tolist()

#load date

cnews_test = pd.read_csv('cnews.test.txt', encoding='utf8', names=['lab','cnews_test'], index_col=False,sep='\t')
cnews_train = pd.read_csv('cnews.train.txt', encoding='utf8', names=['lab','cnews_train'], index_col=False,sep='\t')

cnews_train.head() #look look hh #之后都在这里看



# set the prio model(msra,ctb8,weibo) of cutting words (download in pkuseg ,github)

seg = pkuseg.pkuseg(model_name = "news") 
#file_path = '\Users\hey\Anaconda3\Lib\site-packages\pkuseg\msra'  #windows
#seg = pkuseg.pkuseg(model_name=file_path)

# cut words

cnews_train['cut'] = cnews_train['cnews_train'].apply(lambda x: [i for i in seg.cut(x) if i not in stop_list])
cnews_test['cut'] = cnews_test['cnews_test'].apply(lambda x: [i for i in seg.cut(x) if i not in stop_list])

#cnews_train.head()
```

##### 2. 清洗数据

过滤低频率词以及均匀分布的词

```python
# statistic words
words = []
 
for content in cnews_train['cut']:
    words.extend(content)

from collections import Counter
from pprint import pprint

counter_all = Counter(words)

import numpy as np

cnt_all = np.array(list(counter_all.values()))
words_all = list(counter_all)


# print the 1th to 15th words acrossing counting numbers
pprint(counter_all.most_common(15))


#computer the num or prob of words in different class

def cnt_class (label,cnews_train = cnews_train,counter_all = counter_all):
    cut_class = []

    for i in range(len(cnews_train['lab'])):
        if cnews_train['lab'][i]==label:
            cut_class.extend(cnews_train['cut'][i])

    counter = Counter(cut_class)

    cnt_class = Counter(words)
    for k in counter_all:
        if counter.get(k):
            cnt_class[k]=counter[k]
        else:
            cnt_class[k]=0
        
    cnt_class = np.array(list(cnt_class.values()))
    return cnt_class

#get the name of all lab

lab_cnt = Counter(cnews_train['lab'])
lab_all = list(lab_cnt)
print(lab_all)

#computer the peob

cnt_class_all_p = []

for lab in lab_all:
    cnt_class_all_p.append(cnt_class(label= lab )/cnt_all)
cnt_class_all_p = np.array(cnt_class_all_p)



#computer the var of the distribution of words in different class

def sigm (x):
    s = x-x.sum()/10
    s = sum(s*s)/10
    return s

cnt_class_all_p = pd.DataFrame(cnt_class_all_p)
sig = cnt_class_all_p.apply(lambda x: sigm(x),axis = 0)


#filtering the words we care.

#def which in words is useless

a = np.array(list(sig<0.025))
b = cnt_all<(len(cnews_train['lab'])*0.025)
det = a | b
print(len(det)-sum(det))  #print the number of words we need

#def the dic of words we need

def get_words_f (det,words_all = words_all):
    words_f = []
    for i in range(len(det)):
        if ~det[i]:
            words_f.append(words_all[i])
    return words_f

words_f = get_words_f(det)


words_f[1:15] #view the res #筛选结果还行
```

##### 3. 量化文本信息

监督值

```python
#get the y_train and the t_test as the information of class whichi will be inputting
#they are 1 dim vector which are int type with the number 1 to 9 

lab_num = np.linspace(0, 9, 10, endpoint=True,dtype = 'int')
lab_and_num_dic=dict(zip(lab_all, lab_num))

cnews_train['LabNum'] = cnews_train['lab']
cnews_test['LabNum'] = cnews_test['lab']

def lab_num (x,lab_and_num_dic = lab_and_num_dic):
    x = lab_and_num_dic[x]
    return x

cnews_train['LabNum'] = cnews_train['lab'].apply(lambda x: lab_num(x))
cnews_test['LabNum'] = cnews_test['lab'].apply(lambda x: lab_num(x))

y_train = np.array(cnews_train['LabNum'])
y_test = np.array(cnews_test['LabNum'])
```

词集模型

```python
#using Set Of Words Model for
#getting the X_train and the X_test as the information of the text whichi will be inputting.
#they are len(data) \times len(words dic) dim sparse array(by row) which are int type with the number 0-1.

def get_dic(words_f = words_f):
    A = np.ones(len(words_f),dtype='int')
    words_f_dic = dict(zip(words_f,A))
    return words_f_dic

words_f_dic = get_dic()
#words_f_dic_h = get_dic(words_f = words_f_h)
#words_f_dic_l = get_dic(words_f = words_f_l)

def findwords (x,words_f_dic=words_f_dic):
    y = []
    for i in words_f_dic:
        if i in x:
            y.append(words_f_dic[i])
        else:
            y.append(0)
            
    return y

#transfrom to spares array(csr_matrix)

from scipy import sparse

def get_X (cnews_train = cnews_train , cnews_test = cnews_test , dic = words_f_dic):
    X_train = cnews_train['cut'].apply(lambda x: findwords(x,words_f_dic = dic))
    X_test = cnews_test['cut'].apply(lambda x: findwords(x,words_f_dic = dic))

    X_train = np.array(list(X_train))
    X_train = sparse.csr_matrix(X_train)
    X_test = np.array(list(X_test))
    X_test = sparse.csr_matrix(X_test)
    return X_train , X_test

X_train , X_test = get_X()
```

词袋模型+TF-IDF

```python
#using Set Of Words Model + term frequency–inverse document frequency（TF-IDF） for
#getting the X_train and the X_test as the information of the text whichi will be inputting.
#they are len(data) \times len(words dic) dim sparse array(by row) which are double type.

#for using fun TfidfVectorizer() and HashingVectorizer()
def str_c (x,words_f = words_f):
    y = []
    for i in x:
        if i in words_f:
            y.append(i)
        
    y = " ".join(str(i) for i in y)    
    return y
    
#cnews_train['cut_c'] = cnews_train['cut'].apply(lambda x: str_c(x,words_f = words_f))
#cnews_test['cut_c'] = cnews_test['cut'].apply(lambda x: str_c(x,words_f = words_f))
train_cut_c = cnews_train['cut'].apply(lambda x: str_c(x,words_f = words_f))
test_cut_c = cnews_test['cut'].apply(lambda x: str_c(x,words_f = words_f))


X_w = list(train_cut_c)
X_t = list(test_cut_c)


from sklearn.feature_extraction.text import TfidfVectorizer

#the dim of test set and train set must be same ,
#so we need this dic as the option vocabulary when use TfidfVectorizer()

L = len(words_f)
A = np.linspace(0,L-1 ,L, endpoint=True,dtype = 'int')
the_words_f_dic = dict(zip(words_f,A))

vectorizer_tf = TfidfVectorizer(sublinear_tf=True,token_pattern=r"(?u)\b\w+\b",vocabulary = the_words_f_dic)
X_train_tf = vectorizer_tf.fit_transform(X_w)
X_test_tf = vectorizer_tf.fit_transform(X_t)
```

Hashing + TF-IDF

```python
#using Hashing + TF for
#getting the X_train and the X_test as the information of the text whichi will be inputting.
#they are len(data) \times 2**18 or 2**19 or 2**20 dim sparse array(by row) which are double type.

from sklearn.feature_extraction.text import HashingVectorizer

vectorizer_hs = HashingVectorizer(alternate_sign=False ,n_features=2 ** 18,decode_error='ignore')
X_train_hs = vectorizer_hs.transform(X_w)
X_test_hs = vectorizer_hs.transform(X_t)
```

##### 4. 贝叶斯分类器实现

```python
#def the fun to get and output the res of , score , train time , test time , density and dimensionality

from sklearn import metrics
from sklearn.utils.extmath import density
from time import time

def benchmark(clf,X_train = X_train,y_train = y_train,X_test = X_test,y_test = y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
```

##### 5. 选择量化文本信息的方法

```python
#print BernoulliNB, ComplementNB and MultinomialNB res 
#by train set using Set Of Words Model

results = []

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

print('=' * 80)
print("Naive Bayes （词集模型：one-hot编码向量化文本）")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))
```

```python
#print BernoulliNB, ComplementNB and MultinomialNB res 
#by train set using Set Of Words Model + term frequency–inverse document frequency（TF-IDF）

print('=' * 80)
print("Naive Bayes （词袋模型+IDF：TFIDF向量化文本）")
results.append(benchmark(MultinomialNB(alpha=.01),X_train = X_train_tf,X_test = X_test_tf))
results.append(benchmark(BernoulliNB(alpha=.01),X_train = X_train_tf,X_test = X_test_tf))
results.append(benchmark(ComplementNB(alpha=.1),X_train = X_train_tf,X_test = X_test_tf))
```

```python
#print BernoulliNB, ComplementNB and MultinomialNB res 
#by train set using Hashing + TF Model

print('=' * 80)
print("Naive Bayes （HashingTF模型文本向量化）")
results.append(benchmark(MultinomialNB(alpha=.01),X_train = X_train_hs,X_test = X_test_hs))
results.append(benchmark(BernoulliNB(alpha=.01),X_train = X_train_hs,X_test = X_test_hs))
results.append(benchmark(ComplementNB(alpha=.1),X_train = X_train_hs,X_test = X_test_hs))
```

##### 6. 选择清洗数据的强度

```python
#find the best strength for filtering the words we need.
def NB_out (det_n):
    t0 = time()
    train_time = time() - t0
    words_f_n = get_words_f(det_n)
    
    a = cnews_train['cut'].apply(lambda x: str_c(x,words_f = words_f_n))
    b = cnews_test['cut'].apply(lambda x: str_c(x,words_f = words_f_n))
    
    X_w_n = list(a)
    X_t_n = list(b)
    
    L = len(words_f_n)
    A_n = np.linspace(0,L-1 ,L, endpoint=True,dtype = 'int')
    the_words_f_dic = dict(zip(words_f_n,A_n))
    
    vectorizer = TfidfVectorizer(sublinear_tf=True,token_pattern=r"(?u)\b\w+\b",vocabulary = the_words_f_dic)
    X_train_n = vectorizer.fit_transform(X_w_n)
    X_test_n = vectorizer.fit_transform(X_t_n)

    y = benchmark(MultinomialNB(alpha=.01),X_train = X_train_n,X_test = X_test_n)
    t_1 = time() - t0
    
    print("total time:  %0.3fs" % t_1)
    return y,t_1
```

```python
results_1 = []

a = np.array(list(sig<0.035))
b = cnt_all<(len(cnews_train['lab'])*0.035)
det_h_1 = a | b

results_1.append(NB_out(det_h_1))

a = np.array(list(sig<0.03))
b = cnt_all<(len(cnews_train['lab'])*0.03)
det_h = a | b
results_1.append(NB_out(det_h))

results_1.append(NB_out(det))

a = np.array(list(sig<0.02))
b = cnt_all<(len(cnews_train['lab'])*0.02)
det_l = a | b
results_1.append(NB_out(det_l))

a = np.array(list(sig<0.015))
b = cnt_all<(len(cnews_train['lab'])*0.015)
det_l_1 = a | b
results_1.append(NB_out(det_l_1))
```

##### 7. 用不同的分类器输出结果

```python
#print res of different method

#by using the model we choised to get the information of the text
#Set Of Words Model + term frequency–inverse document frequency（TF-IDF）

#and using the best strength for filtering the words:
#np.array(list(sig<0.02)) or cnt_all<(len(cnews_train['lab'])*0.02)

#get the train set

words_f_l = get_words_f(det_l)
a = cnews_train['cut'].apply(lambda x: str_c(x,words_f = words_f_l))
b = cnews_test['cut'].apply(lambda x: str_c(x,words_f = words_f_l))

#cnews_train['cut_f'] = a
#cnews_test['cut_f'] = b
#cnews_train.to_csv(path_or_buf="cnews_train_A.csv",encoding='utf-8')
#cnews_test.to_csv(path_or_buf="cnews_test_A.csv",index=False,encoding='utf-8')
#cnews_train_r = pd.read_csv("cnews_train_A.csv")
#cnews_test_r = pd.read_csv("cnews_test_A.csv")


X_w_l = list(a)
X_t_l = list(b)
    
L_l = len(words_f_l)
A_l = np.linspace(0,L_l-1 ,L_l, endpoint=True,dtype = 'int')
words_f_l_dic = dict(zip(words_f_l,A_l))
    
vectorizer = TfidfVectorizer(sublinear_tf=True,token_pattern=r"(?u)\b\w+\b",vocabulary = words_f_l_dic)
X_train_l = vectorizer.fit_transform(X_w_l)
X_test_l = vectorizer.fit_transform(X_t_l)
```

```python
#最高准确率为0.949
results_2 = []

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
    print('=' * 80)
    print(name)
    results_2.append(benchmark(clf,X_train = X_train_l,X_test = X_test_l))


from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
    
for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results_2.append(benchmark(LinearSVC(penalty=penalty, dual=False,tol=1e-3),X_train = X_train_l,X_test = X_test_l))

    # Train SGD model
    results_2.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,penalty=penalty),X_train = X_train_l,X_test = X_test_l))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results_2.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,penalty="elasticnet"),X_train = X_train_l,X_test = X_test_l))


from sklearn.neighbors import NearestCentroid

# Train Nearest Centroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results_2.append(benchmark(NearestCentroid(),X_train = X_train_l,X_test = X_test_l))


# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results_2.append(benchmark(MultinomialNB(alpha=.01),X_train = X_train_l,X_test = X_test_l))
results_2.append(benchmark(BernoulliNB(alpha=.01),X_train = X_train_l,X_test = X_test_l))
results_2.append(benchmark(ComplementNB(alpha=.1),X_train = X_train_l,X_test = X_test_l))

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
#from sklearn.svm import LinearSVC

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results_2.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))]),X_train = X_train_l,X_test = X_test_l))
```

```python
#add plt
#The bar plot indicates the accuracy, training time (normalized) and test time (normalized) of each classifier.
import matplotlib.pyplot as plt

indices = np.arange(len(results_2))

results_3 = [[x[i] for x in results_2] for i in range(4)]

clf_names, score, training_time, test_time = results_3
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
```

![res](D:\LiuYangStatistics\code\NLPTextClass\TextClassfer_classcal\res.png)

#### 卷积神经网络CNN文本分类

[GitHub 代码库](https://github.com/NLPxiaoxu/Easy_TextCnn_Rnn)

#### 循环神经网络RNN文本分类(LSTM长短记忆)

[GitHub 代码库](https://github.com/NLPxiaoxu/Easy_TextCnn_Rnn)
$$
c_i = f_i \otimes c_{i-1} + i_i \otimes tanh(W_i x_i + U_i h_{i-1} +b_c)
$$



#### 融合LSTM-CNN文本分类

​	RNN网络在文本分类中，作用是用来提取句子的关键语义信息，根据提取的语义对文本进行区分；CNN的作用是用来提取文本的特征，根据特征进行分类。LSTM+CNN的作用，就是两者的结合，首先抽取文本关键语义，然后对语义提取关键特征。

##### 模型结构

![5ae2ccd60001754204630501](C:\Users\LiuYangstatistic\Pictures\Saved Pictures\5ae2ccd60001754204630501.jpg)

输入层

LSTM层

卷积层

池化层

输出层

##### 实现

[GitHub 代码库](https://github.com/NLPxiaoxu/Easy_Lstm_Cnn)

数据预处理

```python
#encoding:utf-8
from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import jieba
import pandas as pd

#_my_function_______________________________________________________________________________

def read_file(filename):
    df = pd.read_csv(filename, encoding='utf8', index_col=False,sep='\t',header = None,names=['lab','cut'])
    labels = list(df['lab'])
    contents = list(df['cut'])

    for i in range(len(contents)) :
        contents[i] = str(contents[i])
        contents[i] = contents[i].split()
    
    return labels ,contents

#___________________________________________________________________________________________

def my_built_vocab_vector(filename,voc_size = 9334):
    all_data = []
    j = 1
    embeddings = np.zeros([9334, 100])

    labels, content = read_file(filename)
    for eachline in content:
        line =[]
        for i in range(len(eachline)):
            line.append(eachline[i])
        all_data.extend(line)

    counter = Counter(all_data)
    count_paris = counter.most_common(voc_size-1)
    word, _ = list(zip(*count_paris))

    f = codecs.open('./data/vector_word.txt', 'r', encoding='utf-8')
    vocab_word = open('./data/data_a/vocab_word.txt', 'w', encoding='utf-8')
    for ealine in f:
        item = ealine.split(' ')
        key = item[0]
        vec = np.array(item[1:], dtype='float32')
        if key in word:
            embeddings[j] = np.array(vec)
            vocab_word.write(key.strip('\r') + '\n')
            j += 1
    np.savez_compressed('./data/data_a/vector_word.npz', embeddings=embeddings)
#___________________________________________________________________________________________

def his_read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        word.extend(jieba.lcut(blk))
                contents.append(word)
            except:
                pass
    return labels, contents

def built_vocab_vector(filenames,voc_size = 10000):
    
    stopword = open('./data/stopwords.txt', 'r', encoding='utf-8')
    stop = [key.strip(' \n') for key in stopword]

    all_data = []
    j = 1
    embeddings = np.zeros([10000, 100])

    for filename in filenames:
        labels, content = read_file(filename)
        for eachline in content:
            line =[]
            for i in range(len(eachline)):
                if str(eachline[i]) not in stop:#去停用词
                    line.append(eachline[i])
            all_data.extend(line)

    counter = Counter(all_data)
    count_paris = counter.most_common(voc_size-1)
    word, _ = list(zip(*count_paris))

    f = codecs.open('./data/vector_word.txt', 'r', encoding='utf-8')
    vocab_word = open('./data/vocab_word.txt', 'w', encoding='utf-8')
    for ealine in f:
        item = ealine.split(' ')
        key = item[0]
        vec = np.array(item[1:], dtype='float32')
        if key in word:
            embeddings[j] = np.array(vec)
            vocab_word.write(key.strip('\r') + '\n')
            j += 1
    np.savez_compressed('./data/vector_word.npz', embeddings=embeddings)


def get_wordid(filename):
    key = open(filename, 'r', encoding='utf-8')

    wordid = {}
    wordid['<PAD>'] = 0
    j = 1
    for w in key:
        w = w.strip('\n')
        w = w.strip('\r')
        wordid[w] = j
        j += 1
    return wordid


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def process(filename, word_to_id, cat_to_id, max_length=300):
    labels, contents = read_file(filename)

    data_id, label_id = [], []

    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id)
    return x_pad, y_pad

def get_word2vec(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def batch_iter(x, y, batch_size = 64):
    data_len = len(x)
    num_batch = int((data_len - 1)/batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    '''
    np.arange(4) = [0,1,2,3]
    np.random.permutation([1, 4, 9, 12, 15]) = [15,  1,  9,  4, 12]
    '''
    x_shuff = x[indices]
    y_shuff = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield x_shuff[start_id:end_id], y_shuff[start_id:end_id]

def seq_length(x_batch):
    real_seq_len = []
    for line in x_batch:
        real_seq_len.append(np.sum(np.sign(line)))

    return real_seq_len
```

参数设置

```python
# -*- coding: utf-8 -*-
class Parameters(object):

    embedding_dim = 100      #dimension of word embedding
    vocab_size = 9334      #number of vocabulary
    pre_trianing = None      #use vector_char trained by word2vec

    seq_length = 300          #max length of sentence
    num_classes = 10          #number of labels
    hidden_dim = 128        #the number of hidden units
    filters_size = [2, 3, 4]
    num_filters = 128

    keep_prob = 0.5         #droppout
    learning_rate = 1e-3    #learning rate
    lr_decay = 0.9          #learning rate decay
    clip = 5.0              #gradient clipping threshold

    num_epochs = 3          #epochs
    batch_size = 64         #batch_size


    train_filename = './data/train_cut.txt'  #train data  √
    test_filename = './data/test.txt'    #test data  √
    val_filename = './data/cnews.val_a.txt'      #validation data  ×
    vocab_filename = './data/data_a/vocab_word.txt'        #vocabulary  √
    vector_word_filename = './data/vector_word.txt'  #vector_word trained by word2vec  ×
    vector_word_npz = './data/data_a/vector_word.npz'   # save vector_word to numpy file  √
```

建立词向量

```python
from data_processing_my import  my_built_vocab_vector

my_built_vocab_vector('./data/train_cut.txt',voc_size = 9334)
```

训练模型

```python
import os
import tensorflow as tf
from Parameters import Parameters as pm
from data_processing_my import read_category, get_wordid, get_word2vec, process, batch_iter, seq_length
from Lstm_Cnn import Lstm_CNN
```

```python
tensorboard_dir = './tensorboard/Lstm_CNN'
save_dir = './checkpoints/Lstm_CNN_a_2'
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

pm = pm
filenames = [pm.train_filename, pm.test_filename, pm.val_filename]
categories, cat_to_id = read_category()
wordid = get_wordid(pm.vocab_filename)
pm.vocab_size = len(wordid)
pm.pre_trianing = get_word2vec(pm.vector_word_npz)

model = Lstm_CNN()

tf.summary.scalar('loss', model.loss)
tf.summary.scalar('accuracy', model.accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(tensorboard_dir)
saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
writer.add_graph(session.graph)
```

```python
x_train, y_train = process(pm.train_filename, wordid, cat_to_id, max_length=300)
x_test, y_test = process(pm.test_filename, wordid, cat_to_id, max_length=300)
```

```python
for epoch in range(pm.num_epochs):
    print('Epoch:', epoch+1)
    num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
    batch_train = batch_iter(x_train, y_train, batch_size=pm.batch_size)
    for x_batch, y_batch in batch_train:
        real_seq_len = seq_length(x_batch)
        feed_dict = model.feed_data(x_batch, y_batch, real_seq_len, pm.keep_prob)
        _, global_step, _summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step, merged_summary,
                                                                                model.loss, model.accuracy], feed_dict=feed_dict)
        if global_step % 100 == 0:
            test_loss, test_accuracy = model.test(session, x_test, y_test)
            print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                    'test_loss:', test_loss, 'test_accuracy:', test_accuracy)

        if global_step % num_batchs == 0:
            print('Saving Model...')
            saver.save(session, save_path, global_step=global_step)

    pm.learning_rate *= pm.lr_decay
```

计算准确率

```python
import numpy as np
from Lstm_Cnn import Lstm_CNN
import tensorflow as tf
from data_processing import read_category, get_wordid, get_word2vec, process, batch_iter, seq_length
from Parameters import Parameters as pm

def val():

    pre_label = []
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/Lstm_CNN_a')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    val_x, val_y = process(pm.val_filename, wordid, cat_to_id, max_length=pm.seq_length)
    batch_val = batch_iter(val_x, val_y, batch_size=64)
    for x_batch, y_batch in batch_val:
        real_seq_len = seq_length(x_batch)
        feed_dict = model.feed_data(x_batch, y_batch, real_seq_len, 1.0)
        pre_lab = session.run(model.predict, feed_dict=feed_dict)
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label


if __name__ == '__main__':

    pm = pm
    sentences = []
    label2 = []
    categories, cat_to_id = read_category()
    wordid = get_wordid(pm.vocab_filename)
    pm.vocab_size = len(wordid)
    pm.pre_trianing = get_word2vec(pm.vector_word_npz)

    model = Lstm_CNN()
    pre_label, label = val()
    correct = np.equal(pre_label, np.argmax(label, 1))
    accuracy = np.mean(np.cast['float32'](correct))
    print('accuracy:', accuracy)
    #print("预测前10项：", ' '.join(str(pre_label[:10])))
    #print("正确前10项：", ' '.join(str(np.argmax(label[:10], 1))))
```



|    模型   |    模型1   |    模型2   |    模型3   |    模型4   |
| :---------: | :---------: | :---------: | :---------: | :---------: |
|   **分词方法**   | *pkgseg* | *pkgseg* | *pkgseg* | *pkgseg* |
|   **词频过滤**   | *删除低词频* | *删除低词频* | *保留高词频* | *保留高词频* |
|   **词分布过滤**   | *删除小方差词* | *删除小方差词* | *否* | *否* |
|   **字典词数**   | *849* | *849* | *9334* | *9334* |
|   **迭代次数**   | *2* | *3* | *2* | *3* |
|   **测试集准确率**   | *95.33%* | *95.53%* | *96.05%* | *96.96%* |
|   **训练时间**   | - | - | *31m 28s* | - |
|   **测试时间**   | *34.8s* | *35.4s%* | - | *40.0s* |


#### Google AI 研究院FastText文本分类

##### 相关文献

*Bag of Tricks for Efficient Text Classification* (2016)

 	论文主要讲Text Classification，论文对CBOW模型做了一点改变，将模型的输出改为预测的Label而不是预测的单词，从而变成了一个分类模型。再输出层的优化方面，也跟CBOW的做法如出一辙，使用了多层Softmax(Hierarchical Softmax) 和负采样(Negative Sampling)的方法。

*Enriching Word Vectors with Subword Information* (2016)

​	论文主要讲Word Vector，提出了字符级别向量的概念(Character level features for NLP)，提出了类似 Word2Vec 中 Skip-gram 与 CBOW 的模型来训练字符向量。其中也是依靠条件概率公式表示预测词(Character)的出现概率，再通过最大似然估计将概率连乘转化为累加形式，最后使用SGD在最大化概率的同时，更新学习参数。

##### FastText分类模型

​	结构分为三层：

```mermaid
graph TD
A(输入层: 文本中的单词和N-gram Feature的Embedding) -->B(隐藏层: 在这一层进行的操作只是将输入层输入的Embedding Vector进行求平均)	
	B --> C(输出层: 输出文本的类别)
```

其中，输入层的输入为文本中的单词和N-gram Feature的Embedding，实际上是使用一个one-hot向量乘以Embedding存放矩阵得到每个词的Embedding的过程。

​	隐藏层，Hidden Layer，在这一层进行的操作是将输出层输入的Embedding Vector进行求平均(Word2Vec的模型中也把这一层称之为投影层(Project Layer)而并不是隐藏层)。

​	输出层，输出文本的类别。与Word2Vec一样，输出层有三种方法，分别是：Softmax，Hierarchical Softmax和Negative Sampling。其中，Softmax通过Hidden Layer求平均得到的Vector，乘以反变换矩阵，得到长度等于分类数的Vector，再使用Softmax得到概率最高的一类为最终分类结果。而后两种方法则是通过Huffman Tree和Negative Sampling两个trick来节省时间复杂度。

​	其它参数：损失函数为Binary Logistic，二分类时与Cross-Entropy相同。FastText把Softmax的输出看作了每一类别的概率值。Optimizer为SGD。

#####  两种N-gram

​	N-gram是基于统计语言模型的算法。它的基本思想是将文本里面的内容按照字节进行大小为N的滑动窗口操作，形成了长度是N的字节片段序列。每一个字节片段称为gram，对所有gram的出现频度进行统计，并且按照事先设定好的阈值进行过滤，形成关键gram列表，也就是这个文本的向量特征空间，列表中的每一种gram就是一个特征向量维度。

​	Subword n-gram feature(Character n-gram):  字符级别的N-gram. eg: 要将单词“universe”划分为<u，un，ni，iv，ve，er，rs，se>。

​	N-gram feature(Word n-gram)：普通N-gram，以词滑动？？？

​	这两种N-gram同时存在于Fasttext的训练中，其中第一种用于Character Embedding的计算。第二种用于Classification的输入层。

​	Character n-gram首先解决了未登录词的问题，其次对于英语中词根和词缀相同的词，使用Character n-gram可以很好的获取它们之间的相似性。最后一点是论文中提到的，Character n-gram可以更好的来表示土耳其语、芬兰语等形态丰富的语言，对于语料中很少出现的单词也能够有很好的表示。

​	Word n-gram特征的加入，提升了Fasttext获取词序信息的能力，因此在面对复杂的语言表述时，也能够更好的获取文本的语义信息。

##### 搜到的其它技巧

​	首先就是输出层使用的Hierarchical Softmax和Negative Sampling。Hierarchical Softmax使用噪音对比功率(NCE)中的理论，通过将一个多分类转化为多个二分类来实现计算复杂度的降低。为了实现同样的目的，Negative Sampling采用了加权采样的方法来抽样负样本而不是每次计算所有词出现的概率。fastText在计算softmax的时候采用**分层softmax**，这样可以大大提高运行的效率

​	第二点是在存储Character n-gram和Word n-gram时，使用hash map的方式将对应的n-gram信息储存在bucket中，节省了空间复杂度，同时由于hash map寻址方式为直接寻址，也降低了查询的时间复杂度。

​	第三，Fastext在进行训练时，提前计算出Character n-gram和Word n-gram，在训练时直接查询调取，也节省了时间复杂度。

##### 实现

[GitHub 代码库](https://github.com/facebookresearch/fastText/tree/master/python)

处理数据格式(对分词后保存的文件进行格式整理)

```python
import pandas as pd
df_train = pd.read_csv('train_text.txt', encoding='utf8', index_col=False,sep='\t',header = None,names=['lab','cut'])
df_test = pd.read_csv('test_text.txt', encoding='utf8', index_col=False,sep='\t',header = None,names=['lab','cut'])
```

```python
import numpy as np
lab_num = np.linspace(0, 9, 10, endpoint=True,dtype = 'int')

lab_n = []
for i in lab_num:
    a = '__label__' + str(lab_num[i])
    lab_n.append(a)

from collections import Counter
from pprint import pprint

lab_cnt = Counter(df_train['lab'])
lab_all = list(lab_cnt)    
lab_and_num_dic=dict(zip(lab_all, lab_n))

def lab_num (x,lab_and_num_dic = lab_and_num_dic):
    x = lab_and_num_dic[x]
    return x

LabNum_train = df_train['lab'].apply(lambda x: lab_num(x))
LabNum_test = df_test['lab'].apply(lambda x: lab_num(x))

dataset_train = pd.DataFrame(columns = ['lab','cut']) 
dataset_train['lab'] = LabNum_train
dataset_train['cut'] = df_train['cut']

dataset_test = pd.DataFrame(columns = ['lab','cut']) 
dataset_test['lab'] = LabNum_test
dataset_test['cut'] = df_test['cut']

np.savetxt('dataset_train.txt',dataset_train, fmt='%s', delimiter="\t",encoding='utf8')
np.savetxt('dataset_test.txt',dataset_test, fmt='%s', delimiter="\t",encoding='utf8')

```

训练模型并保存

```python
import fasttext

trainDataFile = 'dataset_train.txt'
 
classifier = fasttext.train_supervised(
    input = trainDataFile,
    label_prefix = '__label__',
    dim = 256,
    epoch = 50,
    lr = 1,
    lr_update_rate = 50,
    min_count = 3,
    loss = 'softmax',
    word_ngrams = 2,
    bucket = 1000000)

classifier.save_model("Model.bin")
```

计算测试机模型预测的准确率

```python
testDataFile = 'dataset_test.txt'
 
classifier = fasttext.load_model('Model.bin') 
 
result = classifier.test(testDataFile)
print(result[0])
print(result[1])
print(result[2])
```

结果：

​	词频加分布筛选后的分词数据(字典850词)1000条测试数据的准确率为0.9477。未筛选分词数据1000条测试数据的准确率为0.9609。

#### 贝叶斯网络

文献阅读

*A note minimal d-separation trees for structural learning*

背景知识（参考周志华所著<机器学习>相关内容）

​	贝叶斯网络B = <Graph,Parametr>由表示变量间关系的有向无环图DAG和每个变量所对应的条件概率表CPT构成。

![eg](D:\LiuYangStatistics\code\biji\biji\eg.png)

​    各变量间的独立性是一个十分重要的问题，是进行进一步工作的基础和前提。贝叶斯网的结构有效地表达了各变量间的条件独立性：给定父节点，贝叶斯网假设每个变量与它的非后裔变量独立，于是贝叶斯网B的将各变量的联合分布概率定义为：
$$
P_{B}(x_1 , x_2 , \cdots ,x_d) = \prod^{d}_{i=1}P_B(x_i | \pi_i)=\prod^{d}_{i=1}\theta_{x_i|\pi_i}
$$
以上图为例，联合概率分布为：
$$
P(x_1,x_2,\cdots,x_d)=P(x_1)P(x_2)P(x_3|x_1)P(x_4|x_1,x_2)P(x_5|x_2).
$$
​	在贝叶斯网中，三种基本关系如下图所示。在同父结构中，显然若父节点已知，则两个子节点独立；在顺序结构中，若x已知，则y,z独立；V型结构中情况有所不同，若子节点未知，则两父节点独立，若子节点已知，则两父节点不独立。

![gx](D:\LiuYangStatistics\code\biji\biji\gx.png)

为了分析有向图中所有变量的条件独立性，可以使用有向分离(D-separation)(Probabilistic Networks and Expert Systems,1999;Probabilitic Reasoning in Intelligent Systems: Networks of Plausible Inference,1988)。首先将有向图转化为无向图：

- 找出有向图中所有的V型结构，在父节点间连线(无向)；
- 将所有的边改成无向边。

由此产生的图为“道德图”，其过程为道德化。

![XG](D:\LiuYangStatistics\code\biji\biji\XG.png)

​	若变量x,y可以被变量集合Z在图中分开，即去掉Z后，x,y分属两个连通分支，则称x,y被Z有向分离，x⊥y|Z成立。例中有：
$$
x_3\perp x_4|x_1,x_3 \perp x_2|x_1,x_3 \perp x_5|x_2.
$$

​	贝叶斯网的训练分为两部分，一是网络结构的训练，二是每个节点的概率表。当网络结构已知的时候很简单，只需要以频率代替概率，对训练样本计数就可以得到每个节点的概率表的极大似然估计。当结构网未知的时候情况就比较复杂啦。

​	构造得分函数，来评价一个模型的优劣，得分函数带有强烈的主观倾向，即你希望得到的模型具有什么特点，就需要根据这一特点构造得分函数，在此以最小描述长度(MDL):
$$
s(B|D)=f(\theta)|B|-LL(B|D) \\ D=\{x_1,x_2,\cdots,x_m\}为数据集
$$
得分函数的第一项代表模型规模的大小为描述每个参数所需的编码位数×贝叶斯网的参数个数；第二项为对数似然函数：
$$
LL(B|D)=\sum^{m}_{i=1}logP_B(x_i)
$$
由此可以看出：

- 当f=1时,为AIC；
- 当f=log m / 2时，为BIC；
- 当f=0时，学习任务退化为极大似然估计；
- 若结构网固定，极大似然估计等价于频率值。

​     但是在根据得分函数和观测值训练模型的时候，搜索网络结构是一件十分困难的事情，可以想象一下具有n个节点的贝叶斯网可能具有的结构网络的数量是多少。有两种常用的策略可以使用：

- 贪心法：从一个给定的结构出发，每次删减或增加一条边；
- 约束法：将网络结构限制为某些类型，例如树形，塔型。

可以看出，贪心法对初值过于敏感，在约束法中约束条件的选择影响十分巨大。

​	在贝叶斯网模型训练完成后，在理论上我们若已知一些特定的数据，则可以精确地推断出目标节点的条件概率分布，但在实际操作的过程中过于复杂，可以想象一下，其已经被证明是困难的(The computational complexity of probabilistic inference using Bayesian belief networks,1990)。为此，当贝叶斯网节点较多，连接稠密时，可以采用近似推断的方法求近似解。在此使用吉布斯采样：

![吉布斯](D:\LiuYangStatistics\code\biji\biji\吉布斯.png)

思考如何利用吉布斯采样训练模型，这样的训练方法类似与MCMC方法。

​	事实上，吉布斯分布是在参数空间的子空间上的随机漫步，可以证明吉布斯抽样是平稳分布存在的马尔科夫链，由此间接证明了吉布斯抽样的收敛性，并且其恰巧收敛于P(Q=q |E =e)，其中E为已知的证据变量，Q为需要预测的目标变量(张波等人的<应用随机过程>第四版，李航的<统计机器学习>第二版，周志华的<机器学习>第14章)。

#### 市长电话数据分类

##### LSTM

准确率0.75~0.78

##### FastText

读入数据，整理格式，分离训练集与测试集

```python
import pandas as pd
import numpy as np

def str_replace (x):
    x = x.replace("/"," ")
    return x

def add_LabelIndex (x):
    x = '__label__' + str(x)
    return x

def data_pro (filenames,star = 0, end = 0):
    
    DataSet = pd.DataFrame(columns = ['lab','con']) 
    
    file_seg = open(filenames[0],encoding='utf-8')
    df_seg = pd.read_table(file_seg,header=None,names = ['con'])
    
    file_lab = open(filenames[1],encoding='utf-8')
    df_lab = pd.read_table(file_lab,header=None,names = ['lab'])
    
    df_lab = df_lab[0:len(df_seg['con'])]
    
    if end != 0:
        df_seg = df_seg[star:end]
        df_lab = df_lab[star:end]
    
    print("内容条数: %d ." % len(df_seg['con']))
    print("标签条数：%d ." % len(df_lab['lab']))
    
    DataSet['con'] = df_seg['con'].apply(lambda x: str_replace(x))
    
    DataSet['lab'] = df_lab['lab'].apply(lambda x: add_LabelIndex(x))
    
    np.savetxt(filenames[2],DataSet,fmt='%s',delimiter="\t",encoding='utf8')
    
    return 233

files = ['D:/数据集/训练单位投诉内容分词UTF8.txt','D:/数据集/训练单位代码.txt','DataSet_train.txt']
data_pro(files, end = 50000)
files = ['D:/数据集/训练单位投诉内容分词UTF8.txt','D:/数据集/训练单位代码.txt','DataSet_test.txt']
data_pro(files,star = 50001,end = 55001)
```

训练模型并保存

```python
import fasttext

trainDataFile = 'DataSet_train.txt'
 
classifier = fasttext.train_supervised(
    input = trainDataFile,
    label_prefix = '__label__',
    dim = 128,
    epoch = 5,
    lr = 1,
    lr_update_rate = 100,
    min_count = 10,
    loss = 'hs',
    word_ngrams = 3,
    bucket = 2000000)

classifier.save_model("Model.bin")
```

预测并计算准确率

```python
testDataFile = 'DataSet_test.txt'
 
classifier = fasttext.load_model('Model.bin') 
 
result = classifier.test(testDataFile)
print(result[0])
print(result[1])
print(result[2])
```

准确率0.75~0.78

Group of FastText

```python
import GroupsOfFestText as GF
import time
import numpy as np
```

产生分布律以及日期

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

a,b = GF.get_prob_2dim(16,8,5,6)

sns.heatmap(pd.DataFrame(a), cmap="YlGnBu")
print(b)

print(a)
```

![1](D:\LiuYangStatistics\code\biji\biji\1.png)

```python
a,b = GF.get_prob_2(16,8,4,3)

sns.heatmap(pd.DataFrame(a), cmap="YlGnBu")
print(b)

print(a)
```

![2](D:\LiuYangStatistics\code\biji\biji\2.png)

```python
#分布率产生模块测试
RandomNum = GF.get_RandomNum(year = 16,mounth=11,row = 2,col = 10 ,sigma1 = 1,sigma2 = 1)
```

```python
#抽样数据模块测试
GF.get_ModelsData(2,RandomNum = RandomNum , date='1612',rate = 0.3)
```

```python
#抽样数据模块测试
GF.get_ModelsData(2,RandomNum = RandomNum , date='1612',rate = 0.3)
```

```python
#模型训练模块测试
GF.train_models(2,date='1612')
```

```python
#测试数据整理模块测试
rg = np.linspace(1601, 1612, 12, endpoint=True,dtype = 'int')
GF.get_TestData (rg = rg)
```

```python
#预测以及输出准确率模块测试
pt ,ac= GF.get_NPredict (num = 2,date = '1612',AC = True)
print(pt)
print(ac)
```

```python
#调试参数模块

import GroupsOfFestText as GF
import time

start_time = time.time()

ac = GF.tiaoshi(date = '1609',#预测目标
                year = 16,#已知数据年月
                mounth = 8,
                row = 100,#模型个数  #需调试
                col = 4,#每个模型的数据抽样次数  #需调试
                sigma1 = 1,#时间性  #需调试
                sigma2 = 2,#周期性  #需调试
                rate = 0.3,#模型每次抽样的抽样比例  #需调试
                dim=256,#需调试
                epoch=5,#需调试
                lr=1,#需调试
                lrr=100,#需调试
                min_count=2,#需调试
                word_ngrams=2,#需调试
                bucket=2000000,#需调试
                RS=True,
                jion = False)

end_time = time.time()

print("调试用时:", end_time - start_time)
print("准确率:",ac)
```

准确率0.8~0.81

包中内容

```python
import numpy as np
from sklearn.utils import shuffle as reset
import pandas as pd
import fasttext as ft
#import warnings

def get_prob_2(year,mounth,sigma1 = 1, sigma2 = 1):
    
    a = year
    b = mounth
    
    h1 = [a,a-1,a-2,a-3,a-4,a-5,a-6,a-7,a-8,a-9,a-10,a-11]
    x = np.array([h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1])
    frac_x_d = ((2*3.1415926)**0.5)*sigma1
    frac_x_u_l = -0.5*(((x-a)/sigma1)**2)
    frac_x_u = np.exp(frac_x_u_l)
    prob_x = frac_x_u/frac_x_d
    
    h2 = [b-5,b-4,b-3,b-2,b-1,b,b+1,b+2,b+3,b+4,b+5,b+6]
    y = np.array([h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2])
    y = np.transpose(y)
    frac_y_d = ((2*3.1415926)**0.5)*sigma2
    frac_y_u_l = -0.5*(((y-b-1)/sigma2)**2)
    frac_y_u = np.exp(frac_y_u_l)
    prob_y = frac_y_u/frac_y_d
    
    prob_2 = prob_x*prob_y
    
        
    c = x*100+((y + 12) % 12) #0月代表十二月
    c[c%100==0] =  c[c%100==0]+12
    riqi = a*100+((b + 12) % 12)
    if b == 12 :
        riqi += 12
    
    prob_2[c>riqi] = 0
    
    prob_2 = prob_2/np.sum(prob_2)
    
    return prob_2 ,c

def get_RandomNum_2 (year,mounth,row,col ,sigma1 = 1,sigma2 = 1):
    a , b = get_prob_2(year = year,mounth = mounth ,sigma1 = sigma1,sigma2 = sigma2)
    a = a.flatten()
    b = b.flatten()
    RandomNum = np.random.choice(b,row*col,p = a)
    RandomNum.resize(row,col)
    return RandomNum

def get_prob_2dim (year ,mounth ,sigma1 = 1,sigma2 = 1):
    
    a = year
    b = mounth
    
    h1 = [a,a-1,a-2,a-3,a-4,a-5,a-6,a-7,a-8,a-9,a-10,a-11]
    x = np.array([h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1])
    
    h2 = [b-5,b-4,b-3,b-2,b-1,b,b+1,b+2,b+3,b+4,b+5,b+6]
    y = np.array([h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2])
    y = np.transpose(y)
    
    frac_d = 2*3.1415926*sigma1*sigma2
    frac_u_l = ((x-a)*(y-b)/(sigma1*sigma2))-0.5*((((x-a)**2)/(sigma1**2))+(((y-b)**2)/(sigma2**2)))
    frac_u = np.exp(frac_u_l)
    
    prob_2dim = frac_u/frac_d
    
    c = x*100+((y + 12) % 12) #0月代表十二月
    c[c%100==0] =  c[c%100==0]+12
    riqi = a*100+((b + 12) % 12)
    if b == 12 :
        riqi += 12
    
    prob_2dim[c>riqi] = 0
    
    prob_2dim = prob_2dim/np.sum(prob_2dim)
    return prob_2dim , c


def get_RandomNum (year,mounth,row,col ,sigma1 = 1,sigma2 = 1):
    a , b = get_prob_2dim(year = year,mounth = mounth ,sigma1 = sigma1,sigma2 = sigma2)
    a = a.flatten()
    b = b.flatten()
    RandomNum = np.random.choice(b,row*col,p = a)
    RandomNum.resize(row,col)
    return RandomNum


def get_riqi (x):
    x = x[2:6]
    return int(x)

def str_replace (x):
    x = str(x)
    x = x.replace("/"," ")
    return x

def train_test_split_fun(data, test_size=0.3, shuffle=True, random_state=None):
    if shuffle:
        data = reset(data, random_state=random_state)
    train = data[int(len(data)*test_size):].reset_index(drop = True)
    test  = data[:int(len(data)*test_size)].reset_index(drop = True)
    
    return train, test


def get_ModelsData (num,RandomNum,date,rate = 0.3):
    #df_riqi = pd.read_table(file_riqi,header=None,names = ['NYR'],converters={'NYR':str()})
    df_riqi = pd.read_table('riqi.txt',header=None,names = ['NYR'],dtype=str)
    riqi = df_riqi['NYR'].apply(lambda x: get_riqi(x)).values

    #df = pd.read_table('riqi.txt',header=None,names = ['NYR'],dtype=str)
    file_zh = open('zuhe.txt',encoding='utf-8')#
    df = pd.read_table(file_zh,header=None,names = ['lab','con'])
    df['con'] = df['con'].apply(lambda x: str_replace(x))

    for i in range(0,num):
        data = []
        for j in RandomNum[i,:]:
            Data = df[riqi == j]
            DataSet,_ = train_test_split_fun(Data,test_size = 1-rate)
            data.append(DataSet)
        results=pd.concat(data)
        Name = './models_data/'+date+'/'+str(i+1)+'.txt'
        np.savetxt(Name,results,fmt='%s',delimiter="\t",encoding='utf8')
    return '完成'

def train_models (num,date,dim=128,epoch=5,lr=1,lrr=100,min_count=10,word_ngrams=3,bucket = 2000000):
    for i in range(0,num):
        trainDataFile = './models_data/'+date+'/'+str(i+1)+'.txt'
 
        classifier = ft.train_supervised(
            input = trainDataFile,
            label_prefix = '__label__',
            dim = dim,
            epoch = epoch,
            lr = lr,
            lr_update_rate = lrr,
            min_count = min_count,
            loss = 'hs',
            word_ngrams = word_ngrams,
            bucket = bucket)

        name_save = './models/'+date+'/'+'Model_'+str(i+1)+'.bin'
        classifier.save_model(name_save)
    return '完成'

def get_TestData (rg):
    df_riqi = pd.read_table('riqi.txt',header=None,names = ['NYR'],dtype=str)
    riqi = df_riqi['NYR'].apply(lambda x: get_riqi(x)).values

    file_zh = open('zuhe.txt',encoding='utf-8')#
    df = pd.read_table(file_zh,header=None,names = ['lab','con'])
    df['con'] = df['con'].apply(lambda x: str_replace(x))

    for i in rg:
        DataSet = df[riqi == i]
        Name = './models_test/'+str(i)+'.txt'
        np.savetxt(Name,DataSet,fmt='%s',delimiter="\t",encoding='utf8')
    return '完成'


def get_predict (x,num,date):
    #warnings.filterwarnings('ignore')
    putout = []
    for i in range(0,num):
        name_save = './models/'+date+'/'+'Model_'+str(i+1)+'.bin'
        classifier = ft.load_model(name_save)
        res,_= classifier.predict(x)
        res = res[0]
        res = int(res[9:])
        putout.append(res)

    counts = np.bincount(putout)
    return np.argmax(counts)


def get_NPredict (num,date,pre,AC = False):
    name =  './models_test/'+pre+'.txt'
    file_test = open(name,encoding='utf-8')#
    df = pd.read_table(file_test,header=None,names = ['lab','con'])
    text = list(df['con'])
    
    le = len(text)
    out = np.zeros((num,le))
    
    for i in range(0,num):
        name_save = './models/' + date + '/' + 'Model_' + str(i+1) + '.bin'
        classifier = ft.load_model(name_save)
        res,_= classifier.predict(text)
        for j in range(0,le):
            resj=res[j][0]
            out[i,j] = int(resj[9:])
            
    out = np.transpose(out)
    putout = np.zeros(le)
    for i in range(0,le):
        hp = list(out[i])
        counts = np.bincount(hp)
        putout[i] = np.argmax(counts) 
    
    if AC :
        lab = list(df['lab'])
        lab_num = np.zeros(le)
        tr = np.ones(le)
        for i in range(0,le):
            lab_num[i]=lab[i][9:]
        ac = sum(tr[lab_num==putout])/le
        return putout ,ac
    else :
        return putout
    
def tiaoshi (date,year,mounth,row,col,sigma1,sigma2,rate,dim=128,epoch=5,lr=1,lrr=100,min_count=10,word_ngrams=3,bucket=2000000,RS=True,jion = True):
    if RS :
        if jion :
            RandomNum = get_RandomNum(year = year,
                              mounth=mounth,
                              row = row,
                              col = col,
                              sigma1 = sigma1,
                              sigma2 = sigma2)
        else :
            RandomNum = get_RandomNum_2(year = year,
                               mounth=mounth,
                               row = row,
                               col = col,
                               sigma1 = sigma1,
                               sigma2 = sigma2)
        get_ModelsData(num=row,RandomNum = RandomNum , date=date,rate=rate)
    
    train_models(num = row,
             date = date,
             dim = dim,
             epoch=epoch,
             lr=lr,
             lrr=lrr,
             min_count=min_count,
             word_ngrams=word_ngrams,
             bucket = bucket)
    
    _ ,ac= get_NPredict (num = row,date = date,pre = date,AC = True)
    return ac
```

