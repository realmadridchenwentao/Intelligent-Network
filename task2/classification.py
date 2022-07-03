import graphviz
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


from sklearn.datasets import make_classification

# 查看各个标签的样本量
from collections import Counter

# 使用imblearn进行随机过采样/SMOTE过采样
from imblearn.over_sampling import RandomOverSampler, SMOTE

# 随机欠采样
from imblearn.under_sampling import RandomUnderSampler


# 编写函数，保证输入输出结构不变
X_train = pd.read_excel(r'data.xlsx', usecols='C:I')
X_train = X_train.values
X_train = X_train[1:]

y = pd.read_excel(r'data.xlsx', usecols='J')
y = y.values
y = y[1:]

y[y == 'WWW'] = 0
y[y == 'MAIL'] = 1
y[y == 'P2P'] = 2
y[y == 'FTP-DATA'] = 3

y = y.astype('int')

# print(X_train)
# print(y)


# 数据标准化
# X_train = preprocessing.scale(X_train)

# 数据归一化
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)

# print(X_train)

# 读入决策树训练数据
# 决策树默认参数
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')



# 使用imblearn进行随机过采样
ros = RandomOverSampler(random_state = 0)
X_train, y = ros.fit_resample(X_train, y)

# SMOTE过采样
X_train, y = SMOTE().fit_resample(X_train, y)

# 显示过采样后每个类别的样本个数
print(Counter(y))

# 随机欠采样
# rus = RandomUnderSampler(random_state = 10)
# X_train, y = rus.fit_resample(X_train, y)


# k折交叉验证
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y, test_size=0.1, random_state=0)  # 划分训练集和验证集，test_size为验证集所占的比例
print('训练集大小：', X_train1.shape, y_train1.shape)  # 训练集样本大小
print('验证集大小：', X_val.shape, y_val.shape)  # 测试集样本大小

# 显示划分训练集验证集后每个类别的样本个数
print(Counter(y_train1))
print(Counter(y_val))

# 决策树
tree_model = tree.DecisionTreeClassifier(criterion='gini', 
                                         max_depth=50,
                                         min_samples_leaf=5, 
                                         min_samples_split=10)

# tree_model = tree.DecisionTreeClassifier(criterion='gini')


# tree_model = LogisticRegression()

# 随机森林
# tree_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=100)

# KNN
# tree_model = KNeighborsClassifier()

# SGDClassifier
# tree_model = SGDClassifier()

# SVC
# tree_model = SVC()

clf = tree_model.fit(X_train1, y_train1)

# print(clf.predict(X_val))
# print(y_val)

scores = cross_val_score(clf, X_train1, y_train1, cv = 20)
print(scores)


# 引入graphviz模块用来导出决策树结构图
fn = ['q1_IAT', 'Med_IAT', 'Min_IAT', 'q3_IAT', 'Max_IAT', 'Var_IAT', 'Mean_IAT']
cn = ['WWW', 'MAIL', 'P2P', 'FTP-DATA']


tree.export_graphviz(clf,
                     out_file="tree.dot",
                     feature_names = fn, 
                     class_names = cn,
                     filled = True)

# 使用dot -Tpng tree.dot -o tree.png命令转换为.png格式的图片

# 测试集
# 待分类数据导入
X_test = pd.read_excel(r'test.xlsx', usecols='A:G')
X_test = X_test.values
X_test = X_test[1:]
# print(X_test.shape)


# 得到分类结果
# print(clf.predict(X_test))


# 导入测试集标签
y_test = pd.read_excel(r'label.xlsx', usecols='A')
y_test = y_test.values
y_test = y_test.astype('int')

# print(y_test)

# 计算测试集准确率
scores_final = clf.score(X_test, y_test)
# print(scores_final)
# print(clf.predict(X_test))
print('测试集准确率：', accuracy_score(y_test, clf.predict(X_test)))


# 结果写入excel表格中
result_list = clf.predict(X_test)
columns = ["outputData"]
dt = pd.DataFrame(result_list, columns=columns)
dt.to_excel("result.xlsx", index = 0)
