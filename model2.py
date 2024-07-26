import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn import metrics
import matplotlib.pyplot as plt

"""
二分类模型：乳腺癌预测

使用自己的数据修改的地方：
1. file_path

2. 取数据和标签 
X = df_org.iloc[:, :-1]  # 取出数据
y = df_org.iloc[:, -1]  # 取出标签

3.选择模型
estimator = XGBClassifier()  # 选择模型


"""

# 读取数据
file_path = r'172Sampler.xlsx'
#file_path1 = r'D:/slicer/Slicer 4.11.20210226/python/APg
#
# pc3-Taining.xlsx'
df_org = pd.read_excel(file_path)

X = df_org.iloc[:, :-1]  # 取出数据
y = df_org.iloc[:, -1]  # 取出标签
print(X)
print(y)


#SVM
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
model_svm = svm.SVC(kernel = 'rbf',gamma = 'auto',probability = True).fit(X_train,y_train)
score_svm = model_svm.score(X_test,y_test)
print(score_svm)

#params opt: svm 参数优化
Cs = np.logspace(-1,3,10,base = 2)
gammas = np.logspace(-4,1,50,base = 2)
param_grid = dict(C = Cs, gamma = gammas)
grid = GridSearchCV(svm.SVC(kernel = 'rbf'),param_grid = param_grid, cv = 10).fit(X,y)
print(grid.best_params_)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']

#svm
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
model_svm = svm.SVC(kernel = 'rbf',C = C, gamma = gamma,probability = True).fit(X_train,y_train)
score_svm = model_svm.score(X_test,y_test)
print(score_svm)

# p次k折交叉验证，更普适
rkf = RepeatedKFold(n_splits = 3, n_repeats = 2)
for train_index, test_index in rkf.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    model_svm = svm.SVC(kernel = 'rbf', gamma=0.05,C=1,probability = True).fit(X_train,y_train)
    score_svm = model_svm.score(X_test,y_test)
    print(score_svm)