import pandas as pd
import numpy as np
from numpy import array
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
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import train_test_split

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
# 训练集路径
# train_path = r'171testandtrain.xlsx'
df_train1 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-train')
# df_train2 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CRR-train')
# df_train3 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='Rtrain')
# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
df_test1 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-test')
# df_test2 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CRR-test')
# df_test3 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='Rtest')

# 训练集
# df_train = pd.read_excel(train_path)
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
y_train1 = df_train1.iloc[:, -1]  # 训练集标签
# X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
# y_train2 = df_train2.iloc[:, -1]  # 训练集标签
# X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
# y_train3 = df_train3.iloc[:, -1]  # 训练集标签
# print('train datasets:\n', df_train1)
# print('train datasets:\n', df_train2)

# 测试集
# df_test = pd.read_excel(test_path)
X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
y_test1 = df_test1.iloc[:, -1]  # 测试集标签
# X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
# y_test2 = df_test2.iloc[:, -1]  # 测试集标签
# X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
# y_test3 = df_test3.iloc[:, -1]  # 测试集标签
print('------------------------- 数据压缩 -------------------------')
#exit()
# scaler = preprocessing.StandardScaler()  # 标准化压缩  符合正态分布
scaler = preprocessing.MinMaxScaler()  # 归一化压缩   0-1之间

scaler.fit(X_train1)  # 压缩操作

# 得到压缩的数据
X_train1 = scaler.transform(X_train1)
X_test1 = scaler.transform(X_test1)  # 保证数据分布存在一致性

# print('\n X_train:\n', X_train)
# print('\n X_test:\n', X_test)
# print('\n y_train:\n', y_train)
# print('\n y_test:\n', y_test)

# 模型训练
print('------------------------- 模型训练 -------------------------')
# estimator = XGBClassifier()  # 选择模型
# estimator = RandomForestClassifier()  # 选择模型
estimator = LogisticRegression()  # 选择模型
# estimator = SVC(probability=True) # SVC Classifier
# estimator = KNeighborsClassifier() #
# estimator = DecisionTreeClassifier(criterion='entropy')

#
# param_grid = {
#     'n_estimators': [100,200, 150, 200]
# }  #RF、XGB

# 自动调参(模型训练）
# 调整的参数越多，速度越慢；效果越好
# 某些参数值 为整数，不要超过200
# param_grid = {
#     'n_estimators': range(10, 200, 10),  #
#     'max_depth': range(3, 10),  #  3-9 之间
#     'gamma': [0.3, 0.5, 0.4, 0.7, 0.8]
# }
#
param_grid = {'C': range(100, 500)}  # SVM、LR
# #
# param_grid = [{'weights': ['uniform'], 'n_neighbors':[i for i in range(5,11)]}, {'weights': ['distance'],
#          'n_neighbors':[i for i in range(10,11)],
#          'p':[i for i in range(5, 10)]
#      }]  #KNN
#
# param_grid = {
#     'criterion': ['entropy', 'gini'],
#     'max_depth': [4, 5, 6, 7, 8, 9],
#     'min_samples_split': [4, 5, 6, 7, 8, 9, 12, 16, 20, 24]
# }  # DT

clf = GridSearchCV(estimator, param_grid)
# clf = GridSearchCV(estimator, param_grid, n_jobs=-1, verbose=2) #KNN
# clf = GridSearchCV(estimator, param_grid, scoring='roc_auc', cv=10)  # ,scoring='roc_auc', cv=4  DT

clf.fit(X_train1, y_train1)

best_estimator = clf.best_estimator_  # 最佳模型

best_estimator.fit(X_train1, y_train1)

print('best_score：%f'% clf.best_score_)
print('最好的参数:')

for key in clf.best_params_.keys():
    print('%s = %s'%(key,clf.best_params_[key]))

    y_pred = best_estimator.predict(X_test1)
    y_score: object = best_estimator.predict_proba(X_test1)[:, 1]
    y_pred1 = best_estimator.predict(X_train1)
    y_score1 = best_estimator.predict_proba(X_train1)[:, 1]
    # data = pd.DataFrame(y_score)
    # writer = pd.ExcelWriter('score.xlsx')		# 写入Excel文件
    # data.to_excel(writer, 'Sheet2', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    # writer.save()
    # data = pd.DataFrame(y_score1)
    # writer = pd.ExcelWriter('score1.xlsx')		# 写入Excel文件
    # data.to_excel(writer, 'Sheet2', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    # writer.save()
    # exit()
    # print(y_score)
    # print(y_score1)

    # 模型训练
    print('------------------------- 模型评估 -------------------------')

    # 评估报告
    with open('Resulttest/reportlr.txt', 'w', encoding='utf-8') as fw:
        fw.write(metrics.classification_report(y_test1, y_pred))
        fw.write(metrics.classification_report(y_train1, y_pred1))

    # # 混淆矩阵
    # plt.clf()
    # metrics.plot_confusion_matrix(best_estimator, X_test, y_test)
    # plt.savefig('Resulttest/confusion_matrix.jpg')
    # plt.clf()
    # metrics.plot_confusion_matrix(best_estimator, X_train, y_train)
    # plt.savefig('Resulttest/confusion_matrix1.jpg')

    # ROC 曲线
    plt.clf()
    test_disp = metrics.plot_roc_curve(best_estimator, X_train1, y_train1, name='train')
    train_disp = metrics.plot_roc_curve(best_estimator, X_test1, y_test1, name='test', ax=test_disp.ax_)

    test_disp.figure_.suptitle("ROC Curve Comparison")
    plt.savefig('Resulttest/roc_lr.jpg')
