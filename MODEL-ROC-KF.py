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
file_path = r'171lasso0607.xlsx'
#file_path1 = r'D:/slicer/Slicer 4.11.20210226/python/APg
#
# pc3-Taining.xlsx'
df_org = pd.read_excel(file_path,sheet_name='spss')

X = df_org.iloc[:, :-1]  # 取出数据
y = df_org.iloc[:, -1]  # 取出标签
print(X)
print(y)
#训练集和测试集拆分之后
#file_path1 = r'ARG1/AParg1-lasso-train.xlsx'
#file_path2 = r'ARG1/AParg1-lasso-test.xlsx'
#df_org1 = pd.read_excel(file_path1)
#df_org2 = pd.read_excel(file_path2)
#X_train = df_org1.iloc[:, :-1]  # 取出数据
#y_train = df_org1.iloc[:, -1]  # 取出标签
#X_test = df_org2.iloc[:, :-1]  # 取出数据
#y_test = df_org2.iloc[:, -1]  # 取出标签
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
# 数据拆分
decision_values = []
probas = []
print('------------------------- 数据拆分 -------------------------')
# 拆分成训练集和测试集    参数 test_size 是测试集的比例  0.2  -> 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                stratify=y, random_state=42)
# 可以先用pandas先保存看下结果
#X1 = X_test
#Y1 = y_test
#F = (X1,Y1)
#F.to_excel('XYT.xlsx', index=False)
#X_train.to_excel('XTr.xlsx', index=False)
#X_test.to_excel('XTe.xlsx', index=False)
#y_train.to_excel('YTr.xlsx', index=False)
#y_test.to_excel('YTe.xlsx', index=False)
# 数据压缩

print('------------------------- 数据压缩 -------------------------')
#exit()
# scaler = preprocessing.StandardScaler()  # 标准化压缩  符合正态分布
scaler = preprocessing.MinMaxScaler()  # 归一化压缩   0-1之间


scaler.fit(X_train)  # 压缩操作

# 得到压缩的数据
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) # 保证数据分布存在一致性

print('\n X_train:\n', X_train)
print('\n X_test:\n', X_test)
print('\n y_train:\n', y_train)
print('\n y_test:\n', y_test)
# # 使用自己拆分好的训练集和测试集
# df_train1 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/171testandtrain0620.xlsx',
#     sheet_name='train0725')
#
# df_test1 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/171testandtrain0620.xlsx',
#     sheet_name='test0725')

# 模型训练
print('------------------------- 模型训练 -------------------------')

# estimator = XGBClassifier()  # 选择模型
#estimator = RandomForestClassifier()  # 选择模型
estimator = LogisticRegression()  # 选择模型
#estimator = SVC() # SVC Classifier
#estimator = DecisionTreeClassifier() #
#estimator = KNeighborsClassifier() #
#estimator = SGDClassifier() #

# param_grid = {
#     'n_estimators': [50, 100, 150, 200]
# }


# 自动调参(模型训练）
# 调整的参数越多，速度越慢；效果越好
# 某些参数值 为整数，不要超过200
# param_grid = {
#     'n_estimatores': range(10, 200, 10),  #
#     'max_depth': range(3, 10),  #  3-9 之间
#     'gamma': [0.3, 0.5, 0.4, 0.7, 0.8]
# }

param_grid = {'C': range(1, 10)}

clf = GridSearchCV(estimator, param_grid)

clf.fit(X_train, y_train)

best_estimator = clf.best_estimator_  # 最佳模型

best_estimator.fit(X_train, y_train)

y_pred = best_estimator.predict(X_test)
y_score = best_estimator.predict_proba(X_test)[:, 1]
y_pred1 = best_estimator.predict(X_train)
y_score1 = best_estimator.predict_proba(X_train)[:, 1]
print(y_pred,y_score)
# print(X_train.shape,X_test.shape, y_train.shape, y_test.shape)
#
# # 模型训练
# print('------------------------- 模型评估 -------------------------')
#
# # 评估报告
# with open('result/report3.txt', 'w', encoding='utf-8') as fw:
#     fw.write(metrics.classification_report(y_test, y_pred))
#     fw.write(metrics.classification_report(y_train, y_pred1))
#
# # 混淆矩阵
# plt.clf()
# metrics.plot_confusion_matrix(best_estimator, X_test, y_test)
# plt.savefig('result/confusion_matrix3.jpg')
# plt.clf()
# metrics.plot_confusion_matrix(best_estimator, X_train, y_train)
# plt.savefig('result/confusion_matrix3.jpg')
# # ROC 曲线
# plt.clf()
# metrics.plot_roc_curve(best_estimator, X_test, y_test)
# plt.savefig('result/roc_curve3.jpg')
# plt.clf()
# metrics.plot_roc_curve(best_estimator, X_train, y_train)
# plt.savefig('result/roc_curve3.jpg')
#
# # PR 曲线
# plt.clf()
# metrics.plot_precision_recall_curve(best_estimator, X_test, y_test)
# plt.savefig('result/pr_curve3.jpg')
# plt.clf()
# metrics.plot_precision_recall_curve(best_estimator, X_train, y_train)
# plt.savefig('result/pr_curve3.jpg')
# print('------------------------- 交叉验证 -------------------------')
# # 使用k-fold交叉验证来评估模型性能。
# import numpy as np
# # p次k折交叉验证，更普适
# rkf = RepeatedKFold(n_splits = 10, n_repeats = 2)
# for train_index, test_index in rkf.split(X):
#    X_train = X.iloc[train_index]
#    X_test = X.iloc[test_index]
#    y_train = y.iloc[train_index]
#    y_test = y.iloc[test_index]
#    model_svm = svm.SVC(kernel = 'rbf', gamma=0.9258747122872903, C=1.736182213254837, probability=True).fit(X_train,y_train)
#    score_svm = model_svm.score(X_test,y_test)
#    print(score_svm)
#
#
# #kfold = KFold(n_splits=10, random_state=1, shuffle=True)
# #for train_index, test_index in kfold.split(X):
# #    X_train, X_test = X[train_index], X[test_index]
# #    y_train, y_test = y[train_index], y[test_index]
#     # Follow fitting the classifier
#
# #scores = []
# #for k, (train, test) in enumerate(kfold):
# #    pipe_lr.fit(X_train[train], y_train[train])
# #    score = pipe_lr.score(X_train[test], y_train[test])
# #    scores.append(score)
# #    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k + 1, np.bincount(y_train[train]), score))
#
# #print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# # 计算AUC 95%CI (2)
# print('------------------------- AUC -------------------------')
# # def roc_auc_ci(y_test, y_score, positive=1):
# #     AUC = roc_auc_score(y_test, y_score)
# #     N1 = sum(y_test == positive)
# #     N2 = sum(y_test != positive)
# #     Q1 = AUC / (2 - AUC)
# #     Q2 = 2 * AUC ** 2 / (1 + AUC)
# #     from numpy import sqrt
# #     SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
# #     lower = AUC - 1.96 * SE_AUC
# #     upper = AUC + 1.96 * SE_AUC
# #     if lower < 0:
# #         lower = 0
# #     if upper > 1:
# #         upper = 1
# #     return (lower, upper)
# #
# #
# # AUC_CI_2 = roc_auc_ci(y_test, y_score, positive=1)
# # print("AUC_95%CI(2): ")
# # print(AUC_CI_2)
#
# from dca_utils import plot_decision_curves
#
# probas_arr = np.squeeze(np.array(probas))
# # p_series1, net_benifit1 = decision_curve_analysis(probas_arr, labels, 0, 1, 0.02)
#
# # TP_all = np.sum(labels == 1)
# # TN_all = np.sum(labels == -1)
# # p_series2, net_benifit2 = calculate_net_benefit_all(TP_all, TN_all, 0, 1, 0.02)
#
# prob_list = [probas_arr, ]
# # 第一个参数：分类器的概率输出（不是分类器本身）
# plot_decision_curves(prob_list, ['SVC'], y_test, 0, 1, 0.02, -0.5, 0.6)
#
#
