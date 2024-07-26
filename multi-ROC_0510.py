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
# file_path = r'172lasso.xlsx'
# file_path1 = r'D:/slicer/Slicer 4.11.20210226/python/APgpc3-Taining.xlsx'
df_org = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='ALL-CRR')

X = df_org.iloc[:, :-1]  # 取出数据
y = df_org.iloc[:, -1]  # 取出标签
print(X)
print(y)
# 训练集和测试集拆分之后
# file_path1 = r'ARG1/AParg1-lasso-train.xlsx'
# file_path2 = r'ARG1/AParg1-lasso-test.xlsx'
# df_org1 = pd.read_excel(file_path1)
# df_org2 = pd.read_excel(file_path2)
# X_train = df_org1.iloc[:, :-1]  # 取出数据
# y_train = df_org1.iloc[:, -1]  # 取出标签
# X_test = df_org2.iloc[:, :-1]  # 取出数据
# y_test = df_org2.iloc[:, -1]  # 取出标签
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)
# 数据拆分
print('------------------------- 数据拆分 -------------------------')
# 拆分成训练集和测试集    参数 test_size 是测试集的比例  0.2  -> 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=42)
# 可以先用pandas先保存看下结果
# X1 = X_test
# Y1 = y_test
# F = (X1,Y1)
# F.to_excel('XYTspss.xlsx', index=False)
X_train.to_excel('XTr.xlsx', index=False)
X_test.to_excel('XTe.xlsx', index=False)
y_train.to_excel('YTr.xlsx', index=False)
y_test.to_excel('YTe.xlsx', index=False)
# # # 数据压缩
print('------------------------- 数据压缩 -------------------------')
#exit()
# scaler = preprocessing.StandardScaler()  # 标准化压缩  符合正态分布
scaler = preprocessing.MinMaxScaler()  # 归一化压缩   0-1之间

scaler.fit(X_train)  # 压缩操作

# 得到压缩的数据
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # 保证数据分布存在一致性

print('\n X_train:\n', X_train)
print('\n X_test:\n', X_test)
print('\n y_train:\n', y_train)
print('\n y_test:\n', y_test)

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
param_grid = {'C': range(15, 20)}  # SVM、LR
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

clf.fit(X_train, y_train)

best_estimator = clf.best_estimator_  # 最佳模型

best_estimator.fit(X_train, y_train)

print('best_score：%f'% clf.best_score_)
print('最好的参数:')

for key in clf.best_params_.keys():
    print('%s = %s'%(key,clf.best_params_[key]))

exit()
y_pred = best_estimator.predict(X_test)
y_score: object = best_estimator.predict_proba(X_test)[:, 1]
y_pred1 = best_estimator.predict(X_train)
y_score1 = best_estimator.predict_proba(X_train)[:, 1]
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
with open('Resulttest/report1.txt', 'w', encoding='utf-8') as fw:
    fw.write(metrics.classification_report(y_test, y_pred))
    fw.write(metrics.classification_report(y_train, y_pred1))

# 混淆矩阵
plt.clf()
metrics.plot_confusion_matrix(best_estimator, X_test, y_test)
plt.savefig('Resulttest/confusion_matrix.jpg')
plt.clf()
metrics.plot_confusion_matrix(best_estimator, X_train, y_train)
plt.savefig('Resulttest/confusion_matrix1.jpg')

# ROC 曲线
plt.clf()
test_disp = metrics.plot_roc_curve(best_estimator, X_train, y_train, name='train')
train_disp = metrics.plot_roc_curve(best_estimator, X_test, y_test, name='test', ax=test_disp.ax_)

test_disp.figure_.suptitle("ROC Curve Comparison")
plt.savefig('Resulttest/roc_compare.jpg')

# PR 曲线
plt.clf()
test_disp = metrics.plot_precision_recall_curve(best_estimator, X_train, y_train, name='test')
train_disp = metrics.plot_precision_recall_curve(best_estimator, X_test, y_test, name='train', ax=test_disp.ax_)
train_disp.figure_.suptitle("P-R Curve Comparison")
plt.savefig('Resulttest/pr_compare1.jpg')

# print('------------------------- AUC -------------------------')
# def roc_auc_ci(y_test, y_score, positive=1):
#     AUC = roc_auc_score(y_test, y_score)
#     N1 = sum(y_test == positive)
#     N2 = sum(y_test != positive)
#     Q1 = AUC / (2 - AUC)
#     Q2 = 2 * AUC ** 2 / (1 + AUC)
#     from numpy import sqrt
#     SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
#     lower = AUC - 1.96 * SE_AUC
#     upper = AUC + 1.96 * SE_AUC
#     if lower < 0:
#         lower = 0
#     if upper > 1:
#         upper = 1
#     return (lower, upper)
#
#
# AUC_CI_2 = roc_auc_ci(y_test, y_score, positive=1)
# print("AUC_95%CI(2): ")
# print(AUC_CI_2)
#
# def roc_auc_ci(y_train, y_score1, positive=1):
#     AUC = roc_auc_score(y_test, y_score)
#     N1 = sum(y_train == positive)
#     N2 = sum(y_train != positive)
#     Q1 = AUC / (2 - AUC)
#     Q2 = 2 * AUC ** 2 / (1 + AUC)
#     from numpy import sqrt
#     SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
#     lower = AUC - 1.96 * SE_AUC
#     upper = AUC + 1.96 * SE_AUC
#     if lower < 0:
#         lower = 0
#     if upper > 1:
#         upper = 1
#     return (lower, upper)
#
#
# AUC_CI_2 = roc_auc_ci(y_train, y_score1, positive=1)
# print("AUC_95%CI(2): ")
# print(AUC_CI_2)
#
# print('------------------------- DCA -------------------------')
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
#
# def calculate_net_benefit_model(thresh_group, y_score, y_test):
#     net_benefit_model = np.array([])
#     for thresh in thresh_group:
#         y_pred = y_score > thresh
#         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#         n = len(y_test)
#         net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
#         net_benefit_model = np.append(net_benefit_model, net_benefit)
#     return net_benefit_model
#
#
# def calculate_net_benefit_all(thresh_group, y_test):
#     net_benefit_all = np.array([])
#     tn, fp, fn, tp = confusion_matrix(y_test, y_test).ravel()
#     total = tp + tn
#     for thresh in thresh_group:
#         net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
#         net_benefit_all = np.append(net_benefit_all, net_benefit)
#     return net_benefit_all
#
#
# def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
#     #Plot
#     ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
#     ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
#     ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
#
#     #Fill，显示出模型较于treat all和treat none好的部分
#     y2 = np.maximum(net_benefit_all, 0)
#     y1 = np.maximum(net_benefit_model, y2)
#     ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)
#
#     #Figure Configuration， 美化一下细节
#     ax.set_xlim(0,1)
#     ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
#     ax.set_xlabel(
#         xlabel = 'Threshold Probability',
#         fontdict= {'family': 'Times New Roman', 'fontsize': 15}
#         )
#     ax.set_ylabel(
#         ylabel = 'Net Benefit',
#         fontdict= {'family': 'Times New Roman', 'fontsize': 15}
#         )
#     ax.grid('major')
#     ax.spines['right'].set_color((0.8, 0.8, 0.8))
#     ax.spines['top'].set_color((0.8, 0.8, 0.8))
#     ax.legend(loc = 'upper right')
#
#     return ax
#
#
# if __name__ == '__main__':
#     #构造一个分类效果不是很好的模型
#     y_pred = np.arange(0, 1, 0.001)
#     y_lable = np.array([1]*25 + [0]*25 + [0]*450 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*50 + [1]*125)
#
#     thresh_group = np.arange(0,1,0.01)
#     net_benefit_model = calculate_net_benefit_model(thresh_group, y_score, y_test)
#     net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
#     fig, ax = plt.subplots()
#     ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
#     # fig.savefig('fig1.png', dpi = 300)
#     plt.show()

# print('------------------------- DCA1 -------------------------')
# def calculate_net_benefit_model(thresh_group, y_score1, y_train):
#     net_benefit_model = np.array([])
#     for thresh in thresh_group:
#         y_pred1 = y_score1 > thresh
#         tn, fp, fn, tp = confusion_matrix(y_teain, y_pred1).ravel()
#         n = len(y_train)
#         net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
#         net_benefit_model = np.append(net_benefit_model, net_benefit)
#     return net_benefit_model
#
#
# def calculate_net_benefit_all(thresh_group, y_train):
#     net_benefit_all = np.array([])
#     tn, fp, fn, tp = confusion_matrix(y_train, y_train).ravel()
#     total = tp + tn
#     for thresh in thresh_group:
#         net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
#         net_benefit_all = np.append(net_benefit_all, net_benefit)
#     return net_benefit_all
#
#
# def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
#     #Plot
#     ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
#     ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
#     ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
#
#     #Fill，显示出模型较于treat all和treat none好的部分
#     y2 = np.maximum(net_benefit_all, 0)
#     y1 = np.maximum(net_benefit_model, y2)
#     ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)
#
#     #Figure Configuration， 美化一下细节
#     ax.set_xlim(0,1)
#     ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
#     ax.set_xlabel(
#         xlabel = 'Threshold Probability',
#         fontdict= {'family': 'Times New Roman', 'fontsize': 15}
#         )
#     ax.set_ylabel(
#         ylabel = 'Net Benefit',
#         fontdict= {'family': 'Times New Roman', 'fontsize': 15}
#         )
#     ax.grid('major')
#     ax.spines['right'].set_color((0.8, 0.8, 0.8))
#     ax.spines['top'].set_color((0.8, 0.8, 0.8))
#     ax.legend(loc = 'upper right')
#
#     return ax
#
#
# if __name__ == '__main__':
#     #构造一个分类效果不是很好的模型
#     y_score1 = np.arange(0, 1, 0.001)
#     y_label = np.array([1]*25 + [0]*25 + [0]*450 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25+ [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*50 + [1]*125)
#
#     thresh_group = np.arange(0,1,0.01)
#     net_benefit_model = calculate_net_benefit_model(thresh_group, y_score1, y_train)
#     net_benefit_all = calculate_net_benefit_all(thresh_group, y_train)
#     fig, ax = plt.subplots()
#     ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
#     # fig.savefig('fig1.png', dpi = 300)
#     plt.show()