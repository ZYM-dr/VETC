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
from sklearn import metrics


# # 训练集路径
# train_path = r'171testandtrain.xlsx'
# df_train1 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
#     sheet_name='CR-train')
# df_train2 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
#     sheet_name='CRR-train')
# df_train3 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
#     sheet_name='Rtrain')
df_train1 = pd.read_excel(
    'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/XSS-DT.xlsx',
    sheet_name='Sheet3')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='11-23-fit.best.lse')
# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_test1 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
#     sheet_name='CR-test')
# df_test2 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
#     sheet_name='CRR-test')
# df_test3 = pd.read_excel(
#     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
#     sheet_name='Rtest')

# 训练集
# df_train = pd.read_excel(train_path)
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
feature_names = df_train1.columns[:-1]
y_train1 = df_train1.iloc[:, -1]  # 训练集标签
class_names = df_train1.columns[-1]
# X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
# y_train2 = df_train2.iloc[:, -1]  # 训练集标签
# X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
# y_train3 = df_train3.iloc[:, -1]  # 训练集标签
# print("\n")
# print('train datasets:\n', df_train1)
# print('train datasets:\n', df_train2)
# print('train datasets:\n', df_train3)
# print(class_names)
# print('y_train2:', y_train2.values.tolist())
# print('y_train3:', y_train3.values.tolist())
# exit()
# 测试集
# df_test = pd.read_excel(test_path)
# X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
# feature_names1 = df_test1.columns[:-1]
# y_test1 = df_test1.iloc[:, -1]  # 测试集标签
# class_names1 = df_test1.columns[-1]
# X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
# y_test2 = df_test2.iloc[:, -1]  # 测试集标签
# X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
# y_test3 = df_test3.iloc[:, -1]  # 测试集标签d
# 模型训练
print('------------------------- 模型训练 -------------------------')
# estimator = XGBClassifier()  # 选择模型
# estimator = RandomForestClassifier()  # 选择模型
# estimator = LogisticRegression()  # 选择模型
# estimator = SVC(probability=True) # SVC Classifier
# estimator = KNeighborsClassifier() #
estimator = DecisionTreeClassifier(criterion='entropy')

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
# param_grid = {'C': range(15, 20)}  # SVM、LR
# #
# param_grid = [{'weights': ['uniform'], 'n_neighbors':[i for i in range(5,11)]}, {'weights': ['distance'],
#          'n_neighbors':[i for i in range(10,11)],
#          'p':[i for i in range(5, 10)]
#      }]  #KNN
#
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3,4],
    # 'min_samples_split': [4, 5, 6, 7, 8, 9, 12, 16, 20, 24]
    'min_samples_split': [2,3,4,5,],
    'min_samples_leaf': [1,2,3,4,5,6,7,8],
# 'min_weight_fraction_leaf': [0,0.1,0.2,0.3,0.5],
    # 'max_features': ['none',2,3,4,5,],
                   # 'random_state' : ['none',2,3,4,5,],
                                  # 'max_leaf_nodes' : ['none',2,3,4,5,],
                                  #                  'min_impurity_decrease' : [0,1,2,3,4,5,],
                                  #                                          'class_weight': ['balanced',2,3,4,5,],
                                  #                                                         'ccp_alpha' : [0,0.1,0.2,0.3,0.4,0.5]

}  # DT

# clf = GridSearchCV(estimator, param_grid)
# clf = GridSearchCV(estimator, param_grid, n_jobs=-1, verbose=2) #KNN
clf = GridSearchCV(estimator, param_grid, scoring='roc_auc', cv=5)  # ,scoring='roc_auc', cv=4  DT

clf.fit(X_train1, y_train1)
# exit()
best_estimator = clf.best_estimator_  # 最佳模型

best_estimator.fit(X_train1, y_train1)

print('best_score：%f'% clf.best_score_)
print('最好的参数:')

for key in clf.best_params_.keys():
    print('%s = %s'%(key,clf.best_params_[key]))

# exit()
# y_pred = best_estimator.predict(X_train1)
# y_score: object = best_estimator.predict_proba(X_train1)[:, 1]
# y_pred1 = best_estimator.predict(X_train1)
# y_score1 = best_estimator.predict_proba(X_train1)[:, 1]
# data = pd.DataFrame(y_score)
# writer = pd.ExcelWriter('score.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'Sheet2', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()![](U-M-KNNpr_compare1.jpg)
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
# with open('reporttestCR-DT.txt', 'w', encoding='utf-8') as fw:
#     fw.write(metrics.classification_report(y_test1, y_pred))
# with open('reporttrainCR-DT.txt', 'w', encoding='utf-8') as fw:
#     fw.write(metrics.classification_report(y_train1, y_pred1))

# 混淆矩阵
# plt.clf()
# metrics.plot_confusion_matrix(best_estimator, X_test1, y_test1)
# plt.savefig('confusion_matrixextestCR-DT.jpg')
# plt.clf()
# metrics.plot_confusion_matrix(best_estimator, X_train1, y_train1)
# plt.savefig('confusion_matrix1trainCR-DT.jpg')
# from sklearn import metrics
# print('训练集准确率：', round(metrics.accuracy_score(y_train1, y_pred1)*100, 2), '%') # 训练集
# print('测试集准确率：', round(metrics.accuracy_score(y_test1, y_pred)*100, 2), '%')# 预测集

# ROC 曲线
# plt.clf()
# test_disp = metrics.plot_roc_curve(best_estimator, X_train1, y_train1, name='train')
# train_disp = metrics.plot_roc_curve(best_estimator, X_test1, y_test1, name='test', ax=test_disp.ax_)
#
# test_disp.figure_.suptitle("ROC Curve Comparison")
# plt.savefig('Resulttest/roc_compare-DT.jpg')

# 决策树可视化
import graphviz
import pydotplus
from six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image

# 文件缓存
dot_data = StringIO()
# 将决策树导入到dot中
export_graphviz(best_estimator, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=False, feature_names=feature_names, class_names=True)
# 将生成的dot文件生成graph
# print(X_train1)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# 将结果存入到png文件中
# graph.write_png('diabetes2.png')
# graph.write_pdf('diabetes2.pdf')
# 显示
Image(graph.create_png())

graph.write_jpg("QST-DecisionTree2.jpg")
# exit()
# graph.write_pdf("XSS-DecisionTree2.pdf")
