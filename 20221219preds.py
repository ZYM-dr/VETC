import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
import numpy as np
import pandas as pd
import numpy as np
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

warnings.filterwarnings("ignore")
np.random.seed(49)

"""

多个模型ROC曲线比较

"""

# 训练集路径
# train_path = r'171testandtrain.xlsx'
df_train1 = pd.read_excel('20221219preds.xlsx', sheet_name='train')

# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
df_test1 = pd.read_excel('20221219preds.xlsx', sheet_name='test')

# 外部验证集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
df_extest1 = pd.read_excel('20221219preds.xlsx', sheet_name='extest')

# 结果保存目录
save_dir = r'MultiModels'
os.makedirs(save_dir, exist_ok=True)

# 训练集
# df_train = pd.read_excel(train_path)
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
y_train1 = df_train1.iloc[:, -1]  # 训练集标签


# 测试集
# df_test = pd.read_excel(test_path)
X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
y_test1 = df_test1.iloc[:, -1]  # 测试集标签

# 测试集
# df_test = pd.read_excel(test_path)
X_extest1 = df_extest1.iloc[:, :-1]  # 测试集特征
y_extest1 = df_extest1.iloc[:, -1]  # 测试集标签

# 机器学习建模
# 需要比较的模型，如果有已经训练好的参数，把参数加上。
print('-----train-------')
total_models1 = {
    "LR1": LogisticRegression(C=20.0),  # LR是图上的图例名称，可以修改
    "DT1": DecisionTreeClassifier(max_depth=5),
    "SVM1": LinearSVC(C=3.0),
    "RF1": RandomForestClassifier(n_estimators=120, max_depth=3),
    "XGBoost1": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
    "KNN1": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
}



print('-----test-------')
# #
# total_models1 = {
#     "LR1": LogisticRegression(C=3.0),  # LR是图上的图例名称，可以修改
#     "DT1": DecisionTreeClassifier(max_depth=5),
#     "SVM1": LinearSVC(C=3.0),
#     "RF1": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost1": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     "KNN1": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }

print('-----extest-------')
# #
# total_models1 = {
#     "LR1": LogisticRegression(C=3.0),  # LR是图上的图例名称，可以修改
#     "DT1": DecisionTreeClassifier(max_depth=5),
#     "SVM1": LinearSVC(C=3.0),
#     "RF1": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost1": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     "KNN1": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }

#
# ====================== 模型评估 ======================
# 将多个模型在测试集上的ROC曲线绘制在一起
plt.clf()  # 清空画板
# for name,estimator in total_models1.items():
#     estimator.fit(X_train1, y_train1)
#     ax1 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_train1, y_train1, name=name, ax=ax1)

# plt.title("Multi Models ROC Curve Comparisonlr")
# plt.savefig(os.path.join(save_dir, "roc_multi_models-trainlr.jpg"), dpi=300)  # 保存

# plt.clf()  # 清空画板
# for name, estimator in total_models.items():
#     estimator.fit(X_test1, y_test1, X_test2, y_test2)
#     ax = plt.gca()
#     metrics.plot_roc_curve(estimator, X_train1, y_train1,X_train2, y_train2, name=name, ax=ax)

# plt.title("Multi Models ROC Curve Comparison2")
# plt.savefig(os.path.join(save_dir, "roc_multi_models2.jpg"), dpi=300)  # 保存

# for name,estimator in total_models1.items():
#     estimator.fit(X_test1, y_test1)
#     ax2 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_test1, y_test1, name=name, ax=ax2)


# plt.title("Multi Models ROC Curve Comparison-testlr")
# plt.savefig(os.path.join(save_dir, "20221219-ki-67-CRmodeltest.jpg"), dpi=300)  # 保存


print('------------------------- AUC -------------------------')
# # 计算AUC 95%CI
# def bootstrap_auc(clf, X_train, Y_train, X_test, Y_test, nsamples=1000):
#     auc_values = []
#     for b in range(nsamples):
#         idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
#         clf.fit(X_train[idx], Y_train[idx])
#         pred = clf.predict_proba(X_test)[:, 1]
#         roc_auc = roc_auc_score(Y_test.ravel(), pred.ravel())
#         auc_values.append(roc_auc)
#     return np.percentile(auc_values, (2.5, 97.5))
#
# # x_train = X_train1.reset_index(drop=True).values
# # y_train = y_train1.reset_index(drop=True).values
# # x_test = X_train1.reset_index(drop=True).values
# # y_test = y_train1.reset_index(drop=True).values
# #
# # AUC_CI = bootstrap_auc(clf=LogisticRegression(C=3.0), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # AUC_CI1 = bootstrap_auc(clf=SVC(kernel='rbf', probability=True), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # AUC_CI2 = bootstrap_auc(clf=DecisionTreeClassifier(max_depth=5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # AUC_CI3 = bootstrap_auc(clf=RandomForestClassifier(n_estimators=120, max_depth=3), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # AUC_CI4 = bootstrap_auc(clf=XGBClassifier(n_estimators=150, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # AUC_CI5 = bootstrap_auc(clf=KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
#
# x_train = X_test1.reset_index(drop=True).values
# y_train = y_test1.reset_index(drop=True).values
# x_test = X_test1.reset_index(drop=True).values
# y_test = y_test1.reset_index(drop=True).values
# AUC_CI = bootstrap_auc(clf=LogisticRegression(C=3.0), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI1 = bootstrap_auc(clf=SVC(kernel='rbf', probability=True), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI2 = bootstrap_auc(clf=DecisionTreeClassifier(max_depth=5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI3 = bootstrap_auc(clf=RandomForestClassifier(n_estimators=120, max_depth=3), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI4 = bootstrap_auc(clf=XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI5 = bootstrap_auc(clf=KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
#
# print("AUC_95%CI: ")
# # print(AUC_CI)
#
# print(AUC_CI, AUC_CI1, AUC_CI2, AUC_CI3, AUC_CI4, AUC_CI5)
#
# # C=C, gamma=gamma

