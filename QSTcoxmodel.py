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

warnings.filterwarnings("ignore")
np.random.seed(49)

"""

多个模型ROC曲线比较

"""

# 训练集路径
# train_path = r'171testandtrain.xlsx'
df_train1 = pd.read_excel('D:\VETCNOM.xlsx', sheet_name='Sheet1')
# df_train1 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CR-train')
# df_train2 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CRR-train')
# df_train3 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='Rtrain')
# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_test1 = pd.read_excel('D:\VETCNOM.xlsx', sheet_name='Sheet2')
# df_test1 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CR-test')
# df_test2 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CRR-test')
# df_test3 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='Rtest')
# # 结果保存目录
save_dir = r'MultiModels'
os.makedirs(save_dir, exist_ok=True)

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
# print('test datasets:\n', df_test1)
# print('test datasets:\n', df_test2)
# 机器学习建模
# 需要比较的模型，如果有已经训练好的参数，把参数加上。
print('-----train-------')
total_models1 = {"LR1":LogisticRegression(C=100),  # LR是图上的图例名称，可以修改
    "DT1":DecisionTreeClassifier(max_depth=5),
    "SVM1": LinearSVC(C=3.0),
    "RF1": RandomForestClassifier(n_estimators=120, max_depth=3),
    "XGBoost1": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
    "KNN1":KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
}
# total_models2 = {
#     "LR2": LogisticRegression(C=100),  # LR是图上的图例名称，可以修改
#     "DT2": DecisionTreeClassifier(max_depth=5),
#     "SVM2": LinearSVC(C=3.0),
#     "RF2": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost2": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     "KNN2": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }
# total_models3 = {
#     "LR3": LogisticRegression(C=2.0),  # LR是图上的图例名称，可以修改
#     "DT3": DecisionTreeClassifier(max_depth=5),
#     "SVM3": LinearSVC(C=3.0),
#     "RF3": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost3": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     "KNN3": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }
#

print('-----test-------')

total_models1 = {
    "LR1": LogisticRegression(C=3.0),  # LR是图上的图例名称，可以修改
    "DT1": DecisionTreeClassifier(max_depth=5),
    "SVM1": LinearSVC(C=3.0),
    "RF1": RandomForestClassifier(n_estimators=120, max_depth=3),
    "XGBoost1": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
    "KNN1": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
}
# total_models2 = {
#     "LR2": LogisticRegression(C=3.0),  # LR是图上的图例名称，可以修改
#     "DT2": DecisionTreeClassifier(max_depth=5),
#     "SVM2": LinearSVC(C=3.0),
#     "RF2": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost2": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     "KNN2": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
#
# }
# total_models3 = {
#     "LR3": LogisticRegression(C=3.0),  # LR是图上的图例名称，可以修改
#     "DT3": DecisionTreeClassifier(max_depth=5),
#     "SVM3": LinearSVC(C=3.0),
#     "RF3": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost3": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     "KNN3": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
#
# }


# ====================== 模型评估 ======================
# 将多个模型在测试集上的ROC曲线绘制在一起
plt.clf()  # 清空画板
for name,estimator in total_models1.items():
    estimator.fit(X_train1, y_train1)
    ax1 = plt.gca()
    metrics.plot_roc_curve(estimator, X_train1, y_train1, name=name, ax=ax1)
# for name,estimator in total_models2.items():
#     estimator.fit(X_train2, y_train2)
#     ax1 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_train2, y_train2, name=name, ax=ax1)
# for name, estimator in total_models3.items():
#     estimator.fit(X_train3, y_train3)
#     ax1 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_train3, y_train3, name=name, ax=ax1)
plt.title("Multi Models ROC Curve Comparison_VETC")
plt.savefig(os.path.join(save_dir, "ROC Curve Comparison_VETC_train.jpg"), dpi=300)  # 保存

plt.clf()  # 清空画板
# for name, estimator in total_models.items():
#     estimator.fit(X_test1, y_test1, X_test2, y_test2)
#     ax = plt.gca()
#     metrics.plot_roc_curve(estimator, X_train1, y_train1,X_train2, y_train2, name=name, ax=ax)

# plt.title("Multi Models ROC Curve Comparison2")
# plt.savefig(os.path.join(save_dir, "ROC Curve Comparison_VETC_test.jpg"), dpi=300)  # 保存
for name, estimator in total_models1.items():
    estimator.fit(X_test1, y_test1)
    ax2 = plt.gca()
    metrics.plot_roc_curve(estimator, X_test1, y_test1, name=name, ax=ax2)
# for name,estimator in total_models2.items():
#     estimator.fit(X_test2, y_test2)
#     ax2 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_test2, y_test2, name=name, ax=ax2)
#
# for name,estimator in total_models3.items():
#     estimator.fit(X_test3, y_test3)
#     ax2 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_test3, y_test3, name=name, ax=ax2)
plt.title("Multi Models ROC Curve Comparison-test_vetc")
plt.savefig(os.path.join(save_dir, "ROC Curve Comparison_VETC_test.jpg"), dpi=300)  # 保存