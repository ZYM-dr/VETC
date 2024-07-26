
import os
import pandas as pd
import sklearn
print(sklearn.__version__)
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
# import shap
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
import numpy as np
# from mlxtend.evaluate import delong_test
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import numpy as np
import xgboost
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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from scikitplot.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore")
np.random.seed(49)

"""

多个模型ROC曲线比较

"""

# 训练集路径
# train_path = r'171testandtrain.xlsx'
# df_train1 = pd.read_excel('D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/XSS-DT.xlsx',
#     sheet_name='CT1_NC_NR')
# df_train2 = pd.read_excel('D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/XSS-DT.xlsx',
#     sheet_name='MT1_NC_NR')
df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Female11-23-fit.best')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='11-23-fit.best.lse')
df_train2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Male11-23-fit.best')
# df_train2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CLIF-SOFA')
# df_train3 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CLIF-OF')
# df_train4 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CLIF-C ACLF')
# df_train5 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-MELD')
# df_train6 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CPScore')
# df_train7 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CPclassification')

# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='2Validation11-23-fit.best.lse')
# df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.min')
# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_test1 = pd.read_excel('D:\VETCNOM.xlsx', sheet_name='Sheet2')
# df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_test')

# 外部验证集路径
df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='FemaleVadation11-23-fit.best.ls')
df_test2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='MaleValidation11-23-fit.best.ls')
# df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.lse')
# df_test2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CLIF-SOFA')
# df_test3 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CLIF-OF')
# df_test4 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CLIF-C ACLF')
# df_test5 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-MELD')
# df_test6 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CPScore')
# df_test7 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CPclassification')
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_extest1 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-extest')
# df_test2 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CRR-test')
# df_test3 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='Rtest')
# 结果保存目录
save_dir = r'QST'
os.makedirs(save_dir, exist_ok=True)

# 训练集
# df_train = pd.read_excel(train_path)
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
y_train1 = df_train1.iloc[:, -1]  # 训练集标签
X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
y_train2 = df_train2.iloc[:, -1]  # 训练集标签
# X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
# y_train3 = df_train3.iloc[:, -1]  # 训练集标签\
# X_train4 = df_train4.iloc[:, :-1]  # 训练集特征
# y_train4 = df_train4.iloc[:, -1]  # 训练集标签
# X_train5 = df_train5.iloc[:, :-1]  # 训练集特征
# y_train5 = df_train5.iloc[:, -1]  # 训练集标签
# X_train6 = df_train6.iloc[:, :-1]  # 训练集特征
# y_train6 = df_train6.iloc[:, -1]  # 训练集标签
# X_train7 = df_train7.iloc[:, :-1]  # 训练集特征
# y_train7 = df_train7.iloc[:, -1]  # 训练集标签
# 测试集
# df_test = pd.read_excel(test_path)
X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
y_test1 = df_test1.iloc[:, -1]  # 测试集标签
X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
y_test2 = df_test2.iloc[:, -1]  # 测试集标签
# X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
# y_test3 = df_test3.iloc[:, -1]  # 测试集标签
# X_test4 = df_test4.iloc[:, :-1]  # 测试集特征
# y_test4 = df_test4.iloc[:, -1]  # 测试集标签
# X_test5 = df_test5.iloc[:, :-1]  # 测试集特征
# y_test5 = df_test5.iloc[:, -1]  # 测试集标签
# X_test6= df_test6.iloc[:, :-1]  # 测试集特征
# y_test6 = df_test6.iloc[:, -1]  # 测试集标签
# X_test7 = df_test7.iloc[:, :-1]  # 测试集特征
# y_test7 = df_test7.iloc[:, -1]  # 测试集标签


# 外部验证集
# df_test = pd.read_excel(test_path)
# X_extest1 = df_extest1.iloc[:, :-1]  # 测试集特征
# y_extest1 = df_extest1.iloc[:, -1]  # 测试集标签

# 机器学习建模
# 需要比较的模型，如果有已经训练好的参数，把参数加上。
print('-----train-------')
# total_models1 = {
# #     # "LR": LogisticRegression(C=20.0),  # LR是图上的图例名称，可以修改
# #     # "DT": DecisionTreeClassifier(max_depth=5),
# #     # "SVM": LinearSVC(C=3.0),
# #     # "RF": RandomForestClassifier(n_estimators=120, max_depth=3),
#       "CHAPT":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5),
# #     "XGBoost": XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5),
# #     # "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }
total_models1 = {"Female_Train":XGBClassifier(n_estimators=9, max_depth=3, gamma=0.5), }
total_models2 = {"Male_Train":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
# total_models2 = {"CLIF-SOFA":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
# total_models3 = {"CLIF-OF":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
# total_models4 = {"CLIF-C ACLF":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
# total_models5 = {"MELD":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
# total_models6 = {"CPScore":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
# total_models7 = {"CPclassification":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }

# print('-----shap------')
# model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train1, label=y_train1), 100)
# # 创建Explainer
# explainer = shap.TreeExplainer(model)
# # 以numpy数组的形式输出SHAP值
# shap_values = explainer.shap_values(X_train1)
#
# # 以SHAP的Explanation对象形式输出SHAP值
# shap_values2 = explainer(X_train1)
#
# # 模型自带重要性
# xgboost.plot_importance(model,height = .5,
#                         max_num_features=10,
#                         show_values = False)
#
# # 图形1：全局条形图summary_plot
# shap.summary_plot(shap_values,X_train1, plot_type="bar")
# shap.plots.bar(shap_values2)
# shap.summary_plot(shap_values,X_train1, plot_type="bar")
# # 图形6：蜂群图（方法1）
# shap.summary_plot(shap_values, X_train1)
#
# # 图形6：蜂群图（方法2）
# # 根据最大SHAP对特征排序
# shap.plots.beeswarm(shap_values2,
#                     order=shap_values2.abs.max(0))
# exit()

# total_models2 = {
#     "XGBoost": XGBClassifier(n_estimators=8, max_depth=4, gamma=0.5),
# }
# total_models3 = {
#     "XGBoost": XGBClassifier(n_estimators=8, max_depth=4, gamma=0.5),
# }
# total_models4 = {
#     "XGBoost": XGBClassifier(n_estimators=8, max_depth=4, gamma=0.5),
# }
# total_models5 = {
#     "XGBoost": XGBClassifier(n_estimators=8, max_depth=4, gamma=0.5),
# }
# total_models6 = {
#     "XGBoost": XGBClassifier(n_estimators=8, max_depth=4, gamma=0.5),
# }
# total_models7 = {
#     "XGBoost": XGBClassifier(n_estimators=8, max_depth=4, gamma=0.5),
# }
# exit()
print('-----test-------')
# total_models8 = {
# #     # "LR1": LogisticRegression(C=20.0),  # LR是图上的图例名称，可以修改
# #     # "cT1": DecisionTreeClassifier(max_depth=3),
# #     # "SVM1": LinearSVC(C=3.0),
# #     # "RF1": RandomForestClassifier(n_estimators=12, max_depth=3),
# #
#     "CHAPT": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),
# #     "CLIF-SOFA": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),
# #     "CLIF-OF": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),
# #     "CLIF-C ACLF": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),
# #     "MELD": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),
# #     "CPScore": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),
# #     "CPclassification": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),
# #     # "KNN1": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }
total_models11 = {"Female_Validation":XGBClassifier(n_estimators=3, max_depth=2, gamma=0.5), }
total_models12 = {"Male_Validation":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models12 = {"CLIF-SOFA":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models13 = {"CLIF-OF":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models14 = {"CLIF-C ACLF":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models15 = {"MELD":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models16 = {"CPScore":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models17 = {"CPclassification":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }


# print('-----2validation-------')
# total_models1 = {
#     "LR": LogisticRegression(C=200.0),  # LR是图上的图例名称，可以修改
#     "DT": DecisionTreeClassifier(max_depth=1),
#     "SVM": LinearSVC(C=3.0),
#     "RF": RandomForestClassifier(n_estimators=8, max_depth=1),
#     "XGBoost": XGBClassifier(n_estimators=2, max_depth=4, gamma=0.5),
#     "KNN": KNeighborsClassifier(n_neighbors=25, weights='uniform', metric_params=None, n_jobs=None)
# }
# model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train1, label=y_train1), 100)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test1)
# shap_values2 = explainer(X_test1)
# shap.plots.beeswarm(shap_values2,
#                     order=shap_values2.abs.max(0))
# xgboost.plot_importance(model,height = .5,
#                         max_num_features=10,
#                         show_values = False)





# ====================== 模型评估 ======================
# 将多个模型在测试集上的ROC曲线绘制在一起
plt.clf()# 清空画板
for name,estimator in total_models1.items():
    estimator.fit(X_train1, y_train1)
    ax1 = plt.gca()
    metrics.RocCurveDisplay.from_estimator(estimator, X_train1, y_train1, name=name, ax=ax1)
for name,estimator in total_models2.items():
    estimator.fit(X_train2, y_train2)
    ax1 = plt.gca()
    metrics.RocCurveDisplay.from_estimator(estimator, X_train2, y_train2, name=name, ax=ax1)
# for name, estimator in total_models3.items():
#     estimator.fit(X_train3, y_train3)
#     ax1 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_train3, y_train3, name=name, ax=ax1)
# for name, estimator in total_models4.items():
#     estimator.fit(X_train4, y_train4)
#     ax1 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_train4, y_train4, name=name, ax=ax1)
# for name, estimator in total_models5.items():
#     estimator.fit(X_train5, y_train5)
#     ax1 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_train5, y_train5, name=name, ax=ax1)
# for name, estimator in total_models6.items():
#     estimator.fit(X_train6, y_train6)
#     ax1 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_train6, y_train6, name=name, ax=ax1)
# for name, estimator in total_models7.items():
#     estimator.fit(X_train7, y_train7)
#     ax1 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_train7, y_train7, name=name, ax=ax1)

#
plt.title("Train")
# plt.savefig(os.path.join(save_dir, "Gender.pdf"), dpi=300)  # 保存
plt.show()

plt.clf()  # 清空画板
for name, estimator in total_models11.items():
    estimator.fit(X_test1, y_test1)
    ax = plt.gca()
    metrics.RocCurveDisplay.from_estimator(estimator, X_test1, y_test1, name=name, ax=ax)
for name, estimator in total_models12.items():
    estimator.fit(X_test2, y_test2)
    ax = plt.gca()
    metrics.RocCurveDisplay.from_estimator(estimator, X_test2, y_test2, name=name, ax=ax)
# for name, estimator in total_models13.items():
#     estimator.fit(X_test3, y_test3)
#     ax = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test3, y_test3, name=name, ax=ax)
# for name, estimator in total_models14.items():
#     estimator.fit(X_test4, y_test4)
#     ax = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test4, y_test4, name=name, ax=ax)
# for name, estimator in total_models15.items():
#     estimator.fit(X_test5, y_test5)
#     ax = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test5, y_test5, name=name, ax=ax)
# for name, estimator in total_models16.items():
#     estimator.fit(X_test6, y_test6)
#     ax = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test6, y_test6, name=name, ax=ax)
# for name, estimator in total_models17.items():
#     estimator.fit(X_test7, y_test7)
#     ax = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test7, y_test7, name=name, ax=ax)
plt.title("Test")
# plt.savefig(os.path.join(save_dir, "Gender.pdf"), dpi=300)  # 保存
plt.show()
exit()


# print("SEN-SPE-ACC-NPV-PPV ")
# x_train = X_test2.reset_index(drop=True).values
# y_train = y_test2.reset_index(drop=True).values
# x_test = X_test2.reset_index(drop=True).values
# y_test = y_test2.reset_index(drop=True).values
# # Dictionary to store the evaluation results
# evaluation_results = {}
# # for model_name, model in total_models1.items():
# for model_name, model in total_models8.items():
#     # Ensure the number of features is consistent
#     if x_test.shape[1] != x_train.shape[1]:
#         raise ValueError(f"Number of features in X_test ({x_test.shape[1]}) is different from X_train ({x_train.shape[1]}) for {model_name}")
#
#     # Fit the model
#     model.fit(x_train, y_train)
#
#     # Special handling for models without predict_proba
#     if hasattr(model, "predict_proba"):
#         y_prob = model.predict_proba(x_test)[:, 1]
#     elif hasattr(model, "decision_function"):
#         y_prob = model.decision_function(x_test)
#     else:
#         raise AttributeError(f"{model_name} does not have a predict_proba or decision_function method.")
#
#     # Calculate AUC
#     auc = roc_auc_score(y_test, y_prob)
#
#     # Calculate predictions
#     y_pred = model.predict(x_test)
#
#     # Calculate confusion matrix
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#
#     # Calculate sensitivity, specificity, accuracy, NPV, and PPV
#     # Find the threshold that maximizes sensitivity + specificity
#     thresholds = np.linspace(0, 1, 100)
#     best_threshold = None
#     best_sensitivity = 0
#     best_specificity = 0
#
#     for threshold in thresholds:
#         y_pred = (y_prob > threshold).astype(int)
#
#         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#
#         sensitivity = tp / (tp + fn)
#         specificity = tn / (tn + fp)
#
#         if sensitivity + specificity > best_sensitivity + best_specificity:
#             best_sensitivity = sensitivity
#             best_specificity = specificity
#             best_threshold = threshold
#             # Store results in the dictionary
#
#
#     # Calculate accuracy, NPV, and PPV at the best threshold
#     y_pred_best_threshold = (y_prob > best_threshold).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best_threshold).ravel()
#
#     accuracy = (tp + tn) / (tp + tn + fp + fn)
#     npv = tn / (tn + fn)
#     ppv = tp / (tp + fp)
#
#     # Store results in the dictionary
#     evaluation_results[model_name] = {
#         'AUC': auc,
#         'Best Sensitivity': best_sensitivity,
#         'Best Specificity': best_specificity,
#         # 'Sensitivity': sensitivity,
#         # 'Specificity': specificity,
#         'Accuracy': accuracy,
#         'NPV': npv,
#         'PPV': ppv
#     }
#
# # Print the results
# for model_name, results in evaluation_results.items():
#     print(f"Results for {model_name}:")
#     print(f"AUC: {results['AUC']:.4f}")
#     print(f"Best Sensitivity: {results['Best Sensitivity']:.4f}")
#     print(f"Best Specificity: {results['Best Specificity']:.4f}")
#     # print(f"Best Threshold: {results['Best Threshold']:.4f}")
#     #
#     # print(f"Sensitivity: {results['Sensitivity']:.4f}")
#     # print(f"Specificity: {results['Specificity']:.4f}")
#     print(f"Accuracy: {results['Accuracy']:.4f}")
#     print(f"NPV: {results['NPV']:.4f}")
#     print(f"PPV: {results['PPV']:.4f}")
#     print()
# exit()


print('------------------------- AUC -------------------------')
# 计算AUC 95%CI
def bootstrap_auc(clf, X_train, Y_train, X_test, Y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], Y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))
#
print("train")
# x_train = X_train1.reset_index(drop=True).values
# y_train = y_train1.reset_index(drop=True).values
# x_test = X_train1.reset_index(drop=True).values
# y_test = y_train1.reset_index(drop=True).values
# AUC_CI1 = bootstrap_auc(clf=XGBClassifier(n_estimators=200, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
#
# x_train = X_train2.reset_index(drop=True).values
# y_train = y_train2.reset_index(drop=True).values
# x_test = X_train2.reset_index(drop=True).values
# y_test = y_train2.reset_index(drop=True).values
# AUC_CI2 = bootstrap_auc(clf=XGBClassifier(n_estimators=200, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

# x_train = X_train3.reset_index(drop=True).values
# y_train = y_train3.reset_index(drop=True).values
# x_test = X_train3.reset_index(drop=True).values
# y_test = y_train3.reset_index(drop=True).values
# AUC_CI3 = bootstrap_auc(clf=XGBClassifier(n_estimators=220, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
#
# x_train = X_train4.reset_index(drop=True).values
# y_train = y_train4.reset_index(drop=True).values
# x_test = X_train4.reset_index(drop=True).values
# y_test = y_train4.reset_index(drop=True).values
# AUC_CI4 = bootstrap_auc(clf=XGBClassifier(n_estimators=100, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

# x_train = X_train5.reset_index(drop=True).values
# y_train = y_train5.reset_index(drop=True).values
# x_test = X_train5.reset_index(drop=True).values
# y_test = y_train5.reset_index(drop=True).values
# AUC_CI5 = bootstrap_auc(clf=XGBClassifier(n_estimators=50, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

# x_train = X_train6.reset_index(drop=True).values
# y_train = y_train6.reset_index(drop=True).values
# x_test = X_train6.reset_index(drop=True).values
# y_test = y_train6.reset_index(drop=True).values
# AUC_CI6 = bootstrap_auc(clf=XGBClassifier(n_estimators=250, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

x_train = X_train7.reset_index(drop=True).values
y_train = y_train7.reset_index(drop=True).values
x_test = X_train7.reset_index(drop=True).values
y_test = y_train7.reset_index(drop=True).values
AUC_CI7 = bootstrap_auc(clf=XGBClassifier(n_estimators=1000, max_depth=8), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

#
# AUC_CI = bootstrap_auc(clf=LogisticRegression(C=200), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI2 = bootstrap_auc(clf=DecisionTreeClassifier(max_depth=15), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI1 = bootstrap_auc(clf=SVC(C=3.0,kernel='rbf', probability=True), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI3 = bootstrap_auc(clf=RandomForestClassifier(n_estimators=120, max_depth=3), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI4 = bootstrap_auc(clf=XGBClassifier(n_estimators=15, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI5 = bootstrap_auc(clf=KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

# AUC_CI1 = bootstrap_auc(clf=XGBClassifier(n_estimators=200, max_depth=4), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI2 = bootstrap_auc(clf=XGBClassifier(n_estimators=15, max_depth=4), X_train=x_train2, Y_train=y_train2, X_test=x_test2, Y_test=y_test2, nsamples=1000)
# AUC_CI3 = bootstrap_auc(clf=XGBClassifier(n_estimators=15, max_depth=4), X_train=x_train3, Y_train=y_train3, X_test=x_test3, Y_test=y_test3, nsamples=1000)
# AUC_CI4 = bootstrap_auc(clf=XGBClassifier(n_estimators=15, max_depth=4), X_train=x_train4, Y_train=y_train4, X_test=x_test4, Y_test=y_test4, nsamples=1000)
# AUC_CI5 = bootstrap_auc(clf=XGBClassifier(n_estimators=15, max_depth=4), X_train=x_train5, Y_train=y_train5, X_test=x_test5, Y_test=y_test5, nsamples=1000)
# AUC_CI6 = bootstrap_auc(clf=XGBClassifier(n_estimators=15, max_depth=4), X_train=x_train6, Y_train=y_train6, X_test=x_test6, Y_test=y_test6, nsamples=1000)
# AUC_CI7 = bootstrap_auc(clf=XGBClassifier(n_estimators=15, max_depth=4), X_train=x_train7, Y_train=y_train7, X_test=x_test7, Y_test=y_test7, nsamples=1000)
# print(AUC_CI1,
#       # AUC_CI2, AUC_CI3, AUC_CI4, AUC_CI5, AUC_CI6, AUC_CI7
#       )
# x_train = X_test1.reset_index(drop=True).values
# y_train = y_test1.reset_index(drop=True).values
# x_test = X_test1.reset_index(drop=True).values
# y_test = y_test1.reset_index(drop=True).values
# # AUC_CI1 = bootstrap_auc(clf=LogisticRegression(C=3.0), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # # AUC_CI1 = bootstrap_auc(clf=SVC(kernel='rbf', probability=True), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # # AUC_CI2 = bootstrap_auc(clf=DecisionTreeClassifier(max_depth=5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # # AUC_CI3 = bootstrap_auc(clf=RandomForestClassifier(n_estimators=120, max_depth=3), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # # AUC_CI4 = bootstrap_auc(clf=XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # # AUC_CI5 = bootstrap_auc(clf=KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI1 = bootstrap_auc(clf=XGBClassifier(n_estimators=120, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# # print("AUC_95%CI: ")
# # # print(AUC_CI)
# #
# # print(AUC_CI, AUC_CI1, AUC_CI2, AUC_CI3, AUC_CI4, AUC_CI5)
# print(AUC_CI1,
#       # AUC_CI2, AUC_CI3, AUC_CI4, AUC_CI5, AUC_CI6, AUC_CI7
#       )
# # C=C, gamma=gamma
print("test")
# x_train = X_test1.reset_index(drop=True).values
# y_train = y_test1.reset_index(drop=True).values
# x_test = X_test1.reset_index(drop=True).values
# y_test = y_test1.reset_index(drop=True).values
# AUC_CI11 = bootstrap_auc(clf=XGBClassifier(n_estimators=120, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

# x_train = X_test2.reset_index(drop=True).values
# y_train = y_test2.reset_index(drop=True).values
# x_test = X_test2.reset_index(drop=True).values
# y_test = y_test2.reset_index(drop=True).values
# AUC_CI12 = bootstrap_auc(clf=XGBClassifier(n_estimators=120, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

# x_train = X_test3.reset_index(drop=True).values
# y_train = y_test3.reset_index(drop=True).values
# x_test = X_test3.reset_index(drop=True).values
# y_test = y_test3.reset_index(drop=True).values
# AUC_CI13 = bootstrap_auc(clf=XGBClassifier(n_estimators=120, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
#
# x_train = X_test4.reset_index(drop=True).values
# y_train = y_test4.reset_index(drop=True).values
# x_test = X_test4.reset_index(drop=True).values
# y_test = y_test4.reset_index(drop=True).values
# AUC_CI14 = bootstrap_auc(clf=XGBClassifier(n_estimators=120, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
#
# x_train = X_test5.reset_index(drop=True).values
# y_train = y_test5.reset_index(drop=True).values
# x_test = X_test5.reset_index(drop=True).values
# y_test = y_test5.reset_index(drop=True).values
# AUC_CI15 = bootstrap_auc(clf=XGBClassifier(n_estimators=120, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

x_train = X_test6.reset_index(drop=True).values
y_train = y_test6.reset_index(drop=True).values
x_test = X_test6.reset_index(drop=True).values
y_test = y_test6.reset_index(drop=True).values
AUC_CI16 = bootstrap_auc(clf=XGBClassifier(n_estimators=1000, max_depth=8, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

x_train = X_test7.reset_index(drop=True).values
y_train = y_test7.reset_index(drop=True).values
x_test = X_test7.reset_index(drop=True).values
y_test = y_test7.reset_index(drop=True).values
AUC_CI17 = bootstrap_auc(clf=XGBClassifier(n_estimators=1000, max_depth=8, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
print(
    # AUC_CI1, AUC_CI2,
    # AUC_CI3, AUC_CI4,
    # AUC_CI5,
    # AUC_CI6,
    AUC_CI7,
    # AUC_CI11, AUC_CI12, AUC_CI13, AUC_CI14, AUC_CI15,
    AUC_CI16, AUC_CI17      )


