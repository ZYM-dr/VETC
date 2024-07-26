import os
import pandas as pd
import sklearn
print(sklearn.__version__)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
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
# df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_train')
# df_train3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIOR_train')
# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_test1 = pd.read_excel('D:\VETCNOM.xlsx', sheet_name='Sheet2')
# df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_test')
# df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR_test')
# df_test3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIOR_test')
# 外部验证集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_train1 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-train')
df_train1 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-TRAIN+INTEST')
df_test1 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-test')
# df_test3 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='Rtest')
df_test2 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-extest')
# df_test2 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CRR-test')
# df_test3 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='Rtest')
# df_extest1 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CR-extest')
# df_test2 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='CRR-test')
# df_test3 = pd.read_excel('ki670819-R-mse.xlsx', sheet_name='Rtest')
# 结果保存目录
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
X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
y_test2 = df_test2.iloc[:, -1]  # 测试集标签
# X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
# y_test3 = df_test3.iloc[:, -1]  # 测试集标签
# print('test datasets:\n', df_test1)
# print('test datasets:\n', df_test2)
# 外部验证集
# df_test = pd.read_excel(test_path)
# X_extest1 = df_extest1.iloc[:, :-1]  # 测试集特征
# y_extest1 = df_extest1.iloc[:, -1]  # 测试集标签
# X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
# y_test2 = df_test2.iloc[:, -1]  # 测试集标签
# X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
# y_test3 = df_test3.iloc[:, -1]  # 测试集标签
# print('test datasets:\n', df_test1)
# print('test datasets:\n', df_test2)
# 机器学习建模
# 需要比较的模型，如果有已经训练好的参数，把参数加上。
print('-----train-------')
total_models1 = {
#
    "DT": DecisionTreeClassifier(max_depth=5),
#     "KNN": KNeighborsClassifier(n_neighbors=1, weights='uniform', metric_params=None, n_jobs=None),
#     "LR": LogisticRegression(C=0.001),  # LR是图上的图例名称，可以修改
#     "RF": RandomForestClassifier(n_estimators=1, max_depth=1),
#     "SVM": LinearSVC(C=0.0001),
#     "XGBoost": XGBClassifier(n_estimators=1, max_depth=1, gamma=0.5),
#
}
# total_models2 = {
#     "LR2": LogisticRegression(C=2.0),  # LR是图上的图例名称，可以修改
#     # "mT1": DecisionTreeClassifier(max_depth=5),
#     # "SVM2": LinearSVC(C=3.0),
#     # "RF2": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost2": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     # "KNN2": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }
# total_models3 = {
#     "LR3": LogisticRegression(C=2.0),  # LR是图上的图例名称，可以修改
#     # "DT3": DecisionTreeClassifier(max_depth=5),
#     # "SVM3": LinearSVC(C=3.0),
#     # "RF3": RandomForestClassifier(n_estimators=120, max_depth=3),
#     # "XGBoost3": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
#     # "KNN3": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }


print('-----test-------')
# total_models2 = {
#       "DT": DecisionTreeClassifier(max_depth=500),
#       "KNN1": KNeighborsClassifier(n_neighbors=1, weights='uniform', metric_params=None, n_jobs=None),
#       "LR": LogisticRegression(C=0.00000000000000001),  # LR是图上的图例名称，可以修改
#       "RF": RandomForestClassifier(n_estimators=1, max_depth=3),
#       "SVM": LinearSVC(C=0.0000000000000000000000000000000000000000000000001),
#       "XGBoost": XGBClassifier(n_estimators=1, max_depth=1, gamma=0.1),
#
# }
# total_models3 = {   #test2
#     "DT": DecisionTreeClassifier(max_depth=5),
#     "KNN": KNeighborsClassifier(n_neighbors=1, weights='uniform', metric_params=None, n_jobs=None),
#     "LR": LogisticRegression(C=0.00000000000000001),  # LR是图上的图例名称，可以修改
#     "RF": RandomForestClassifier(n_estimators=1, max_depth=1),
#     "SVM": LinearSVC(C=0.00000000000000000000000000000000000000000000000001),
#     "XGBoost": XGBClassifier(n_estimators=1, max_depth=8, gamma=0.0),
# }
# total_models3 = {
#     "LR3": LogisticRegression(C=3.0),  # LR是图上的图例名称，可以修改
# #     # "DT3": DecisionTreeClassifier(max_depth=5),
# #     # "SVM3": LinearSVC(C=3.0),
# #     # "RF3": RandomForestClassifier(n_estimators=120, max_depth=3),
# #     # "XGBoost3": XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5),
# #     "KNN3": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# #
# }
print("SEN-SPE-ACC-NPV-PPV ")
x_train = X_train1.reset_index(drop=True).values
y_train = y_train1.reset_index(drop=True).values
x_test = X_train1.reset_index(drop=True).values
y_test = y_train1.reset_index(drop=True).values
# x_train = X_test1.reset_index(drop=True).values
# y_train = y_test1.reset_index(drop=True).values
# x_test = X_test1.reset_index(drop=True).values
# y_test = y_test1.reset_index(drop=True).values
# x_train = X_test2.reset_index(drop=True).values  #model3
# y_train = y_test2.reset_index(drop=True).values
# x_test = X_test2.reset_index(drop=True).values
# y_test = y_test2.reset_index(drop=True).values
# Dictionary to store the evaluation results
evaluation_results = {}
# for model_name, model in total_models1.items():
for model_name, model in total_models1.items():
    # Ensure the number of features is consistent
    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError(f"Number of features in X_test ({x_train.shape[1]}) is different from X_train ({x_train.shape[1]}) for {model_name}")

    # Fit the model
    model.fit(x_train, y_train)

    # Special handling for models without predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(x_test)
    else:
        raise AttributeError(f"{model_name} does not have a predict_proba or decision_function method.")
    print(y_prob)
    df = pd.DataFrame(y_prob)
    print(df)
    # Calculate AUC
    # print(y_prob)
    # df = pd.DataFrame(y_prob)
    # print(df)
    # data = y_prob.data(0)
    # "DT":     "KNN1":     "LR":     "RF":    "SVM":  "XGBoost"
    # df.to_excel('KI67-DT-TE-prob1.xlsx')
    # exit()
    auc = roc_auc_score(y_test, y_prob)

    # Calculate predictions
    y_pred = model.predict(x_test)
    # print(y_pred)
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(y_pred)
    df = pd.DataFrame(y_pred)
    print(df)
    df.to_excel('KI67-DT-TR+INTE-prED1.xlsx')
    # Calculate sensitivity, specificity, accuracy, NPV, and PPV
    # Find the threshold that maximizes sensitivity + specificity
    thresholds = np.linspace(0, 1, 100)
    best_threshold = None
    best_sensitivity = 0
    best_specificity = 0

    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        if sensitivity + specificity > best_sensitivity + best_specificity:
            best_sensitivity = sensitivity
            best_specificity = specificity
            best_threshold = threshold
            # Store results in the dictionary


    # Calculate accuracy, NPV, and PPV at the best threshold
    y_pred_best_threshold = (y_prob > best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best_threshold).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)

    # Store results in the dictionary
    evaluation_results[model_name] = {
        'AUC': auc,
        'Best Sensitivity': best_sensitivity,
        'Best Specificity': best_specificity,
        # 'Sensitivity': sensitivity,
        # 'Specificity': specificity,
        'Accuracy': accuracy,
        'NPV': npv,
        'PPV': ppv
    }

# Print the results
for model_name, results in evaluation_results.items():
    print(f"Results for {model_name}:")
    print(f"AUC: {results['AUC']:.4f}")
    print(f"Best Sensitivity: {results['Best Sensitivity']:.4f}")
    print(f"Best Specificity: {results['Best Specificity']:.4f}")
    # print(f"Best Threshold: {results['Best Threshold']:.4f}")
    # print(f"Sensitivity: {results['Sensitivity']:.4f}")
    # print(f"Specificity: {results['Specificity']:.4f}")
    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"NPV: {results['NPV']:.4f}")
    print(f"PPV: {results['PPV']:.4f}")
    print()
exit()
# estimator = LogisticRegression(C=3.0)
# estimator1 = DecisionTreeClassifier(max_depth=5)
# estimator2 = LinearSVC(C=3.0)
# estimator3 = RandomForestClassifier(n_estimators=120, max_depth=3)
# estimator4 = XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5)
# estimator5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)

# for name,estimator in total_models1.items():
#     def y_score_test(estimator):
#         # 机器学习建模
#         estimator.fit(X_train1, y_train1)
#         try:
#             score = estimator.predict_proba(X_train1)[:, 1].tolist()  # 预测概率
#         except:
#             score = estimator._predict_proba_lr(X_train1)[:, 1].tolist()
#         return score
#     print(y_score_test(estimator))
#     data = pd.DataFrame(y_score_test(estimator))
#     writer = pd.ExcelWriter('scoretrain.xlsx')		# 写入Excel文件
#     data.to_excel(writer, 'KNN', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
#     writer.save()
    # lr_score=y_score_test(estimator)
    # DT_score=y_score_test(estimator)
    # SVM_score=y_score_test(estimator)
    # RF_score=y_score_test(estimator)
    # XGBoost_score=y_score_test(estimator)
    # KNN_score=y_score_test(estimator)
# ====================== 模型评估 ======================
# 将多个模型在测试集上的ROC曲线绘制在一起
plt.clf()  # 清空画板
for name,estimator in total_models1.items():
    estimator.fit(X_train1, y_train1)
    ax1 = plt.gca()
    metrics.RocCurveDisplay.from_estimator(estimator, X_train1, y_train1, name=name, ax=ax1)
plt.show()
# plt.clf()  # 清空画板
# for name,estimator in total_models2.items():
#     estimator.fit(X_test1, y_test1)
#     ax2 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test1, y_test1, name=name, ax=ax2)
# plt.show()
# plt.clf()  # 清空画板
# for name, estimator in total_models3.items():
#     estimator.fit(X_test2, y_test2)
#     ax3 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test2, y_test2, name=name, ax=ax3)
# plt.show()
# plt.title("VETC_RAD_TRAIN")
# plt.savefig(os.path.join(save_dir, "VETC_RAD_TRAIN1.jpg"), dpi=300)  # 保存

# plt.clf()  # 清空画板
# for name, estimator in total_models2.items():
#     estimator.fit(X_test1, y_test1)
#     ax = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_train1, y_train1, name=name, ax=ax)

# plt.title("Multi Models ROC Curve Comparison2")
# plt.savefig(os.path.join(save_dir, "roc_multi_models2.jpg"), dpi=300)  # 保存
# for name,estimator in total_models2.items():
#     estimator.fit(X_test2, y_test2)
#     ax2 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_train2, y_train2, name=name, ax=ax2)
# for name,estimator in total_models1.items():
#     estimator.fit(X_test1, y_test1)
#     ax2 = plt.gca()
#     metrics.plot_roc_curve(estimator, X_test1, y_test1, name=name, ax=ax2)

# # for name,estimator in total_models3.items():
# #     estimator.fit(X_test3, y_test3)
# #     ax2 = plt.gca()
# #     metrics.plot_roc_curve(estimator, X_train3, y_train3, name=name, ax=ax2)
# plt.title("Multi Models ROC Curve Comparison-testlr")
# plt.show()
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

