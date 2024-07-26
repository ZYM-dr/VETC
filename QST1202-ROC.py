import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
from sklearn.utils import resample
from scipy.stats import sem, t
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
# print(sklearn.__version__)
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
# from sklearn.metrics import plot_roc_curve
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
from scikitplot.metrics import plot_roc_curve
# from sklearn.metrics.RocCurveDisplay
warnings.filterwarnings("ignore")
np.random.seed(49)

from scipy.stats import sem, t

"""

多个模型ROC曲线比较

"""

# 训练集路径
# train_path = r'171testandtrain.xlsx'
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='11-23-fit.best.lse')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='trainday3')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='trainday7')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='trainday14')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='validationday3')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='validationday7')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='validationday14')yxh-lasso
df_train1 = pd.read_excel('yxh-lasso.xlsx')

# df_train1 = pd.read_excel('D:\VETCNOM.xlsx', sheet_name='Sheet1')
# df_train1 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CR-train')
# df_train2 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CRR-train')
# df_train3 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='Rtrain')
# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.lse')
# df_test1 = pd.read_excel('D:\VETCNOM.xlsx', sheet_name='Sheet2')
# df_test1 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CR-test')
# df_test2 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='CRR-test')
# df_test3 = pd.read_excel('0.84-0.70gpc3-20220816-mse-cv5-1se.xlsx', sheet_name='Rtest')
# # 结果保存目录
save_dir = r'QST'
os.makedirs(save_dir, exist_ok=True)

# 训练集

# df_train = pd.read_excel(train_path)
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
y_train1 = df_train1.iloc[:, -1]  # 训练集标签
# X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
# y_train1 = df_train1.iloc[:, -1]  # 训练集标签
# X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
# y_train2 = df_train2.iloc[:, -1]  # 训练集标签
# X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
# y_train3 = df_train3.iloc[:, -1]  # 训练集标签
# print('train datasets:\n', df_train1)
# print('train datasets:\n', df_train2)

# 测试集
# df_test = pd.read_excel(test_path)
# X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
# y_test1 = df_test1.iloc[:, -1]  # 测试集标签
# X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
# y_test1 = df_test1.iloc[:, -1]  # 测试集标签
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
    "LR": LogisticRegression(C=9),  # LR是图上的图例名称，可以修改
    "DT": DecisionTreeClassifier(max_depth=4),
    "SVM": LinearSVC(C=1.0),
    "RF": RandomForestClassifier(n_estimators=120, max_depth=3),
    # "XGBoost": XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5),  #训练集
    "XGBoost": XGBClassifier(n_estimators=1, max_depth=3, gamma=0.5),#3时间训练
    # "XGBoost": XGBClassifier(n_estimators=1, max_depth=6, gamma=0.8),#7时间训练
    # "XGBoost": XGBClassifier(n_estimators=2, max_depth=3, gamma=1),#14时间训练
    # "XGBoost": XGBClassifier(n_estimators=1, max_depth=3, gamma=0.5),#3时间验证
    # "XGBoost": XGBClassifier(n_estimators=1, max_depth=3, gamma=20),#14时间验证
    # "XGBoost": XGBClassifier(n_estimators=5, max_depth=110, gamma=3),  # 7时间验证
    # "KNN": KNeighborsClassifier(n_neighbors=9, weights='uniform', metric_params=None, n_jobs=None)
}
# total_models1 = {
#     "LR": LogisticRegression(C=9),  # LR是图上的图例名称，可以修改
#     "DT": DecisionTreeClassifier(max_depth=4),
#     "SVM": LinearSVC(C=1.0),
#     "RF": RandomForestClassifier(n_estimators=120, max_depth=3),
#     "XGBoost": XGBClassifier(n_estimators=8, max_depth=4, gamma=0.5),
#     "KNN": KNeighborsClassifier(n_neighbors=10, weights='uniform', metric_params=None, n_jobs=None)
# }
#

print('-----test-------')

# total_models2 = {
#     # "LR": LogisticRegression(C=20.0),  # LR是图上的图例名称，可以修改
#     # "DT": DecisionTreeClassifier(max_depth=3),
#     # "SVM": LinearSVC(C=3.0),
#     # "RF": RandomForestClassifier(n_estimators=12, max_depth=3),
#       "XGBoost": XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5),#test
#       "XGBoost": XGBClassifier(n_estimators=2, max_depth=4, gamma=0.5),#day14
#     # "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# }
# # total_models2 = {
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
# # 将多个模型在测试集上的ROC曲线绘制在一起
plt.clf()  # 清空画板
for name,estimator in total_models1.items():
    estimator.fit(X_train1, y_train1)
    ax1 = plt.gca()
    metrics.RocCurveDisplay.from_estimator(estimator, X_train1, y_train1, name=name, ax=ax1)
plt.show()
# plt.title("Trainday3")
# plt.savefig(os.path.join(save_dir, "ROC_trainday14-1.pdf"), dpi=300)  # 保存
# plt.savefig(os.path.join(save_dir, "ROC_validationday14-1.pdf"), dpi=300)  # 保存
# plt.clf()  # 清空画板
# for name, estimator in total_models.items():

# plt.title("Multi Models ROC Curve Comparison2")
# plt.savefig(os.path.join(save_dir, "ROC Curve Comparison_VETC_test.jpg"), dpi=300)  # 保存
# for name, estimator in total_models2.items():
#     estimator.fit(X_test1, y_test1)
#     ax2 = plt.gca()
#     metrics.RocCurveDisplay.from_estimator(estimator, X_test1, y_test1, name=name, ax=ax2)
# for name,estimator in total_models2.items():
# plt.title("Test")
# plt.savefig(os.path.join(save_dir, "ROC_test.pdf"), dpi=300)  # 保存
exit()
print("SEN-SPE-ACC-NPV-PPV ")
X_train = X_train1.reset_index(drop=True).values
y_train = y_train1.reset_index(drop=True).values
X_test = X_train1.reset_index(drop=True).values
y_test = y_train1.reset_index(drop=True).values
# X_train = X_test1.reset_index(drop=True).values
# y_train = y_test1.reset_index(drop=True).values
# X_test = X_test1.reset_index(drop=True).values
# y_test = y_test1.reset_index(drop=True).values

# Assuming you have your data in X and y
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store the evaluation results
evaluation_results = {}
# for model_name, model in total_models1.items():
for model_name, model in total_models1.items():
    # Ensure the number of features is consistent
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError(f"Number of features in X_test ({X_test.shape[1]}) is different from X_train ({X_train.shape[1]}) for {model_name}")

    # Fit the model
    model.fit(X_test, y_test)

    # Special handling for models without predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        raise AttributeError(f"{model_name} does not have a predict_proba or decision_function method.")
    print(y_prob)
    df = pd.DataFrame(y_prob)
    print(df)
    # data = y_prob.data(0)
    # df.to_excel('trainday3-1-prob.xlsx')
    # df.to_excel('trainday7-1-prob.xlsx')
    # df.to_excel('trainday14-1-prob.xlsx')
    # df.to_excel('validationday3-1-prob.xlsx')
    # df.to_excel('validationday7-1-prob.xlsx')
    # df.to_excel('validationday14-1-prob.xlsx')
    # exit()
    # Calculate AUC
    auc = roc_auc_score(y_test, y_prob)
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
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate sensitivity, specificity, accuracy, NPV, and PPV
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # npv = tn / (tn + fn)
    # ppv = tp / (tp + fp)
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Avoid division by zero
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0  # Avoid division by zero
    # Store results in the dictionary
    evaluation_results[model_name] = {
        'AUC': auc,
        'Best Sensitivity': best_sensitivity,
        'Best Specificity': best_specificity,
        'Best Threshold': best_threshold,
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
    print(f"Best Threshold: {results['Best Threshold']:.4f}")
    # print(f"Sensitivity: {results['Sensitivity']:.4f}")
    # print(f"Specificity: {results['Specificity']:.4f}")
    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"NPV: {results['NPV']:.4f}")
    print(f"PPV: {results['PPV']:.4f}")
    print()
exit()
# 计算AUC 95%CI
print("AUC_95%CI: ")
def bootstrap_auc(clf, X_train, Y_train, X_test, Y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], Y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 100.00))

# x_train = X_train.reset_index(drop=True).values
# y_train = y_train.reset_index(drop=True).values
# x_test = X_train.reset_index(drop=True).values
# y_test = y_train.reset_index(drop=True).values
# #
# AUC_CI = bootstrap_auc(clf=LogisticRegression(C=1500), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI2 = bootstrap_auc(clf=DecisionTreeClassifier(max_depth=1), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI1 = bootstrap_auc(clf=SVC(C=100,probability=True), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI3 = bootstrap_auc(clf=RandomForestClassifier(n_estimators=500, max_depth=3), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI4 = bootstrap_auc(clf=XGBClassifier(n_estimators=200, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI5 = bootstrap_auc(clf=KNeighborsClassifier(n_neighbors=2, weights='uniform', metric_params=None, n_jobs=None), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

x_train = X_test1.reset_index(drop=True).values
y_train = y_test1.reset_index(drop=True).values
x_test = X_test1.reset_index(drop=True).values
y_test = y_test1.reset_index(drop=True).values
# AUC_CI6 = bootstrap_auc(clf=LogisticRegression(C=200.0), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI7 = bootstrap_auc(clf=DecisionTreeClassifier(max_depth=400), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI8 = bootstrap_auc(clf=SVC(C=5,probability=True), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI9 = bootstrap_auc(clf=RandomForestClassifier(n_estimators=200, max_depth=3), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
# AUC_CI10 = bootstrap_auc(clf=XGBClassifier(n_estimators=120, max_depth=4, gamma=0.5), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)
AUC_CI11 = bootstrap_auc(clf=KNeighborsClassifier(n_neighbors=3, weights='uniform', metric_params=None, n_jobs=None), X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, nsamples=1000)

print(
    # AUC_CI6,
    # AUC_CI7, AUC_CI8,
    # AUC_CI9,
    # AUC_CI10,
    AUC_CI11)

print("AUC_95%CI: ")
# print(AUC_CI)

# print(AUC_CI, AUC_CI1, AUC_CI2,
      # AUC_CI3,
      # AUC_CI4,
      # AUC_CI5
      # )


