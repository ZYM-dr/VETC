import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.metrics import plot_roc_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
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
import joblib
# 结果保存目录
save_dir = r'MultiModels'
os.makedirs(save_dir, exist_ok=True)
# 读取数据
df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_CRIR')
# df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='Intest')
# df_train3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='Extest')

# 训练集
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
y_train1 = df_train1.iloc[:, -1]  # 训练集标签
# X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
# y_train2 = df_train2.iloc[:, -1]  # 训练集标签
# X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
# y_train3 = df_train3.iloc[:, -1]  # 训练集标签
# 测试集
# df_test = pd.read_excel(test_path)
# X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
# y_test1 = df_test1.iloc[:, -1]  # 测试集标签
# X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
# y_test2 = df_test2.iloc[:, -1]  # 测试集标签
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
    # "DT": DecisionTreeClassifier(max_depth=5),
#     "KNN": KNeighborsClassifier(n_neighbors=1, weights='uniform', metric_params=None, n_jobs=None),
    "LR": LogisticRegression(C=9),  # LR是图上的图例名称，可以修改
#     "RF": RandomForestClassifier(n_estimators=1, max_depth=1),
#     "SVM": LinearSVC(C=0.0001),
#     "XGBoost": XGBClassifier(n_estimators=1, max_depth=1, gamma=0.5),
#
}

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
plt.clf()  # 清空画板
for name,estimator in total_models1.items():
    estimator.fit(X_train1, y_train1)
    ax1 = plt.gca()
    metrics.RocCurveDisplay.from_estimator(estimator, X_train1, y_train1, name=name, ax=ax1)
plt.show()
evaluation_results = {}
# for model_name, model in total_models1.items():
for model_name, model in total_models1.items():
    # Ensure the number of features is consistent
    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError(f"Number of features in X_test ({x_train.shape[1]}) is different from X_train ({x_train.shape[1]}) for {model_name}")

    # Fit the model
    model.fit(x_train, y_train)
    joblib.dump(model, 'vetc_model.pkl')
    exit()
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
    df.to_excel('vetc_radiomics-prED1.xlsx')
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
