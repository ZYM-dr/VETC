import os
import pandas as pd
import sklearn
print(sklearn.__version__)
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import shap
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
df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='11-23-fit.best.lse')
df_train2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CLIF-SOFA')
df_train3 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CLIF-OF')
df_train4 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CLIF-C ACLF')
df_train5 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-MELD')
df_train6 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CPScore')
df_train7 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='T-CPclassification')
# df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='2Validation11-23-fit.best.lse')
# df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.min')
# 测试集路径
# test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
# df_test1 = pd.read_excel('D:\VETCNOM.xlsx', sheet_name='Sheet2')
# df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_test')

# 外部验证集路径
df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.lse')
df_test2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CLIF-SOFA')
df_test3 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CLIF-OF')
df_test4 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CLIF-C ACLF')
df_test5 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-MELD')
df_test6 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CPScore')
df_test7 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='V-CPclassification')
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
X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
y_train3 = df_train3.iloc[:, -1]  # 训练集标签\
X_train4 = df_train4.iloc[:, :-1]  # 训练集特征
y_train4 = df_train4.iloc[:, -1]  # 训练集标签
X_train5 = df_train5.iloc[:, :-1]  # 训练集特征
y_train5 = df_train5.iloc[:, -1]  # 训练集标签
X_train6 = df_train6.iloc[:, :-1]  # 训练集特征
y_train6 = df_train6.iloc[:, -1]  # 训练集标签
X_train7 = df_train7.iloc[:, :-1]  # 训练集特征
y_train7 = df_train7.iloc[:, -1]  # 训练集标签
# 测试集
# df_test = pd.read_excel(test_path)
X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
y_test1 = df_test1.iloc[:, -1]  # 测试集标签
X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
y_test2 = df_test2.iloc[:, -1]  # 测试集标签
X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
y_test3 = df_test3.iloc[:, -1]  # 测试集标签
X_test4 = df_test4.iloc[:, :-1]  # 测试集特征
y_test4 = df_test4.iloc[:, -1]  # 测试集标签
X_test5 = df_test5.iloc[:, :-1]  # 测试集特征
y_test5 = df_test5.iloc[:, -1]  # 测试集标签
X_test6= df_test6.iloc[:, :-1]  # 测试集特征
y_test6 = df_test6.iloc[:, -1]  # 测试集标签
X_test7 = df_test7.iloc[:, :-1]  # 测试集特征
y_test7 = df_test7.iloc[:, -1]  # 测试集标签


# 外部验证集
# df_test = pd.read_excel(test_path)
# X_extest1 = df_extest1.iloc[:, :-1]  # 测试集特征
# y_extest1 = df_extest1.iloc[:, -1]  # 测试集标签

# 机器学习建模
# 需要比较的模型，如果有已经训练好的参数，把参数加上。
print('-----train-------')
total_models1 = {
#     # "LR": LogisticRegression(C=20.0),  # LR是图上的图例名称，可以修改
#     # "DT": DecisionTreeClassifier(max_depth=5),
#     # "SVM": LinearSVC(C=3.0),
#     # "RF": RandomForestClassifier(n_estimators=120, max_depth=3),
      "CHAPT":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5),
#     "XGBoost": XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5),
#     # "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
}
total_models2 = {"CLIF-SOFA":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
total_models3 = {"CLIF-OF":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
total_models4 = {"CLIF-C ACLF":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
total_models5 = {"MELD":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
total_models6 = {"CPScore":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }
total_models7 = {"CPclassification":XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5), }

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
# total_models12 = {"CLIF-SOFA":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models13 = {"CLIF-OF":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models14 = {"CLIF-C ACLF":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models15 = {"MELD":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models16 = {"CPScore":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
# total_models17 = {"CPclassification":XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5), }
print("SEN-SPE-ACC-NPV-PPV ")
x_train = X_train1.reset_index(drop=True).values
y_train = y_train1.reset_index(drop=True).values
x_test = X_train1.reset_index(drop=True).values
y_test = y_train1.reset_index(drop=True).values
# x_train = X_test7.reset_index(drop=True).values
# y_train = y_test7.reset_index(drop=True).values
# x_test = X_test7.reset_index(drop=True).values
# y_test = y_test7.reset_index(drop=True).values
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

    # Calculate AUC
    print(y_prob)
    df = pd.DataFrame(y_prob)
    print(df)
    # data = y_prob.data(0)
    df.to_excel('CHAPT-T-prob.xlsx')
    exit()
    auc = roc_auc_score(y_test, y_prob)

    # Calculate predictions
    y_pred = model.predict(x_test)
    print(y_pred)
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

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
    #
    # print(f"Sensitivity: {results['Sensitivity']:.4f}")
    # print(f"Specificity: {results['Specificity']:.4f}")
    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"NPV: {results['NPV']:.4f}")
    print(f"PPV: {results['PPV']:.4f}")
    print()
exit()