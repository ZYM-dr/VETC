import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import scikitplot as skp
from sklearn.calibration import calibration_curve

if __name__ == '__main__':
    """
    多个模型DCA曲线比较
    """

    # # 训练集路径
    df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='11-23-fit.best.lse')
    # df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.lse')
    # train_path = r'171testandtrain.xlsx'
    # df_train1 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='CR-train')
    # df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='11-23-fit.best.lse')
    # df_train2 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='CRR-train')
    # df_train3 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='Rtrain')

    # 测试集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
    # df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.lse')
    df_extest1 = pd.read_excel(
        'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
        sheet_name='CR-extest')
    # df_extest2 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='CRR-extest')
    # df_extest3 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='R-extest')
    # 测试集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
    # df_test2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='2Validation11-23-fit.best.lse')
    df_test1 = pd.read_excel(
        'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
        sheet_name='CR-test')
    # df_test2 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='CRR-test')
    # df_test3 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='Rtest')

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
    print("\n")
    # print('train datasets:\n', df_train1)
    # print('train datasets:\n', df_train2)
    # print('train datasets:\n', df_train3)
    # print('y_train1:', y_train1.values.tolist())
    # print('y_train2:', y_train2.values.tolist())
    # print('y_train3:', y_train3.values.tolist())

    # 测试集
    # df_test = pd.read_excel(test_path)
    X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
    y_test1 = df_test1.iloc[:, -1]  # 测试集标签
    X_extest1 = df_extest1.iloc[:, :-1]  # 测试集特征
    y_extest1 = df_extest1.iloc[:, -1]  # 测试集标签
    # X_extest3 = df_extest3.iloc[:, :-1]  # 测试集特征
    # y_extest3 = df_extest3.iloc[:, -1]  # 测试集标签d

    # 测试集
    # df_test = pd.read_excel(test_path)
    # X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
    # y_test2 = df_test2.iloc[:, -1]  # 测试集标签
    # X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
    # y_test2 = df_test2.iloc[:, -1]  # 测试集标签
    # X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
    # y_test3 = df_test3.iloc[:, -1]  # 测试集标签d

# print("---------ki67-train-----------")
#
# # clf1 = DecisionTreeClassifier(max_depth=5)
# clf1 = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
# clf2 = KNeighborsClassifier(n_neighbors=9, weights='uniform', metric_params=None, n_jobs=None)
# clf3 = LogisticRegression(random_state=42)
# clf4 = RandomForestClassifier(n_estimators=120, max_depth=5, criterion='gini', n_jobs=1)
# clf5 = SVC(probability=True, random_state=4)
# clf6 = XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5)
# print("---------ki67-text-----------")
# # clf1 = DecisionTreeClassifier(max_depth=5)
# clf1 = RandomForestClassifier(n_estimators=120, max_depth=5, criterion='gini', n_jobs=1)
# clf2 = KNeighborsClassifier(n_neighbors=9, weights='uniform', metric_params=None, n_jobs=None)
# clf3 = LogisticRegression(random_state=42)
# clf4 = RandomForestClassifier(n_estimators=1, max_depth=5, criterion='gini', n_jobs=1)
# clf5 = SVC(probability=True, random_state=4)
# clf6 = XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5)
print("---------ki67-extext-----------")
# # clf1 = DecisionTreeClassifier(max_depth=5)
clf1 = RandomForestClassifier(n_estimators=120, max_depth=4, criterion='gini', n_jobs=1)
clf2 = KNeighborsClassifier(n_neighbors=9, weights='uniform', metric_params=None, n_jobs=None)
clf3 = LogisticRegression(random_state=42)
clf4 = RandomForestClassifier(n_estimators=1, max_depth=5, criterion='gini', n_jobs=1)
clf5 = SVC(probability=True, random_state=4)
clf6 = XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5)

print("---------train-----------")
# clf1 = LogisticRegression(random_state=9)
# clf2 = DecisionTreeClassifier(min_samples_split=100, max_depth=10,criterion='gini')
# clf3 = SVC(probability=True, random_state=3)
# clf4 = RandomForestClassifier(n_estimators=120, max_depth=5, criterion='gini', n_jobs=1)
# clf5 = XGBClassifier(n_estimators=9, max_depth=4, gamma=0.5)
# clf6 = KNeighborsClassifier(n_neighbors=9, weights='uniform', metric_params=None, n_jobs=None)
# print("---------test-----------")
# clf1 = LogisticRegression(random_state=20)
# clf2 = DecisionTreeClassifier(max_depth=3)
# clf3 = SVC(probability=True, random_state=1)
# clf4 = RandomForestClassifier(random_state=12)
# clf5 = XGBClassifier(n_estimators=3, max_depth=4, gamma=0.5)
# clf6 = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)

clfs = [
    clf1,   clf2,clf3,    clf4, clf5, clf6
        ]
# clf_names = ['Logistic Regression', 'DecisionTree', 'SVM' ,'Random Forest','XGBoost', "KNN"]
clf_names = ['DT', "KNN",'LR', 'RF', 'SVM' ,'XGBoost']
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
for clf, name in zip(clfs, clf_names):
    clf.fit(X_extest1, y_extest1)
    prob_pos = clf.predict_proba(X_extest1)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_extest1, prob_pos, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=name)

ax.set_xlabel('Mean predicted value')
ax.set_ylabel('Fraction of positives')
ax.set_ylim([-0.05, 1.05])
ax.legend(loc='lower right')
# plt.title("Train")
# plt.savefig(os.path.join(save_dir, "ki67-CA-exTEST-231207.jpg"), dpi=300)  # 保存
plt.show()

print("---------PRI-CODE-VETC-----------")
# clf1 = XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5)
# clf2 = DecisionTreeClassifier(max_depth=5)
# clf3 = RandomForestClassifier(random_state=42)
# clf4 = LogisticRegression(random_state=42)
# clf5 = SVC(probability=True, random_state=42)
# clf6 = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)