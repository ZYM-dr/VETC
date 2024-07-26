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
    # train_path = r'171testandtrain.xlsx'
    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_train')
    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR_train')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIPR_train')
    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR-validaton1')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR-validaton1')
    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_CR')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_IR')
    # df_test3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_PR')
    # df_test4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_IPR')
    # df_test5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_CRIR')
    # df_test6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_CRIPR')

    df_train1 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Train')
    df_train2 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Intest')
    df_train3 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Extest')

    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_CR')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_IR')
    # df_test3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_PR1')
    # df_test4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_IPR1')
    # df_test5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_CRIR')
    # df_test6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_CRIPR')
    #
    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_CR')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_IR')
    # df_test3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_PR')
    # df_test4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_IPR')
    # df_test5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_CRIR')
    # df_test6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_CRIPR')
    #
    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_CR')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_IR')
    # df_test3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_PR')
    # df_test4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_IPR')
    # df_test5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_CRIR')
    # df_test6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_CRIPR')
    # 测试集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_train')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR_train')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIPR_train')
    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR-validaton2')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR-validaton2')
    # 测试集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
    # df_test1 = pd.read_excel(
    #     )
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
    X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
    y_train2 = df_train2.iloc[:, -1]  # 训练集标签
    X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
    y_train3 = df_train3.iloc[:, -1]  # 训练集标签


    # 测试集
    # df_test = pd.read_excel(test_path)
    # X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
    # y_test1 = df_test1.iloc[:, -1]  # 测试集标签
    # X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
    # y_test2 = df_test2.iloc[:, -1]  # 测试集标签
    # X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
    # y_test3 = df_test3.iloc[:, -1]  # 测试集标签d
    # X_test4 = df_test4.iloc[:, :-1]  # 测试集特征
    # y_test4 = df_test4.iloc[:, -1]  # 测试集标签
    # X_test5 = df_test5.iloc[:, :-1]  # 测试集特征
    # y_test5 = df_test5.iloc[:, -1]  # 测试集标签d
    # X_test6 = df_test6.iloc[:, :-1]  # 测试集特征
    # y_test6 = df_test6.iloc[:, -1]  # 测试集标签d




clf = LogisticRegression(random_state=40)

# 计算三个训练集的校准曲线数据

# 训练集
prob_pos1 = clf.fit(X_train1, y_train1).predict_proba(X_train1)[:, 1]
fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_train1, prob_pos1, n_bins=20)

prob_pos2 = clf.fit(X_train2, y_train2).predict_proba(X_train2)[:, 1]
fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_train2, prob_pos2, n_bins=10)

prob_pos3 = clf.fit(X_train3, y_train3).predict_proba(X_train3)[:, 1]
fraction_of_positives3, mean_predicted_value3 = calibration_curve(y_train3, prob_pos3, n_bins=10)
# 测试集INTEST
# prob_pos1 = clf.fit(X_test1, y_test1).predict_proba(X_test1)[:, 1]
# fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_test1, prob_pos1, n_bins=10)
#
# prob_pos2 = clf.fit(X_test2, y_test2).predict_proba(X_test2)[:, 1]
# fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_test2, prob_pos2, n_bins=10)
#
# prob_pos3 = clf.fit(X_test3, y_test3).predict_proba(X_test3)[:, 1]
# fraction_of_positives3, mean_predicted_value3 = calibration_curve(y_test3, prob_pos3, n_bins=10)
#
# prob_pos4 = clf.fit(X_test4, y_test4).predict_proba(X_test4)[:, 1]
# fraction_of_positives4, mean_predicted_value4 = calibration_curve(y_test4, prob_pos4, n_bins=10)
#
# prob_pos5 = clf.fit(X_test5, y_test5).predict_proba(X_test5)[:, 1]
# fraction_of_positives5, mean_predicted_value5 = calibration_curve(y_test5, prob_pos5, n_bins=10)
#
# prob_pos6 = clf.fit(X_test6, y_test6).predict_proba(X_test6)[:, 1]
# fraction_of_positives6, mean_predicted_value6 = calibration_curve(y_test6, prob_pos6, n_bins=10)

# V1
# prob_pos1 = clf.fit(X_test1, y_test1).predict_proba(X_test1)[:, 1]
# fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_test1, prob_pos1, n_bins=10)
#
# prob_pos2 = clf.fit(X_test2, y_test2).predict_proba(X_test2)[:, 1]
# fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_test2, prob_pos2, n_bins=4)
#
# prob_pos3 = clf.fit(X_test3, y_test3).predict_proba(X_test3)[:, 1]
# fraction_of_positives3, mean_predicted_value3 = calibration_curve(y_test3, prob_pos3, n_bins=10)
#
# prob_pos4 = clf.fit(X_test4, y_test4).predict_proba(X_test4)[:, 1]
# fraction_of_positives4, mean_predicted_value4 = calibration_curve(y_test4, prob_pos4, n_bins=9)

# prob_pos5 = clf.fit(X_test5, y_test5).predict_proba(X_test5)[:, 1]
# fraction_of_positives5, mean_predicted_value5 = calibration_curve(y_test5, prob_pos5, n_bins=10)

# prob_pos6 = clf.fit(X_test6, y_test6).predict_proba(X_test6)[:, 1]
# fraction_of_positives6, mean_predicted_value6 = calibration_curve(y_test6, prob_pos6, n_bins=10)
#V2
# prob_pos1 = clf.fit(X_test1, y_test1).predict_proba(X_test1)[:, 1]
# fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_test1, prob_pos1, n_bins=10)
#
# prob_pos2 = clf.fit(X_test2, y_test2).predict_proba(X_test2)[:, 1]
# fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_test2, prob_pos2, n_bins=10)
#
# prob_pos3 = clf.fit(X_test3, y_test3).predict_proba(X_test3)[:, 1]
# fraction_of_positives3, mean_predicted_value3 = calibration_curve(y_test3, prob_pos3, n_bins=10)
#
# prob_pos4 = clf.fit(X_test4, y_test4).predict_proba(X_test4)[:, 1]
# fraction_of_positives4, mean_predicted_value4 = calibration_curve(y_test4, prob_pos4, n_bins=10)
#
# prob_pos5 = clf.fit(X_test5, y_test5).predict_proba(X_test5)[:, 1]
# fraction_of_positives5, mean_predicted_value5 = calibration_curve(y_test5, prob_pos5, n_bins=10)
#
# prob_pos6 = clf.fit(X_test6, y_test6).predict_proba(X_test6)[:, 1]
# fraction_of_positives6, mean_predicted_value6 = calibration_curve(y_test6, prob_pos6, n_bins=10)
#train
# prob_pos1 = clf.fit(X_test1, y_test1).predict_proba(X_test1)[:, 1]
# fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_test1, prob_pos1, n_bins=10)
#
# prob_pos2 = clf.fit(X_test2, y_test2).predict_proba(X_test2)[:, 1]
# fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_test2, prob_pos2, n_bins=10)
#
# prob_pos3 = clf.fit(X_test3, y_test3).predict_proba(X_test3)[:, 1]
# fraction_of_positives3, mean_predicted_value3 = calibration_curve(y_test3, prob_pos3, n_bins=10)
#
# prob_pos4 = clf.fit(X_test4, y_test4).predict_proba(X_test4)[:, 1]
# fraction_of_positives4, mean_predicted_value4 = calibration_curve(y_test4, prob_pos4, n_bins=10)
#
# prob_pos5 = clf.fit(X_test5, y_test5).predict_proba(X_test5)[:, 1]
# fraction_of_positives5, mean_predicted_value5 = calibration_curve(y_test5, prob_pos5, n_bins=9)
#
# prob_pos6 = clf.fit(X_test6, y_test6).predict_proba(X_test6)[:, 1]
# fraction_of_positives6, mean_predicted_value6 = calibration_curve(y_test6, prob_pos6, n_bins=10)
# 创建图形和坐标轴
fig, ax = plt.subplots()

# 添加完美校准曲线
ax.plot([0.0, 0.6], [0.0, 0.6], 'k:', label='Perfectly calibrated')

# 绘制三个训练集的校准曲线
# ax.plot(mean_predicted_value3, fraction_of_positives3, '^-', label='Train')
ax.plot(mean_predicted_value2, fraction_of_positives2, 'o-', label='Internal test')
# ax.plot(mean_predicted_value1, fraction_of_positives1, 's-', label='External test')


# ax.plot(mean_predicted_value4, fraction_of_positives4, '^-', label='IPR')
# ax.plot(mean_predicted_value5, fraction_of_positives5, '^-', label='CRIR')
# ax.plot(mean_predicted_value6, fraction_of_positives6, '^-', label='CRIPR')

# 设置图形属性
ax.set_xlabel('Mean predicted value',fontdict={'family': 'Times New Roman', 'weight':'bold','fontsize': 15})
ax.set_ylabel('Fraction of positives',fontdict={'family': 'Times New Roman','weight':'bold', 'fontsize': 15})
font1 = {'family':'Times New Roman','weight':'bold','size':10,}
plt.tick_params(labelsize=10)#坐标轴刻度数字大小
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]

ax.set_ylim([-0.05, 0.6])
# plt.xlabel('X', prop=font1)
# plt.ylabel('Y',prop=font1)
plt.legend(prop=font1,loc='lower right')
# ax.legend(loc='lower right')

# 显示图形
plt.show()
exit()
# 计算三个训练集的校准曲线数据
# prob_pos1 = clf.fit(X_train1, y_train1).predict_proba(X_train1)[:, 1]
# fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_train1, prob_pos1, n_bins=10)
#
# prob_pos2 = clf.fit(X_train2, y_train2).predict_proba(X_train2)[:, 1]
# fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_train2, prob_pos2, n_bins=10)

# prob_pos3 = clf.fit(X_test3, y_test3).predict_proba(X_test3)[:, 1]
# fraction_of_positives3, mean_predicted_value3 = calibration_curve(y_test3, prob_pos3, n_bins=10)
# 测试集
# prob_pos1 = clf.fit(X_test1, y_test1).predict_proba(X_test1)[:, 1]
# fraction_of_positives1, mean_predicted_value1 = calibration_curve(y_test1, prob_pos1, n_bins=10)
#
# prob_pos2 = clf.fit(X_test2, y_test2).predict_proba(X_test2)[:, 1]
# fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_test2, prob_pos2, n_bins=10)

# prob_pos3 = clf.fit(X_test3, y_test3).predict_proba(X_test3)[:, 1]
# fraction_of_positives3, mean_predicted_value3 = calibration_curve(y_test3, prob_pos3, n_bins=10)

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 添加完美校准曲线
ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

# 绘制三个训练集的校准曲线
ax.plot(mean_predicted_value1, fraction_of_positives1, 's-', label='CR')
ax.plot(mean_predicted_value2, fraction_of_positives2, 'o-', label='CRIR')
ax.plot(mean_predicted_value3, fraction_of_positives3, '^-', label='CRIPR')

# 设置图形属性
ax.set_xlabel('Mean predicted value')
ax.set_ylabel('Fraction of positives')
ax.set_ylim([-0.05, 1.05])
ax.legend(loc='lower right')




exit()



clfs = [clf1, clf2, ]
clf_names = ['CR', 'CRIR', ]



fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
for clf, name in zip(clfs, clf_names):
    clf.fit(X_train1, y_train1)
    prob_pos = clf.predict_proba(X_train1)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_train1, prob_pos, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=name)

ax.set_xlabel('Mean predicted value',fontdict={'family': 'Times New Roman', 'fontsize': 15})
ax.set_ylabel('Fraction of positives',fontdict={'family': 'Times New Roman', 'fontsize': 15})
ax.set_ylim([-0.05, 1.05])
ax.legend(loc='lower right')
plt.show()
