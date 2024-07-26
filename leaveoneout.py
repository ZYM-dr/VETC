# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 04:42:21 2021

@author: tianyu
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn import svm

# ------ import feature set and labels ----------------------------------------
import pandas as pd

## ------ import feature set and labels--csv--using pandas ----------------------------------------
data = pd.read_excel('172.xlsx')
data = data.astype(float)
col = data.columns
data_all_pre = data[col[1:]]
all_data = np.array(data_all_pre)
labels_all_pre = data[col[0]]
labels = np.array(labels_all_pre)
labels[np.where(labels == 0)] = -1
# f = open('data1.dat', 'rb')
# dataset = pickle.load(f)
# all_data = dataset['all_data']
# labels = dataset['labels']

# ------ Nested Cross Validation ----------------------------------------------

p_labels = []
weights = []

decision_values = []
probas = []

loo = LeaveOneOut()
for train_idx, test_idx in loo.split(all_data):
    train_data = all_data[train_idx, :]
    train_label = labels[train_idx]
    test_data = all_data[test_idx, :]
    test_label = labels[test_idx]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # feature_selection
    train_data1 = train_data[np.squeeze(np.where(train_label == 1)), :]
    train_data2 = train_data[np.squeeze(np.where(train_label == -1)), :]

    t_stats = ttest_ind(train_data1, train_data2)
    t_values = t_stats[0]
    T_values_abs = np.abs(t_values)
    T_values_abs_sorted = np.flip(np.sort(T_values_abs))
    T_idx_sorted = np.flip(np.argsort(T_values_abs))

    # filterd train & test data
    total_feature_num = all_data.shape[1]
    filtered_feature_num = np.int_(total_feature_num * 0.1)
    train_data = train_data[:, T_idx_sorted[:filtered_feature_num]]
    test_data = test_data[:, T_idx_sorted[:filtered_feature_num]]

    # tunning hyper-parameters
    C = np.power(2.0, np.arange(-5, 6))
    params = {'kernel': ('linear',), 'C': C}
    svc = svm.SVC()
    clf = GridSearchCV(svc, params, cv=5)
    clf.fit(train_data, train_label)
    best_params = clf.best_params_

    svci = svm.SVC(**best_params, probability=True)  # probability output
    svci.fit(train_data, train_label)
    predicted_label = svci.predict(test_data)
    p_labels.append(predicted_label)

    predicted_proba = svci.predict_proba(test_data)  # 对测试集进行概率预测
    probas.append(predicted_proba)  # 汇总概率预测结果

    d_value = svci.decision_function(test_data)  # 获取决策值（样本到超平面的距离）
    decision_values.append(d_value[0])  # 汇总决策值

    w = svci.coef_
    wi = np.zeros((1, total_feature_num))
    wi[:, T_idx_sorted[:filtered_feature_num]] = w
    weights.append(wi)

    print('CV %d done...' % test_idx)

p_labels = np.squeeze(p_labels)
acc_final = np.mean(labels == p_labels)

# ---------- 绘制ROC曲线，计算曲线下面积AUC----------------------------------------
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay

y_score = np.array(decision_values)  # 决策值列表 -> numpy数组
# 输入：样本真实标签、决策值、正例标识
# 输出：假阳率（ROC曲线横轴）、真正率（ROC曲线纵轴）、一系列决策值的阈值
fpr, tpr, thds = roc_curve(y_true=labels, y_score=y_score, pos_label=1)

# 绘图
disp1 = RocCurveDisplay(fpr=fpr, tpr=tpr)
disp1.plot()
# 计算曲线下面积
auc = roc_auc_score(labels, y_score)

# ---------- Calibration Curve ------------------------------------------------
from sklearn.calibration import calibration_curve, CalibrationDisplay

probas_arr = np.squeeze(np.array(probas))  # 样本概率输出 -> numpy数组
y_prob = probas_arr[:, 1]
prob_true, prob_pred = calibration_curve(labels, y_prob, n_bins=10)

disp = CalibrationDisplay(prob_true, prob_pred, y_prob, estimator_name='SVC')
disp.plot()

# plt.plot(prob_pred, prob_true)
# plt.show()

# ---------- Decision Curve Analysis ------------------------------------------
from dca_utils import plot_decision_curves

probas_arr = np.squeeze(np.array(probas))
# p_series1, net_benifit1 = decision_curve_analysis(probas_arr, labels, 0, 1, 0.02)

# TP_all = np.sum(labels == 1)
# TN_all = np.sum(labels == -1)
# p_series2, net_benifit2 = calculate_net_benefit_all(TP_all, TN_all, 0, 1, 0.02)

prob_list = [probas_arr, ]
# 第一个参数：分类器的概率输出（不是分类器本身）
plot_decision_curves(prob_list, ['SVC'], labels, 0, 1, 0.02, -0.5, 0.6)