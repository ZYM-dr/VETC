import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

from sksurv.datasets import load_whas500
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
from sklearn.neighbors import KNeighborsClassifier  # KNN
# from sksurv.neighbors import KNeighborsSurvivalAnalysis
# from lifelines import KNeighborsSurvivalAnalysis
from sklearn.linear_model import LogisticRegression
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.tree import DecisionTreeClassifier
# from lifelines.plotting import plot_decisions
from lifelines import KaplanMeierFitter
# from lifelines.plotting import plot_decision_function
import xgboost as xgb

from sksurv.linear_model import CoxnetSurvivalAnalysis # LogisticRegression'
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
# 导入训练集和测试集的Excel表格数据
train_data = pd.read_excel('QST_inputdata.xlsx', sheet_name='TRAIN-LASSO')
test_data = pd.read_excel('QST_inputdata.xlsx', sheet_name='TEST-LASSO')

# 提取训练集的生存数据
train_event = train_data['Event']
train_time = train_data['SurvivalTime']

# 提取测试集的生存数据
test_event = test_data['Event']
test_time = test_data['SurvivalTime']

# 提取训练集和测试集的特征
train_X = train_data.drop(['SurvivalTime', 'Event'], axis=1)
test_X = test_data.drop(['SurvivalTime', 'Event'], axis=1)
# 创建结构化数组
train_y = np.array(list(zip(train_event, train_time)), dtype=[('event', bool), ('time', float)])
# 随机森林模型
estimator_rf = RandomSurvivalForest(n_estimators=100, random_state=42)
estimator_rf.fit(train_X, train_y)
# estimator_rf.fit(train_X, train_time, train_event)

# 支持向量机模型
estimator_svm = FastKernelSurvivalSVM()
estimator_svm.fit(train_X, train_y)
# estimator_svm.fit(train_X, train_time, train_event)

# 决策树模型
estimator_dt = DecisionTreeClassifier()
estimator_dt.fit(train_X, train_time, train_event)

# XGBoost模型
dtrain = xgb.DMatrix(train_X, label=train_time)
params = {'objective': 'survival:cox', 'eval_metric': 'cox-nloglik'}
estimator_xgb = xgb.train(params, dtrain)

# 最近邻模型
estimator_knn = KNeighborsClassifier()
estimator_knn.fit(train_X, train_y)
# estimator_knn.fit(train_X, train_time, train_event)

# 逻辑回归模型
estimator_lr = CoxnetSurvivalAnalysis()
estimator_lr.fit(train_X, train_y)
# estimator_lr.fit(train_X, train_time, train_event)

# 在测试集上进行预测
survival_time_rf = estimator_rf.predict(test_X)
survival_time_svm = estimator_svm.predict(test_X)
survival_time_dt = estimator_dt.predict(test_X)
dtest = xgb.DMatrix(test_X, label=test_time)
survival_time_xgb = estimator_xgb.predict(dtest)
survival_time_knn = estimator_knn.predict(test_X)
survival_time_lr = estimator_lr.predict(test_X)

# 打印测试集上的生存时间预测结果
print("Random Survival Forest:")
print(survival_time_rf)
print("Support Vector Machine:")
print(survival_time_svm)
print("Decision Tree:")
print(survival_time_dt)
print("XGBoost:")
print(survival_time_xgb)
print("K-Nearest Neighbors:")
print(survival_time_knn)
print("Logistic Regression:")
print(survival_time_lr)
test_event = test_event.astype(bool)
test_time = test_time.astype(float)


# 评估模型性能（以C-Index为例）
cindex_rf = concordance_index_censored(test_event, test_time, survival_time_rf)
cindex_svm = concordance_index_censored(test_event, test_time, survival_time_svm)
cindex_dt = concordance_index_censored(test_event, test_time, survival_time_dt)
cindex_xgb = concordance_index_censored(test_event, test_time, survival_time_xgb)
event_knn = survival_time_knn['event'].astype(bool)
time_knn = survival_time_knn['time'].astype(float)
# cindex_knn = concordance_index_censored(test_event, test_time, (event_knn, time_knn))

cindex_lr = concordance_index_censored(test_event, test_time, survival_time_lr)

# 打印模型性能指标
print("Concordance Index (C-Index):")
print("Random Survival Forest:", cindex_rf[0])
print("Support Vector Machine:", cindex_svm[0])
print("Decision Tree:", cindex_dt[0])
print("XGBoost:", cindex_xgb[0])
# print("K-Nearest Neighbors:", cindex_knn[0])
print("Logistic Regression:", cindex_lr[0])
# 逻辑回归模型
estimator_lr = CoxnetSurvivalAnalysis(fit_baseline_model=True)
estimator_lr.fit(train_X, train_y)

# 在测试集上进行预测
risk_scores_lr = estimator_lr.predict(test_X)
# 计算每个模型的预测概率
pred_rf = estimator_rf.predict_survival_function(test_X)
pred_svm = estimator_svm.predict(test_X)
# pred_svm = estimator_svm.predict_survival_function(test_X)

pred_dt = estimator_dt.predict_proba(test_X)[:, 1]
pred_xgb = estimator_xgb.predict(dtest)
pred_knn = estimator_knn.predict_proba(test_X)[:, 1]
estimator_lr = CoxnetSurvivalAnalysis(fit_baseline_model=True)
estimator_lr.fit(train_X, train_y)
pred_lr = estimator_lr.predict_survival_function(test_X)

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# 计算决策曲线数据
def calculate_decision_curve(y_pred, y_event, threshold_range):
    decision_curve = []
    for threshold in threshold_range:
        tn = np.sum((y_pred <= threshold) & (~y_event))
        fp = np.sum((y_pred <= threshold) & y_event)
        tp = np.sum((y_pred > threshold) & y_event)
        fn = np.sum((y_pred > threshold) & (~y_event))

        net_benefit = (tp - fp) / (tp + fn)
        decision_curve.append(net_benefit)

    return decision_curve


# 计算校准曲线数据
def calculate_calibration_curve(y_pred, y_event, n_bins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_event, y_pred, n_bins=n_bins)
    brier_score = brier_score_loss(y_event, y_pred)

    return fraction_of_positives, mean_predicted_value, brier_score


# 定义阈值范围
threshold_range = np.arange(0.1, 1.1, 0.1)

# 计算每个模型的决策曲线数据和校准曲线数据
decision_curve_rf = calculate_decision_curve(survival_time_rf, test_event, threshold_range)
decision_curve_svm = calculate_decision_curve(survival_time_svm, test_event, threshold_range)
decision_curve_dt = calculate_decision_curve(survival_time_dt, test_event, threshold_range)
decision_curve_xgb = calculate_decision_curve(survival_time_xgb, test_event, threshold_range)
decision_curve_knn = calculate_decision_curve(time_knn, test_event, threshold_range)
decision_curve_lr = calculate_decision_curve(survival_time_lr, test_event, threshold_range)

# calibration_curve_rf = calculate_calibration_curve(survival_time_rf, test_event)
# calibration_curve_svm = calculate_calibration_curve(survival_time_svm, test_event)
# calibration_curve_dt = calculate_calibration_curve(survival_time_dt, test_event)
# calibration_curve_xgb = calculate_calibration_curve(survival_time_xgb, test_event)
calibration_curve_knn = calculate_calibration_curve(time_knn, test_event)
calibration_curve_lr = calculate_calibration_curve(survival_time_lr, test_event)

# 绘制决策曲线
plt.figure()
plt.plot(threshold_range, decision_curve_rf, label='Random Survival Forest')
plt.plot(threshold_range, decision_curve_svm, label='Support Vector Machine')
plt.plot(threshold_range, decision_curve_dt, label='Decision Tree')
plt.plot(threshold_range, decision_curve_xgb, label='XGBoost')
plt.plot(threshold_range, decision_curve_knn, label='K-Nearest Neighbors')
plt.plot(threshold_range, decision_curve_lr, label='Logistic Regression')
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('Decision Curve')
plt.legend()
plt.show()

# 绘制校准曲线
plt.figure()
plt.plot(calibration_curve_rf[1], calibration_curve_rf[0],
         label='Random Survival Forest (Brier Score: {:.3f})'.format(calibration_curve_rf[2]))
plt.plot(calibration_curve_svm[1], calibration_curve_svm[0],
         label='Support Vector Machine (Brier Score: {:.3f})'.format(calibration_curve_svm[2]))
plt.plot(calibration_curve_dt[1], calibration_curve_dt[0],
         label='Decision Tree (Brier Score: {:.3f})'.format(calibration_curve_dt[2]))
plt.plot(calibration_curve_xgb[1], calibration_curve_xgb[0],
         label='XGBoost (Brier Score: {:.3f})'.format(calibration_curve_xgb[2]))
plt.plot(calibration_curve_knn[1], calibration_curve_knn[0],
         label='K-Nearest Neighbors (Brier Score: {:.3f})'.format(calibration_curve_knn[2]))
plt.plot(calibration_curve_lr[1], calibration_curve_lr[0],
         label='Logistic Regression (Brier Score: {:.3f})'.format(calibration_curve_lr[2]))
plt.xlabel('Mean Predicted Value')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()









#
# # 计算决策曲线
# kmf = KaplanMeierFitter()
# kmf.fit(test_time, test_event)
# baseline_survival = kmf.survival_function_.values.flatten()
# plt.figure()
# # plt.plot(decision_thresholds_rf, decision_curve_rf, label='Random Survival Forest')
# # plt.plot(decision_thresholds_svm, decision_curve_svm, label='Support Vector Machine')
# # plt.plot(decision_thresholds_dt, decision_curve_dt, label='Decision Tree')
# # plt.plot(decision_thresholds_xgb, decision_curve_xgb, label='XGBoost')
# # plt.plot(decision_thresholds_knn, decision_curve_knn, label='K-Nearest Neighbors')
# # plt.plot(decision_thresholds_lr, decision_curve_lr, label='Logistic Regression')
# # plt.xlabel('Threshold')
# # plt.ylabel('Net Benefit')
# # plt.title('Decision Curve')
# # plt.legend()
# # plt.show()
# plt.plot(kmf.survival_function_.index, kmf.survival_function_.values, label='Baseline')
# plt.plot(pred_rf.index, pred_rf.values, label='Random Survival Forest')
# plt.plot(pred_svm.index, pred_svm.values, label='Support Vector Machine')
# plt.plot(pred_dt.index, pred_dt, label='Decision Tree')
# plt.plot(pred_xgb.index, pred_xgb, label='XGBoost')
# plt.plot(pred_knn.index, pred_knn, label='K-Nearest Neighbors')
# plt.plot(pred_lr.index, pred_lr.values, label='Logistic Regression')
# plt.xlabel('Time')
# plt.ylabel('Survival Probability')
# plt.title('Decision Curve')
# plt.legend(loc='lower left')
# plt.show()
#
# # 绘制校准曲线
# prob_true_rf, prob_pred_rf = calibration_curve(test_event, pred_rf, n_bins=10)
# prob_true_svm, prob_pred_svm = calibration_curve(test_event, pred_svm, n_bins=10)
# prob_true_dt, prob_pred_dt = calibration_curve(test_event, pred_dt, n_bins=10)
# prob_true_xgb, prob_pred_xgb = calibration_curve(test_event, pred_xgb, n_bins=10)
# prob_true_knn, prob_pred_knn = calibration_curve(test_event, pred_knn, n_bins=10)
# prob_true_lr, prob_pred_lr = calibration_curve(test_event, pred_lr, n_bins=10)
#
# plt.figure()
# plt.plot(prob_pred_rf, prob_true_rf, marker='o', label='Random Survival Forest')
# plt.plot(prob_pred_svm, prob_true_svm, marker='o', label='Support Vector Machine')
# plt.plot(prob_pred_dt, prob_true_dt, marker='o', label='Decision Tree')
# plt.plot(prob_pred_xgb, prob_true_xgb, marker='o', label='XGBoost')
# plt.plot(prob_pred_knn, prob_true_knn, marker='o', label='K-Nearest Neighbors')
# plt.plot(prob_pred_lr, prob_true_lr, marker='o', label='Logistic Regression')
# plt.xlabel('Predicted Probability')
# plt.ylabel('True Probability')
# plt.title('Calibration Curve')
# plt.legend(loc='lower right')
# plt.show()
