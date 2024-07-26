import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sksurv.metrics import concordance_index_censored
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
# 导入Excel表格数据

data = pd.read_excel('QST_inputdata0717-1.xlsx', sheet_name='Train')
# 提取特征和目标变量
X = data.drop(['SurvivalTime', 'Event'], axis=1)
y = data[['SurvivalTime', 'Event']]

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 将目标变量转换为适当的格式
y_surv = Surv.from_arrays(y['Event'].values, y['SurvivalTime'].values)



# 设置不同的正则化参数范围
alphas = np.logspace(-3, 1, num=50)

# # 初始化AUC列表
# auc_scores = []

# # 计算每个λ对应的AUC
# for alpha in alphas:
#     lasso = Lasso(alpha=alpha)
#     c_index = cross_val_score(lasso, X_scaled, y_surv, cv=5, scoring=concordance_index_censored)
#     auc_scores.append(np.mean(c_index))
#
# # 绘制横坐标为log(λ)，纵坐标为AUC的图
# plt.figure(figsize=(8, 6))
# plt.plot(np.log10(alphas), auc_scores)
# plt.xlabel('log(lambda)')
# plt.ylabel('AUC')
# plt.title('AUC vs. log(lambda)')
# plt.grid(True)
# plt.show()



# exit()
# 使用LassoCV选择特征
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_scaled, y['SurvivalTime'])
# 记录系数路径
alphas = lasso_cv.alphas_
coefs = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y['SurvivalTime'])
    coefs.append(lasso.coef_)

coefs = np.array(coefs)
# 绘制系数路径图
plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.plot(-np.log10(alphas), coefs[:, i], label=feature)
plt.xlabel('-log(alpha)')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficient Paths')
plt.legend()
plt.grid(True)
plt.show()

# 绘制最佳λ图
plt.figure(figsize=(8, 6))
plt.plot(-np.log10(lasso_cv.alphas_), lasso_cv.mse_path_.mean(axis=1))
plt.axvline(-np.log10(lasso_cv.alpha_), linestyle='--', color='r', label='Best alpha')
plt.xlabel('-log(alpha)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Mean Squared Error vs. -log(alpha)')
plt.legend()
plt.grid(True)
plt.show()
# exit()
# # 将完整的数据集传递给Surv.from_dataframe
# y_surv = Surv.from_dataframe(y, 'Event', 'SurvivalTime')
#
#
# # 设置不同的正则化参数范围
# alphas = np.logspace(-3, 1, num=50)

# # 初始化AUC列表
# auc_scores = []
#
# # 计算每个λ对应的AUC
# for alpha in alphas:
#     lasso = Lasso(alpha=alpha)
#     c_index = cross_val_score(lasso, X_scaled, y_surv, cv=5, scoring=concordance_index_censored)
#     auc_scores.append(np.mean(c_index))
#
# # 绘制横坐标为log(λ)，纵坐标为AUC的图
# plt.figure(figsize=(8, 6))
# plt.plot(np.log10(alphas), auc_scores)
# plt.xlabel('log(lambda)')
# plt.ylabel('AUC')
# plt.title('AUC vs. log(lambda)')
# plt.grid(True)
# plt.show()

# 选择非零系数的特征
selected_features = X.columns[(lasso_cv.coef_ != 0) & (abs(lasso_cv.coef_) > 60)]

# 打印选择的特征
print("Selected features:", selected_features)
# 提取选择特征和其系数
selected_features = X.columns[(lasso_cv.coef_ != 0)] #& (abs(lasso_cv.coef_) > 60)
feature_importance = lasso_cv.coef_[(lasso_cv.coef_ != 0) ] #& (abs(lasso_cv.coef_) > 60)

# 创建特征重要性表格
feature_table = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
feature_table = feature_table.sort_values(by='Importance', ascending=False)

# 打印特征重要性表格
print(feature_table)
# 保存特征重要性表格为Excel文件
# 导出特征重要性表格为Excel文件
feature_table.to_excel('QST-feature_importance0717-1.xlsx', sheet_name='Sheet2', index=False)
# 绘制特征重要性图
plt.barh(range(len(selected_features)), feature_importance)
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('QST_Feature Importance')
plt.show()

exit()



# 提取生存时间和事件状态信息
event = data['Event']
time = data['SurvivalTime']

# 将数据集转换为生存分析需要的格式
survival_data = Surv.from_arrays(event, time)

# 提取用于建模的特征列
X = data.drop(['SurvivalTime', 'Event'], axis=1)

# 创建Cox-LASSO模型并进行拟合
estimator = CoxnetSurvivalAnalysis(l1_ratio=1)
estimator.fit(X, survival_data)

# 获取模型的系数和特征名称
coef = estimator.coef_
feature_names = np.array(X.columns)

# 提取系数大于0的特征及其重要性
selected_features = feature_names[np.abs(coef) > 0]
selected_coef = coef[np.abs(coef) > 0]

# 创建特征重要性表格
feature_importance = pd.DataFrame({'Feature': selected_features, 'Coefficient': selected_coef})

# 保存特征重要性表格为Excel文件
feature_importance.to_excel('feature_importance.xlsx', index=False)

# 绘制特征重要性图
plt.barh(range(len(selected_features)), selected_coef)
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()