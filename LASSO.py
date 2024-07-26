import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

# MSE
# MAE

# R2
# 可解释方差


"""
特征筛选

确保被筛选数据中没有空值

"""


def LassoScreening(X, y):
    """
    return: 特征重要性
    """
    X_std = preprocessing.StandardScaler().fit_transform(X)
    print(X_std.shape)
    clf = LassoCV(max_iter=100000, alphas=[0])
    clf.fit(X_std, y.values.reshape(-1))
    return np.abs(clf.coef_)
    # return clf.coef_

#
# def BorutaScreeningClassifier(X, y):
#     X_std = preprocessing.StandardScaler().fit_transform(X)
#     estimator = RandomForestClassifier(class_weight='balanced')
#     feat_selector = BorutaPy(estimator, n_estimators='auto')
#     feat_selector.fit(X_std, y)
#     ranking = feat_selector.ranking_  # 重要性等级, 数字表示, 1最重要, 2一般, 3及以后均不重要
#     return ranking

#
# def BorutaScreeningRegressor(X, y):
#     X_std = preprocessing.StandardScaler().fit_transform(X)
#     y_max_min = y / max(abs(y.max()), abs(y.min()))  # 归一化
#     estimator = RandomForestRegressor()
#     feat_selector = BorutaPy(estimator, n_estimators='auto')
#     feat_selector.fit(X_std, y_max_min)
#     ranking = feat_selector.ranking_  # 重要性等级, 数字表示, 1最重要, 2一般, 3及以后均不重要
#     return ranking
#

if __name__ == '__main__':
    df = pd.read_excel('peri_apvpdp_Train_zscore.xlsx', sheet_name='Sheet1')
    print(df)
    y = df.iloc[:, -1]  # 标签,最后一列
    X = df.iloc[:, :-1]  # 数据
    print(X)
    print(y)
    # lasso筛选：选择值大的列
    # 举例： y = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4
    ret_lasso = LassoScreening(X, y)
    print(ret_lasso)

    imp = ret_lasso / ret_lasso.sum()  # 求百分比

    print('***********************************')

    # 画图
    #    print(imp)
    #    plt.scatter([i for i in range(len(imp))], imp)
    #    plt.show()

    #保存重要性
save_dict = {}
for k, v in zip(X.columns, imp):
           save_dict[k] = v
print(save_dict)

    #保存特征重要性
df_imp = pd.DataFrame(save_dict, index=[0])
df_imp = df_imp.T
df_imp.insert(0, 'Feature', df_imp.index)
print(df_imp)
df_imp.columns = ['Feature', 'Importance']
# df_imp.to_excel('peri_apvpdp_Train_zscore_Sheet1_lassso.xlsx', index=False)

    #设定阈值
threshold = -1


lasso_feature = X.columns[ret_lasso > threshold]
print(lasso_feature)

    #保存lasso的筛选结果
df_lasso = df.loc[:, [y.name, *lasso_feature]]
# df_lasso.to_excel('peri_apvpdp_Train_zscore_Sheet1_fit.best.min_lassoinput.xlsx',  index=False)


    # ******************************************************************************************
    # boruta筛选  重要性等级, 数字表示, 1最重要, 2一般, 3及以后均不重要
    # ret_boruta = BorutaScreeningClassifier(X, y)
    #  print(ret_boruta)
    #  boruta_feature = X.columns[ret_boruta > 2]
    #  print(boruta_feature)

    # 保存重要性
    # save_dict = {}
    #  for k, v in zip(X.columns, ret_boruta):
    #    save_dict[k] = v
#  print(save_dict)

    # 保存特征重要性
    # df_imp = pd.DataFrame(save_dict, index=[0])
    #  df_imp = df_imp.T
    #  df_imp.insert(0, 'Feature', df_imp.index)
    #  print(df_imp)
    #  df_imp.columns = ['Feature', 'Importance']
    #  df_imp.to_excel('data/feature_importance_boruta.xlsx', index=False)


    # 保存boruta的筛选结果
    #  df_boruta = df.loc[:, [y.name, *boruta_feature]]
    # df_boruta.to_excel('data/breast_cancer_boruta.xlsx', index=False)
#
import pandas as pd
import numpy as np
# exit()
print('------------------------- 权重作图 -------------------------')
import matplotlib.pyplot as plt
#%matplotlib inline
xva = pd.read_excel("fit.best.min.xlsx", sheet_name='fit.best.min', usecols=[0])  # 打开excel文件
yva = pd.read_excel("fit.best.min.xlsx", sheet_name='fit.best.min', usecols=[1])
print(xva)
print(yva)
xva = np.array(xva).flatten()
yva = np.array(yva).flatten()
# xva = np.hsplit(xva, 1)[0]
# yva = np.hsplit(yva, 2)[0]
plt.bar(xva, yva
        , color='lightblue'
        , edgecolor='black'
        , alpha=0.8  #不透明度
        )
plt.xticks(xva,  rotation='30'
           , ha='right'
           , va='top'
           )
plt.xlabel('feature')
plt.ylabel('weight')
plt.show()


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文黑体
# plt.rcParams['axes.unicode_minus'] = False # 负值显示
# plt.barh(xva, yva, height=0.7, color='#008792', edgecolor='#005344') # 更多颜色可参见颜色大全
# plt.xlabel('feature importance') # x 轴
# plt.ylabel('features') # y轴
# plt.title('Feature Importances') # 标题
# for a,b in zip( features_import['importance'],features_import['feature']): # 添加数字标签
#   print(a,b)
#   plt.text(a+0.001, b,'%.3f'%float(a)) # a+0.001代表标签位置在柱形图上方0.001处
# plt.show()
