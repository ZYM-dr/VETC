import pandas as pd
from imblearn import over_sampling, under_sampling

# pip install imblearn

"""
数据采样，针对离散数据（标签是离散的、分类）

采样时，确保数据中不含有空值（数据填充之后再采样）

# 只对训练集做采样  测试集验证集不能采样



"""


def randomOverSampler(X, y):
    """随机上采样 #样本量少的增多
    X: 数据
    y: 标签
    """
    sampler = over_sampling.RandomOverSampler()
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    X_resampled.insert(0, y_resampled.name, y_resampled)  # 将标签列放在第一列
    return X_resampled


def SMOTE(X, y):
    """SMOTE上采样
    X: 数据
    y: 标签
    """
    sampler = over_sampling.SMOTE()
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    X_resampled.insert(0, y_resampled.name, y_resampled)  # 将标签列放在第一列
    return X_resampled


#def randomUnderSampler(X, y):
    """随机下采样  #样本量多的减少
    X: 数据
    y: 标签
    """
#    sampler = under_sampling.RandomUnderSampler()
#    X_resampled, y_resampled = sampler.fit_resample(X, y)
#    X_resampled.insert(0, y_resampled.name, y_resampled)  # 将标签列放在第一列
#    return X_resampled


if __name__ == '__main__':
    # 上采样：数据增多
    # 下采样：数据减少

    df = pd.read_excel('172lasso.xlsx', sheet_name='Sheet4')
    print(df)
    y = df.iloc[:, 0]  # 取出标签
    X = df.iloc[:, 1:]  # 取出数据
    print(y)
    print(X)

    # 随机上采样  复制操作
    over = randomOverSampler(X, y)  # 随机上采样
    print(over)
    over.to_excel('172Sampler.xlsx',
                  index=False)

    # 随机下采样  删除操作
#    under = randomUnderSampler(X, y)
#    print(under)
#    under.to_excel('data/exp_imputing_mean_under.xlsx',
#                   index=False)

    # SMOTE上采样   smote算法  新的数据是由原数据预测的
#    smote = SMOTE(X, y)
#    smote.to_excel('data/exp_imputing_mean_smote.xlsx',
#                   index=False)

    #
    # smote = SMOTE(X, y)
    # print(smote)
    #
    # under = randomUnderSampler(X, y)
    # print(under)

    # 尝试将数据保存为excel和csv两种格式

    # smote.to_excel()
