import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
'有问题没用'
def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def plot_DCA_multiple_models(y_labels, y_scores_list, model_names, save_path):
    fig, ax = plt.subplots()
    thresh_group = np.arange(0, 1, 0.01)
    y_lim_min = 0  # 图像纵坐标最小值
    y_lim_max = 0  # 图像纵坐标最大值
    for y_label, y_scores, model_name in zip(y_labels, y_scores_list, model_names):
        net_benefit_model = calculate_net_benefit_model(thresh_group, y_scores, y_label)
        ax.plot(thresh_group, net_benefit_model, label=model_name)
        print(net_benefit_model.min(), net_benefit_model.max())

        # 调整纵坐标区间
        if y_lim_min > net_benefit_model.min() - 0.05:
            y_lim_min = net_benefit_model.min() - 0.05
        if y_lim_max < net_benefit_model.max() + 0.05:
            y_lim_max = net_benefit_model.max() + 0.05
    font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': 10, }
    plt.tick_params(labelsize=10)  # 坐标轴刻度数字大小
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]

    # 计算所有模型的组合标签
    all_labels = np.concatenate(y_labels)
    net_benefit_all = calculate_net_benefit_model(thresh_group, np.concatenate(y_scores_list), all_labels)
    ax.plot(thresh_group, net_benefit_all, color='black', label='All')

    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='None')

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0.0, 1.00)  # 横坐标值区间
    # ax.set_ylim(y_lim_min, y_lim_max)  # adjustify the y axis limitation
    ax.set_ylim(-0.10, 0.4)
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    # ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(prop=font1, loc='upper right')  # 图例位置
    # ax.legend()  # 自动选择图例位置


    plt.savefig(save_path, dpi=300)
    plt.show()

# 读取数据
df_train1 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Train')
df_train2 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Intest')
df_train3 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Extest')

# 训练集
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
y_train1 = df_train1.iloc[:, -1]  # 训练集标签
X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
y_train2 = df_train2.iloc[:, -1]  # 训练集标签
X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
y_train3 = df_train3.iloc[:, -1]  # 训练集标签

# 模型训练
estimator_lr = LogisticRegression(C=2.0,
                                  class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                                  max_iter=100, multi_class='ovr', n_jobs=10, penalty='l2', random_state=None,
                                  solver='liblinear', tol=0.0001,
                                  verbose=0, warm_start=False
                                  )

estimator_lr.fit(X_train1, y_train1)
y_score_train_lr1 = estimator_lr.predict_proba(X_train1)[:, 1]
estimator_lr.fit(X_train2, y_train2)
y_score_train_lr2 = estimator_lr.predict_proba(X_train2)[:, 1]
estimator_lr.fit(X_train3, y_train3)
y_score_train_lr3 = estimator_lr.predict_proba(X_train3)[:, 1]

# 调用绘制函数
y_labels = [y_train3, y_train2, y_train1]
y_scores_list = [y_score_train_lr3, y_score_train_lr2, y_score_train_lr1]
model_names = ["Train", "Internal test", "External test"]
save_path = os.path.join("MultiModels", 'VETC-dca-2024-3-1.jpg')

plot_DCA_multiple_models(y_labels, y_scores_list, model_names, save_path)







exit()
'test'
def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def plot_DCA_multiple_models(y_labels, y_scores_list, model_names, save_path):
    """
    绘制多个模型的决策曲线
    :param y_labels: list
        包含多个标签数组的列表，每个标签数组对应一个模型的真实标签
    :param y_scores_list: list
        包含多个预测概率列表的列表，每个预测概率列表对应一个模型的预测概率
    :param model_names: list
        模型的名称列表，用于图例
    :param save_path: str
        图片保存路径
    """
    fig, ax = plt.subplots()
    thresh_group = np.arange(0, 1, 0.01)
    y_lim_min = 0  # 图像纵坐标最小值
    y_lim_max = 0  # 图像纵坐标最大值
    for y_label, y_scores, model_name in zip(y_labels, y_scores_list, model_names):
        net_benefit_model = calculate_net_benefit_model(thresh_group, y_scores, y_label)
        ax.plot(thresh_group, net_benefit_model, label=model_name)
        print(net_benefit_model.min(), net_benefit_model.max())

        # 调整纵坐标区间
        if y_lim_min > net_benefit_model.min() - 0.05:
            y_lim_min = net_benefit_model.min() - 0.05
        if y_lim_max < net_benefit_model.max() + 0.05:
            y_lim_max = net_benefit_model.max() + 0.05
    font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': 10, }
    plt.tick_params(labelsize=10)  # 坐标轴刻度数字大小
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]
    net_benefit_all = calculate_net_benefit_model(thresh_group, y_label)
    ax.plot(thresh_group, net_benefit_all, color='black', label='All')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='None')

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0.0, 1.00)  # 横坐标值区间
    # ax.set_ylim(y_lim_min, y_lim_max)  # adjustify the y axis limitation
    ax.set_ylim(-0.10, 0.4)
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    # ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(prop=font1, loc='upper right')  # 图例位置
    # ax.legend()  # 自动选择图例位置

    # # fig.savefig(save_path, dpi=300)
    # # plt.show()
    #
    #
    #
    # # # 添加完美校准曲线和无信息曲线
    # # ax.plot([0, 1], [0, 0], color='black', linestyle=':', label='None')
    # # ax.plot([0, 1], [1, 1], color='black', linestyle='--', label='Perfect')
    #
    #
    #
    # # 其他设置
    # ax.set_xlim(0.0, 1.0)
    # ax.set_ylim(-0.1, 0.4)
    # ax.set_xlabel('Threshold Probability',
    #     fontdict={'family': 'Times New Roman', 'fontsize': 15})
    # ax.set_ylabel('Net Benefit',
    #     fontdict={'family': 'Times New Roman', 'fontsize': 15})
    # ax.legend()
    # plt.savefig(save_path, dpi=300)
    plt.show()

# 读取数据
df_train1 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Train')
df_train2 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Intest')
df_train3 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Extest')

# 训练集
X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
y_train1 = df_train1.iloc[:, -1]  # 训练集标签
X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
y_train2 = df_train2.iloc[:, -1]  # 训练集标签
X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
y_train3 = df_train3.iloc[:, -1]  # 训练集标签

# 模型训练
estimator_lr = LogisticRegression(C=2.0,
                                  class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                                  max_iter=100, multi_class='ovr', n_jobs=10, penalty='l2', random_state=None,
                                  solver='liblinear', tol=0.0001,
                                  verbose=0, warm_start=False
                                  )

estimator_lr.fit(X_train1, y_train1)
y_score_train_lr1 = estimator_lr.predict_proba(X_train1)[:, 1]
estimator_lr.fit(X_train2, y_train2)
y_score_train_lr2 = estimator_lr.predict_proba(X_train2)[:, 1]
estimator_lr.fit(X_train3, y_train3)
y_score_train_lr3 = estimator_lr.predict_proba(X_train3)[:, 1]

# 调用绘制函数
y_labels = [y_train3, y_train2, y_train1]
y_scores_list = [y_score_train_lr3, y_score_train_lr2, y_score_train_lr1]
model_names = ["Train", "Internal test", "External test"]
save_path = os.path.join("MultiModels", 'VETC-dca-2024-3-1.jpg')

plot_DCA_multiple_models(y_labels, y_scores_list, model_names, save_path)