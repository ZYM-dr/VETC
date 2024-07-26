import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
# from sklearn.metrics import dca_curve
import scikitplot as skplt
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC



def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(y_label, y_scores: list, save_path):
# def plot_DCA_multiple_models(y_labels, y_scores_list, model_names, save_path):
    """
    多个模型的二分类决策曲线
    :param y_label: np.ndarray or list
        标签
    :param y_scores: list
        预测的概率, 格式是 [[name1, y_score1], [name2, y_score2], ...]
        name*: 曲线名称
        y_score*: 预测的概率
    :param save_path: str
        图片保存路径
    """
    fig, ax = plt.subplots()
    thresh_group = np.arange(0, 1, 0.01)

    y_lim_min = 0  # 图像纵坐标最小值
    y_lim_max = 0  # 图像纵坐标最大值

    for name, y_score in y_scores:
        net_benefit_model = calculate_net_benefit_model(thresh_group, y_score, y_label)
        ax.plot(thresh_group, net_benefit_model, label=name)
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
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    ax.plot(thresh_group, net_benefit_all, color='black', label='All')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='None')

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0.0, 0.8)  # 横坐标值区间
    # ax.set_ylim(y_lim_min, y_lim_max)  # adjustify the y axis limitation
    ax.set_ylim(-0.10, 0.45)
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman', 'weight': 'bold','fontsize': 15}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'weight': 'bold','fontsize': 15}
    )
    # ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(prop=font1,loc='upper right')  # 图例位置
    # ax.legend()  # 自动选择图例位置

    fig.savefig(save_path, dpi=300)
    # plt.show()


if __name__ == '__main__':
    """
    多个模型DCA曲线比较
    """
    # 训练集路径
    # train_path = r'../../datasets/breast_cancer/feature_screening/LASSO/train_screen.xlsx'
    # # 测试集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
    # # 训练集路径
    # train_path = r'171testandtrain.xlsx'
    # df_train1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='11-23-fit.best.lse')
    df_train1 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Train')
    df_train2 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Intest')
    df_train3 = pd.read_excel('VETCmodel_data.xlsx', sheet_name='Extest')

    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_train')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR_train')
    # df_train3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIPR_train')CR-validaton1
    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR-validaton1')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR-validaton1')

    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_CR')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_IR')
    # df_train3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_PR')
    # df_train4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_IPR')
    # df_train5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_CRIR')
    # df_train6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='intest_CRIPR')

    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_CR')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_IR')
    # df_train3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_PR1')
    # df_train4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_IPR1')
    # df_train5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_CRIR')
    # df_train6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='train_CRIPR')
    #
    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_CR')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_IR')
    # df_train3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_PR')
    # df_train4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_IPR')
    # df_train5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_CRIR')
    # df_train6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton1_CRIPR')
    #
    # df_train1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_CR')
    # df_train2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_IR')
    # df_train3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_PR')
    # df_train4 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_IPR')
    # df_train5 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_CRIR')
    # df_train6 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='validaton2_CRIPR')
    # 测试集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
    # df_test1 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='Validation11-23-fit.best.lse')
    # df_test2 = pd.read_excel('QST-23-11-18-CB.xlsx', sheet_name='2Validation11-23-fit.best.lse')
    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR_test')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR_test')
    # df_test3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIPR_test')
    # df_test1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR-validaton2')
    # df_test2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRIR-validaton2')

    # 外部验证集集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
    # df_extest1 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CR-test')
    # df_extest2 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='CRR-test')
    # df_extest3 = pd.read_excel('VETC_RADIOMICS.xlsx', sheet_name='Rtest')
    # 结果保存目录
    save_dir = r'MultiModels'
    os.makedirs(save_dir, exist_ok=True)

    # 结果保存目录
    # save_dir = r'../../datasets/breast_cancer/machine_learning/MultiModels'
    # os.makedirs(save_dir, exist_ok=True)

    # 训练集
    # df_train = pd.read_excel(train_path)
    X_train1 = df_train1.iloc[:, :-1]  # 训练集特征
    y_train1 = df_train1.iloc[:, -1]  # 训练集标签
    X_train2 = df_train2.iloc[:, :-1]  # 训练集特征
    y_train2 = df_train2.iloc[:, -1]  # 训练集标签
    X_train3 = df_train3.iloc[:, :-1]  # 训练集特征
    y_train3 = df_train3.iloc[:, -1]  # 训练集标签
    # X_train4 = df_train4.iloc[:, :-1]  # 训练集特征
    # y_train4 = df_train4.iloc[:, -1]  # 训练集标签
    # X_train5 = df_train5.iloc[:, :-1]  # 训练集特征
    # y_train5 = df_train5.iloc[:, -1]  # 训练集标签
    # X_train6 = df_train6.iloc[:, :-1]  # 训练集特征
    # y_train6 = df_train6.iloc[:, -1]  # 训练集标签
    # print('train datasets:\n', df_train)
    # print('y_train:', y_train1.values.tolist())

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
    # print('test datasets:\n', df_test)
    # print('y_test:', y_test1.values.tolist())
    print('训练集DCA')
    # # 在测试集上预测

# 模型4
# estimator_lr = LogisticRegression(C=2.0,
#                                       class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
#                                       max_iter=100, multi_class='ovr', n_jobs=10, penalty='l2', random_state=None,
#                                       solver='liblinear', tol=0.0001,
#                                       verbose=0, warm_start=False
#                                       )
estimator_lr = LogisticRegression(C=100.0,
                                      class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                                      max_iter=100, multi_class='ovr', n_jobs=100, penalty='l2', random_state=None,
                                      solver='liblinear', tol=0.0001,
                                      verbose=0, warm_start=False
                                      )
estimator_lr.fit(X_train1, y_train1)
y_score_train_lr1 = estimator_lr.predict_proba(X_train1)[:, 1]
estimator_lr.fit(X_train2, y_train2)
y_score_train_lr2 = estimator_lr.predict_proba(X_train2)[:, 1]
estimator_lr.fit(X_train3, y_train3)
y_score_train_lr3 = estimator_lr.predict_proba(X_train3)[:, 1]
# estimator_lr.fit(X_train4, y_train4)
# y_score_train_lr4 = estimator_lr.predict_proba(X_train4)[:, 1]
# estimator_lr.fit(X_train5, y_train5)
# y_score_train_lr5 = estimator_lr.predict_proba(X_train5)[:, 1]
# estimator_lr.fit(X_train6, y_train6)
# y_score_train_lr6 = estimator_lr.predict_proba(X_train6)[:, 1]



# 绘制三个训练集的决策曲线

# plot_DCA(y_label=y_train3,
#              y_scores=[
#                  ["CR", y_score_train_lr1],
#                  ["IR", y_score_train_lr2],
#                  ["PR", y_score_train_lr3],
#                  ["IPR", y_score_train_lr4],
#                  ["CRIR", y_score_train_lr5],
#                  ["CRIPR", y_score_train_lr6],
#
#              ],
# save_path = os.path.join(save_dir, 'VETC-RAD-dca-Validation12.jpg'))
# dca_curve_data =dca_curve(y_label=y_train2, y_proba)

# save_path = os.path.join(save_dir, 'VETC-RAD-dca-Validation12.jpg')

# plot_DCA(y_label=y_train1, y_scores=[["External test", y_score_train_lr1],],save_path=os.path.join(save_dir, 'DCA-20240301-V1.jpg'))
# plot_DCA(y_label=y_train2, y_scores=[["Internal test", y_score_train_lr2],],save_path=os.path.join(save_dir, 'dca-20240301-V2.jpg'))
plot_DCA(y_label=y_train3, y_scores=[["Train", y_score_train_lr3],],save_path=os.path.join(save_dir, 'dca-20240301-V3.jpg'))
# save_path=os.path.join(save_dir, 'dca-RAD-20240119-V2.jpg'))
plt.show()

exit()
print('测试集DCA')

# 模型4
estimator_lr = LogisticRegression(C=2.0,)
estimator_lr.fit(X_test1, y_test1)
y_score_test_lr1 = estimator_lr.predict_proba(X_test1)[:, 1]
estimator_lr.fit(X_test2, y_test2)
y_score_test_lr2 = estimator_lr.predict_proba(X_test2)[:, 1]
# estimator_lr.fit(X_test3, y_test3)
# y_score_test_lr3 = estimator_lr.predict_proba(X_test3)[:, 1]

plot_DCA(y_label=y_test2,
         y_scores=[

             ["CR", y_score_test_lr1],
             ["CRIR", y_score_test_lr2],
             # ["CRIPR", y_score_test_lr3],
         ],
         save_path=os.path.join(save_dir, 'VETC-RAD-dca-Validation12.jpg'))
