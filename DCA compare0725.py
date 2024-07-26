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

    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    ax.plot(thresh_group, net_benefit_all, color='black', label='All')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='None')

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1.0)  # 横坐标值区间
    # ax.set_ylim(y_lim_min, y_lim_max)  # adjustify the y axis limitation
    ax.set_ylim(-0.1, 0.7)
    # cursor = Cursor(ax, useblit =Ture)
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'}
    )
    # ax.grid('major')
    ax.spines['right'].set_color((1, 1, 1))
    ax.spines['top'].set_color((1, 1, 1))
    # ax.spines['left'].set_linewidth(5)
    # ax.spines['bottom'].set_linewidth(5)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')  # 图例位置
    legend_font ={'family': 'Times New Roman', 'weight': 'bold'}
    # ax.legend()  # 自动选择图例位置

    fig.savefig(save_path, dpi=300)
    plt.show()


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
    df_train1 = pd.read_excel(
        'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
        sheet_name='CR-train')
    # df_train2 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='CRR-train')
    # df_train3 = pd.read_excel(
    #     'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
    #     sheet_name='Rtrain')

    # 测试集路径
    # test_path = r'../../datasets/breast_cancer/feature_screening/LASSO/test_screen.xlsx'
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

    # 结果保存目录
    # save_dir = r'../../datasets/breast_cancer/machine_learning/MultiModels'
    # os.makedirs(save_dir, exist_ok=True)

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
    X_extest1 = df_extest1.iloc[:, :-1]  # 测试集特征
    y_extest1 = df_extest1.iloc[:, -1]  # 测试集标签
    # X_extest2 = df_extest2.iloc[:, :-1]  # 测试集特征
    # y_extest2 = df_extest2.iloc[:, -1]  # 测试集标签
    # X_extest3 = df_extest3.iloc[:, :-1]  # 测试集特征
    # y_extest3 = df_extest3.iloc[:, -1]  # 测试集标签d

    # 测试集
    # df_test = pd.read_excel(test_path)
    X_test1 = df_test1.iloc[:, :-1]  # 测试集特征
    y_test1 = df_test1.iloc[:, -1]  # 测试集标签
    # X_test2 = df_test2.iloc[:, :-1]  # 测试集特征
    # y_test2 = df_test2.iloc[:, -1]  # 测试集标签
    # X_test3 = df_test3.iloc[:, :-1]  # 测试集特征
    # y_test3 = df_test3.iloc[:, -1]  # 测试集标签d

    print("\n")
    # print('test datasets:\n', df_test1)
    # print('test datasets:\n', df_test2)
    # print('test datasets:\n', df_test3)
    # print('y_test1:', y_test1.values.tolist())
    # print('y_test2:', y_test2.values.tolist())
    # print('y_test3:', y_test3.values.tolist())
    print('训练集DCA')
    # # 在测试集上预测
    # 模型1
    estimator_xgboost = XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5)
    estimator_xgboost.fit(X_train1, y_train1)  # 训练集上建模
    y_score_test_xgboost1 = estimator_xgboost.predict_proba(X_train1)[:, 1]  # 预测概率
    # estimator_xgboost.fit(X_extest1, y_extest1)  # 训练集上建模
    # y_score_test_xgboost2 = estimator_xgboost.predict_proba(X_extest1)[:, 1]  # 预测概率
    # # print('y_score_test_xgboost:', y_score_test_xgboost1.tolist())
    # # print('y_score_test_xgboost:', y_score_test_xgboost2.tolist())
    # # 模型2
    estimator_dt = DecisionTreeClassifier(max_depth=5)
    estimator_dt.fit(X_train1, y_train1)
    y_score_test_dt1 = estimator_dt.predict_proba(X_train1)[:, 1]  # 预测概率
    # estimator_dt.fit(X_extest1, y_extest1)
    # y_score_test_dt2 = estimator_dt.predict_proba(X_extest1)[:, 1]  # 预测概率
    # # print('y_score_test_dt:', y_score_test_dt.tolist())
    #
    # # 模型3
    estimator_rf = RandomForestClassifier(n_estimators=120, max_depth=3, criterion='gini', n_jobs=1)
    estimator_rf.fit(X_train1, y_train1)
    y_score_test_rf1 = estimator_rf.predict_proba(X_train1)[:, 1]  # 预测概率
    # estimator_rf.fit(X_extest1, y_extest1)
    # y_score_test_rf2 = estimator_rf.predict_proba(X_extest1)[:, 1]
    # # print('y_score_test_rf:', y_score_test_rf.tolist())
    # # 模型4
    estimator_lr = LogisticRegression(C=2.0,
                                      # class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                                      # max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                                      # solver='liblinear', tol=0.0001,
                                      # verbose=0, warm_start=False
                                      )
    estimator_lr.fit(X_train1, y_train1)
    y_score_test_lr1 = estimator_lr.predict_proba(X_train1)[:, 1]
    # estimator_lr.fit(X_extest1, y_extest1)
    # y_score_test_lr2 = estimator_lr.predict_proba(X_extest1)[:, 1]
    # estimator_lr.fit(X_train3, y_train3)
    # y_score_test_lr3 = estimator_lr.predict_proba(X_train3)[:, 1]

    # # 模型5
    estimator_svm = LinearSVC(C=2.0,
                              penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,  multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000
                               )
    estimator_svm.fit(X_train1, y_train1)
    y_score_test_svm1 = estimator_svm._predict_proba_lr(X_train1)[:, 1].tolist()
    # estimator_svm.fit(X_extest1, y_extest1)
    # y_score_test_svm2 = estimator_svm._predict_proba_lr(X_extest1)[:, 1].tolist()
    # # 模型6
    estimator_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
    estimator_knn.fit(X_train1, y_train1)
    y_score_test_knn1 = estimator_knn.predict_proba(X_train1)[:, 1]
    # estimator_knn.fit(X_extest1, y_extest1)
    # y_score_test_knn2 = estimator_knn.predict_proba(X_extest1)[:, 1]


    plot_DCA(y_label=y_train1,
             y_scores=[

                 ['DT', y_score_test_dt1],
                 ["KNN", y_score_test_knn1],
                 ["LR", y_score_test_lr1],
                 ['RF', y_score_test_rf1],
                 # ["CRR", y_score_test_lr2],
                 # ["R", y_score_test_lr3],
                 ["SVM", y_score_test_svm1],
                 ['XGBoost', y_score_test_xgboost1],
                 # ['XGBoost2', y_score_test_xgboost2],
                 # ['DT2', y_score_test_dt2],
                 # ['RF2', y_score_test_rf2],
                 # ["lr2", y_score_test_lr2],
                 # ["svm2", y_score_test_svm2],
                 # ["knn2", y_score_test_knn2],
                 # ['XGBoost2', y_score_test_xgboost2],
                 # ['DT2', y_score_test_dt2],
                 # ['RF2', y_score_test_rf2],
             ],
    save_path=os.path.join(save_dir, 'KI67-DCA-train-231207.jpg')
             )
    #


    # print('测试集DCA')
      # estimator_xgboost = XGBClassifier(n_estimators=150, max_depth=4, gamma=0.5)
      # estimator_xgboost.fit(X_test1, y_test1)  # 训练集上建模
      # y_score_test_xgboost1 = estimator_xgboost.predict_proba(X_train1)[:, 1]  # 预测概率
      # estimator_xgboost.fit(X_test2, y_test2)  # 训练集上建模
      # y_score_test_xgboost2 = estimator_xgboost.predict_proba(X_train2)[:, 1]  # 预测概率
    # # print('y_score_test_xgboost:', y_score_test_xgboost1.tolist())
    # # print('y_score_test_xgboost:', y_score_test_xgboost2.tolist())
    # # 模型2
    #   estimator_dt = DecisionTreeClassifier(max_depth=5)
    #   estimator_dt.fit(X_test1, y_test1)
    #   y_score_test_dt1 = estimator_dt.predict_proba(X_train1)[:, 1]  # 预测概率
    #   estimator_dt.fit(X_test2, y_test2)
    #   y_score_test_dt2 = estimator_dt.predict_proba(X_train2)[:, 1]  # 预测概率
    # # print('y_score_test_dt:', y_score_test_dt.tolist())
    #
    # # 模型3
    #   estimator_rf = RandomForestClassifier(n_estimators=120, max_depth=3, criterion='gini', n_jobs=1)
    #   estimator_rf.fit(X_test1, y_test1)
    #   y_score_test_rf1 = estimator_rf.predict_proba(X_train1)[:, 1]  # 预测概率
    #   estimator_rf.fit(X_test2, y_test2)
    #   y_score_test_rf2 = estimator_rf.predict_proba(X_train2)[:, 1]
    # # print('y_score_test_rf:', y_score_test_rf.tolist())
        # # 模型4
    # estimator_lr = LogisticRegression(C=2.0,
    #                                       # class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
    #                                       # max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
    #                                       # solver='liblinear', tol=0.0001,
    #                                       # verbose=0, warm_start=False
    #                                       )
    # estimator_lr.fit(X_train1, y_train1)
    # y_score_train_lr1 = estimator_lr.predict_proba(X_train1)[:, 1]
    # estimator_lr.fit(X_train2, y_train2)
    # y_score_train_lr2 = estimator_lr.predict_proba(X_train2)[:, 1]
    # estimator_lr.fit(X_train3, y_train3)
    # y_score_train_lr3 = estimator_lr.predict_proba(X_train3)[:, 1]
    #
    # estimator_lr.fit(X_test1, y_test1)
    # y_score_test_lr1 = estimator_lr.predict_proba(X_test1)[:, 1]
    # estimator_lr.fit(X_test2, y_test2)
    # y_score_test_lr2 = estimator_lr.predict_proba(X_test2)[:, 1]
    # estimator_lr.fit(X_test3, y_test3)
    # y_score_test_lr3 = estimator_lr.predict_proba(X_test3)[:, 1]
    #
    # estimator_lr.fit(X_extest1, y_extest1)
    # y_score_extest_lr1 = estimator_lr.predict_proba(X_extest1)[:, 1]
    # estimator_lr.fit(X_extest2, y_extest2)
    # y_score_extest_lr2 = estimator_lr.predict_proba(X_extest2)[:, 1]
    # estimator_lr.fit(X_extest3, y_extest3)
    # y_score_extest_lr3 = estimator_lr.predict_proba(X_extest3)[:, 1]


    # # 模型5
    # estimator_svm = LinearSVC(C=2.0,)
    # estimator_svm.fit(X_test1, y_test1)
    # y_score_test_svm1 = estimator_svm._predict_proba_lr(X_train1)[:, 1].tolist()
    # estimator_svm.fit(X_test2, y_test2)
    # y_score_test_svm2 = estimator_svm._predict_proba_lr(X_train2)[:, 1].tolist()
    # # 模型6
    # estimator_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric_params=None, n_jobs=None)
    # estimator_knn.fit(X_test1, y_test1)
    # y_score_test_knn1 = estimator_knn.predict_proba(X_train1)[:, 1]
    # estimator_knn.fit(X_test2, y_test2)
    # y_score_test_knn2 = estimator_knn.predict_proba(X_train2)[:, 1]
    #

    # plot_DCA(y_label=y_train1,
    #          y_scores=[
    #              ['XGBoost1', y_score_test_xgboost1],
    #              ['DT1', y_score_test_dt1],
    #              ['RF1', y_score_test_rf1],
    #              ["CR", y_score_train_lr1],
    #              #
    #              # ["CRR", y_score_train_lr2],
    #              #
    #              # ["Rad_score", y_score_train_lr3],
    #
    #              ["svm1", y_score_test_svm1],
    #              ["knn1", y_score_test_knn1],
    #              # ['XGBoost2', y_score_test_xgboost2],
    #              # ['DT2', y_score_test_dt2],
    #              # ['RF2', y_score_test_rf2],
    #              # ["lr2", y_score_test_lr2],
    #              # ["svm2", y_score_test_svm2],
    #              # ["knn2", y_score_test_knn2],
    #              # ['XGBoost2', y_score_test_xgboost2],
    #              # ['DT2', y_score_test_dt2],
    # #              # ['RF2', y_score_test_rf2],
    #          ],
    #          # save_path=os.path.join(save_dir, '20221119ki76train.jpg'))
    #          save_path = os.path.join(save_dir, '20221119ki76train.jpg'))
