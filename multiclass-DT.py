import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# 从 .xlsx 文件中读取数据
# data = pd.read_excel('D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/ki670819-R-mse.xlsx',
#     sheet_name='CR-train')  # 将 '文件路径.xlsx' 替换为你的文件路径和文件名

data = pd.read_excel(
    'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/XSS-DT.xlsx',
    sheet_name='20240620')
# exit()
# 提取特征和目标变量
X = data.iloc[:, :-1]  # 假设最后一列是目标变量，如果不是，请适当调整索引
# y = data.iloc[:, -1]

y = data.iloc[:, -1].astype(str)



# 构建决策树模型并调整节点参数
model = DecisionTreeClassifier(criterion='gini', max_depth=2
                               , min_samples_split=2)
model2 = DecisionTreeClassifier(criterion='gini', max_depth=2
                               , min_samples_split=2)
# model = LogisticRegression(C=20.0)
model.fit(X, y)

# ---------------预测测试数据(二分类)---------------

# y_pred = model.predict(X)
# # 计算混淆矩阵
# cm = confusion_matrix(y, y_pred)
# # 从混淆矩阵中提取真正例、假正例、真反例和假反例的数量
# tn, fp, fn, tp = cm.ravel()
# # 计算准确度
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# # 计算灵敏度（真正例率）
# sensitivity = tp / (tp + fn)
# # 计算特异度（真反例率）
# specificity = tn / (tn + fp)
# # 打印结果
# print("Accuracy:", accuracy)
# print("Sensitivity:", sensitivity)
# print("Specificity:", specificity)
# exit()
# -----------预测测试数据(多分类)-------------------------------------
# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
# 预测训练数据(多分类)
y_pred = model.predict(X)
# 计算混淆矩阵
cm = confusion_matrix(y, y_pred)
# 定义类别文本
class_labels = ['NC', 'Cirrhosis-NR', 'Cirrhosis-RD']
# class_labels = ['NC', 'Cirrhosis-NR']
# 绘制混淆矩阵热力图
plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 18}, xticklabels=class_labels, yticklabels=class_labels)  # 设置字体大小为12
plt.xticks(fontsize=14, fontweight='bold')  # 设置横坐标标签的字体大小和粗细
plt.yticks(fontsize=14, fontweight='bold')  # 设置纵坐标标签的字体大小和粗细
plt.xlabel('Predicted', fontsize=15, fontweight='bold')  # 设置x轴标签字体大小为14
plt.ylabel('True', fontsize=15, fontweight='bold')  # 设置y轴标签字体大小为14
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')  # 设置标题字体大小为16

# 计算每个类别的预测灵敏度、特异度和准确性
sensitivity = []
specificity = []
accuracy = []
for i in range(len(model.classes_)):
    true_positive = cm[i, i]
    false_negative = sum(cm[i, :]) - true_positive
    false_positive = sum(cm[:, i]) - true_positive
    true_negative = cm.sum() - true_positive - false_negative - false_positive

    sensitivity.append(true_positive / (true_positive + false_negative))
    specificity.append(true_negative / (true_negative + false_positive))
    accuracy.append((true_positive + true_negative) / cm.sum())

# 显示每个类别的预测灵敏度、特异度和准确性
for i in range(len(model.classes_)):
    print("Class {}: Sensitivity = {:.4f}, Specificity = {:.4f}, Accuracy = {:.4f}".format(model.classes_[i],
                                                                                           sensitivity[i],
                                                                                           specificity[i], accuracy[i]))

# 计算平均准确性
avg_accuracy = accuracy_score(y, y_pred)
print("Average Accuracy: {:.4f}".format(avg_accuracy))
# 预测类别概率
y_pred_prob = model.predict_proba(X)
# 计算每个类别的真阳率、假阳率和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(model.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y == model.classes_[i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制ROC曲线
plt.figure()
for i in range(len(model.classes_)):
    plt.plot(fpr[i], tpr[i], label='class {} (AUC = {:.2f})'.format(model.classes_[i], roc_auc[i]))

plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
# class_labels = ['NC', 'Cirrhosis-NR']  # 自定义类别标签
class_labels = ['NC', 'Cirrhosis-NR', 'Cirrhosis-RD']  # 自定义类别标签
plt.legend(labels=class_labels, loc="lower right", prop={'size': 15})

plt
# 打印每个类别的AUC值
for i in range(len(model.classes_)):
    print("Class {}: AUC = {:.4f}".format(model.classes_[i], roc_auc[i]))
# 计算准确度
accuracy = accuracy_score(y, y_pred)
# 计算精确度（Precision）
precision = precision_score(y, y_pred, average='weighted')
# 计算召回率（Recall）
recall = recall_score(y, y_pred, average='weighted')
# 计算F1-score
f1 = f1_score(y, y_pred, average='weighted')
# 打印结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
# 生成一组阈值
thresholds = np.linspace(0, 1, num=100)
# 预测概率
y_pred_prob = model.predict_proba(X)

# 计算每个类别的精确度和召回率
precision = dict()
recall = dict()
for i in range(len(model.classes_)):
    precision[i], recall[i], _ = precision_recall_curve(y == model.classes_[i], y_pred_prob[:, i])
# 找到F1得分最高的阈值点
f1_scores = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(model.classes_))]
best_thresholds = [thresholds[np.argmax(f1)] for f1 in f1_scores]
# 获取F1得分最高的阈值点的数值
best_threshold_values = [y_pred_prob[:, i][y_pred_prob[:, i] >= threshold].min() for i, threshold in enumerate(best_thresholds)]
# 打印F1得分最高阈值点的数值
for i, value in enumerate(best_threshold_values):
    print("Class {}: Best Threshold Value = {:.4f}".format(model.classes_[i], value))
# 绘制多分类P-R曲线
plt.figure()
for i in range(len(model.classes_)):
    plt.plot(recall[i], precision[i], label='class {}'.format(model.classes_[i]))
# 绘制平衡点
for i, threshold in enumerate(best_thresholds):
    idx = np.where(thresholds == threshold)[0][0]
    plt.scatter(recall[i][idx], precision[i][idx], marker='o', color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Multi-class)')
plt.legend(loc="lower left")
# 修改图例
class_labels = ['normal', 'cirrhosis with normal renal function', 'cirrhosis with normal renal dysfunction']  # 自定义类别标签
plt.legend(labels=class_labels, loc="lower left")

# 可视化决策树
fig = plt.figure(figsize=(12, 8))
_ = tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
# 展示图形
plt.show()
