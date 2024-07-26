import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# 从 Excel 文件中读取数据
file_path = 'D:/DOCZ 2021-2025/Gd-EOB-DTPA ralated/Gd-EOB-DTPA DATA/RadiomicDATA20220404/yushiyan1/XSS-DT-PRCURVE.xlsx'  # 替换为你的文件路径和文件名
excel_data = pd.read_excel(file_path, sheet_name=None)

# 创建图形对象
plt.figure()

# 遍历每个 sheet
for sheet_name, data in excel_data.items():
    # 提取特征和目标变量
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(str)

    # 构建决策树模型
    model = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
    model.fit(X, y)

    # 生成一组阈值
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 计算 P-R 曲线
    y_pred_prob = model.predict_proba(X)
    precision, recall, _ = precision_recall_curve(y, y_pred_prob[:, 1])

    # 绘制 P-R 曲线，并添加到图形对象中
    plt.plot(recall, precision, label=sheet_name)

# 设置图例和标签
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()

# 展示图形
plt.show()