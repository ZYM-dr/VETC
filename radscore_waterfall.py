import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
# 读取CSV文件
data = pd.read_excel('poolmap.xlsx', sheet_name='Validation1zl+slyy')

# 按照'radscore'列对整个数据进行排序
data.sort_values('radscore', ascending=False, inplace=True)

# 绘制瀑布图
fig, ax = plt.subplots(figsize=(8, 6))

# 设置正负样本的颜色和宽度
pos_color = 'orange'
neg_color = 'skyblue'
# width = 0.95
pos_width = 0.95
neg_width = 0.95
# bar_spacing = 0.1  # 设置条形图之间的间距
# 绘制数据
x = range(len(data))
y = data['radscore']
colors = [pos_color if label == 1 else neg_color for label in data['label']]
ax.bar(x, y, color=colors)

# 添加图例
ax.bar(0, 0, color=pos_color, label='Positive samples ')
ax.bar(0, 0, color=neg_color, label='Negative samples')
ax.legend()

plt.show()
exit()
# 分离正负样本
pos_data = data[data['label'] == 1]
neg_data = data[data['label'] == 0]
# 绘制瀑布图
fig, ax = plt.subplots(figsize=(8, 6))
data.sort_values('radscore', inplace=True)

# pos_data.sort_values('radscore', ascending=False, inplace=True)
# neg_data.sort_values('radscore', ascending=False, inplace=True)
# # 设置正负样本的颜色和宽度
pos_color = 'red'
neg_color = 'blue'
pos_width = 0.95
neg_width = 0.95

# 计算正负样本的x轴位置和对应的病人编号
pos_x = [i*2 for i in range(len(pos_data))]
# pos_patients = list(pos_data['PatientID'])
neg_x = [i*2+1 if i < len(pos_data) else i+len(pos_data) for i in range(len(neg_data))]
# neg_patients = list(neg_data['PatientID'])
# 绘制正样本
y1=[]
for i in range(len(pos_data)):
    x = pos_x[i]
    y1 .append(pos_data.iloc[i]['radscore'])
ax.bar(pos_x, y1, color=pos_color,  label='Positive samples')
    # ax.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=6)

# 绘制负样本
y2=[]
for i in range(len(neg_data)):
    x = neg_x[i]
    y2.append(neg_data.iloc[i]['radscore'])
ax.bar(neg_x, y2, color=neg_color,  label='Negative samples')
    # ax.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=6)
ax.legend()
plt.show()
# 设置x轴刻度和标签
# x_ticks = pos_x + neg_x
# ax.set_xticks(x_ticks)
# ax.set_xticklabels([f"P{i+1}" for i in range(len(x_ticks))], fontsize=8, rotation=90)
# ax.set_xticks([i for i in range(len(pos_data)+len(neg_data))])
# ax.set_xticklabels([f"P{i+1}" for i in range(len(pos_data))] + [f"N{i+1}" for i in range(len(neg_data))], fontsize=8)
# ax.tick_params(axis='x', length=0)
# xticks = pos_x + neg_x
# xticklabels = pos_patients + neg_patients
# ax.set_xticks(xticks)
# ax.set_xticklabels('PatientID')
# ax.set_xlabel('PatientID')
# ax.tick_params(axis='x', length=0)

# # 设置y轴刻度和标签
# # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_ylabel('Radscore')
# 设置图例
# ax.legend([plt.Rectangle((0,0),1,1,fc=pos_color, edgecolor = 'none'), plt.Rectangle((0,0),1,1,fc=neg_color,  edgecolor = 'none')], ['Positive samples', 'Negative samples'], fontsize=8)
# 展示图像
# plt.tight_layout()
# plt.figure(figsize=(3, 6.5))
# plt.show()

# 分离正负样本
# pos_data = data[data['label'] == 1]
# neg_data = data[data['label'] == 0]

# 绘制瀑布图
# fig, ax = plt.subplots(figsize=(8, 6))
# pos_data.sort_values('radscore', ascending=False, inplace=True)
# neg_data.sort_values('radscore', ascending=False, inplace=True)
# ax.bar(range(len(pos_data)), pos_data['radscore'], color='r',label='Positive samples')
# ax.bar(range(len(neg_data)), neg_data['radscore'], color='b', label='Negative samples')
# ax.set_xticks([])
# ax.set_xticklabels('PatientID')
# ax.set_ylabel('Radscore')
# ax.legend()
# plt.figure(figsize=(6, 6.5))
# plt.show()
