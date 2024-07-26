import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从CSV文件读取数据，您需要将文件路径替换为实际文件路径
data = pd.read_csv('D:\VETCNOMtrain.csv')

# 设置类别作为索引
data.set_index('VETC', inplace=True)

# 使用seaborn绘制图表
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

# 绘制柱状图
sns.barplot(data=data, palette="pastel")

plt.title('Binary Classification Performance')
plt.ylabel('Scores')
plt.show()






