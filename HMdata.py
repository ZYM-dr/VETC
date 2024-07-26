import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('ggplot')
#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False
#df =pd.read_excel('HMdata.xlsx',sheet_name='Sheet2')
#print(df)
#df_new = df.corr()
#print(df_new)#打印值
#df_new.to_excel('heatmapdata4.xlsx')
#df3 = pd.read_excel('4.xlsx')
df1 =pd.read_excel('CD34heatmap.xlsx',sheet_name='Sheet3')
# print(df1)
df1 =df1.astype(float)
# print(df1)
plt.figure(dpi=300,figsize=(5, 5))

plt.figure(1)
features = ['VETC', 'MVI', 'Grade', 'Ki-67', 'CK19', 'GPC3']
#sns.heatmap(df_new,annot=False, vmax=1, square=True)#绘制new_df的矩阵热力图
#data = pd.DataFrame(df1,index = features,columns=features)
#data.to_excel('LZX.xlsx')
#data1 = pd.read_excel('LZX.xlsx')
#print(data), cmap='YlOrRd_r'
ax = sns.heatmap(df1, xticklabels=features, yticklabels=features,vmax=1, square=True,annot=True, cmap='YlGnBu', annot_kws={"fontsize":5})
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# 设置横纵坐标的名称及热力图名称以及对应字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 7,
         'color': 'black'
         }

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=7)
plt.show()#显示图片
