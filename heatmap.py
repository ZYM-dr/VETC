import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#设置绘图风格
plt.style.use('ggplot')
#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False
#" 读取数据
#Sales = pd.read_excel(r'.xlsx')
# 根据交易日期，衍生出年份和月份字段
#Sales['year'] = Sales.Date.dt.year
#Sales['month'] = Sales.Date.dt.month
# 统计每年各月份的销售总额
#Summary = Sales.pivot_table(index = 'month', columns = 'year', values = 'Sales', aggfunc = np.sum)
#打印销售额的列联表格式
#print(Summary.head(13))
# 绘制热力图
#sns.heatmap(data = Summary, # 指定绘图数据
            #cmap = 'PuBuGn', # 指定填充色
            #linewidths = .1, # 设置每个单元格边框的宽度
            #annot = True, # 显示数值
            #fmt = '.1e' # 以科学计算法显示数据
           # )"
#添加标题
#plt.title('每年各月份销售总额热力图')
# 显示图形
#plt.show()

#df =pd.read_excel('.xlsx')
#print(df)
#df_new = df.corr()
#print(df_new)#打印值
#df_new.to_excel('CD34heatmapdata.xlsx')
# df3 = pd.read_excel('.xlsx')
df1 =pd.read_excel(r'CD34heatmapdata.xlsx', sheet_name='Sheet2')
df1 =df1.astype(float)
#df1['乙肝或丙肝病毒感染'] = df1['乙肝或丙肝病毒感染'].astype(float)
#df1['TBIL（nmol/L）'] = df1['TBIL（nmol/L）'].astype(float)
#df1['ALB（g/l)'] = df1['ALB（g/l)'].astype(float)
#df1['CA199(ng/ml)（U/ml）'] = df1['CA199(ng/ml)（U/ml）'].astype(float)
#df1['CEA(ng/ml)0-5'] = df1['CEA(ng/ml)0-5'].astype(float)
#df1['最大径'] = df1['最大径'].astype(float)
#df1['肿瘤边界'] = df1['肿瘤边界'].astype(float)
#df1['动脉期瘤周强化'] = df1['动脉期瘤周强化'].astype(float)
#df1['瘤内坏死'] = df1['瘤内坏死'].astype(float)
#df1['肿瘤破裂'] = df1['肿瘤破裂'].astype(float)
#df1['静脉癌栓'] = df1['静脉癌栓'].astype(float)
#df1['肝硬化'] = df1['肝硬化'].astype(float)
#df1['瘤内动脉'] = df1['瘤内动脉'].astype(float)
#df1['瘤周低密度环'] = df1['瘤周低密度环'].astype(float)
#df1['瘤与肝脏边界的低密度影'] = df1['瘤与肝脏边界的低密度影'].astype(float)
#df1['RIV'] = df1['RIV'].astype(float)
#def string_to_float(str):
    #return float(str)

#import seaborn as sns
#引入seaborn库
#plt.figure(1)
#sns.heatmap(df3,annot=False, vmax=1, square=True)#绘制new_df的矩阵热力图
#plt.show()#显示图片

#a = np.random.rand(4,3)
#fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
#sns.heatmap(pd.DataFrame(np.around(df.corr('spearman')), columns = ['GPC3', 'Ki67-14', 'CD34','ARG1', 'cK19'], index = ['GPC3', 'Ki67-14', 'CD34','ARG1', 'cK19'])), annot = True, vmax = 1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(df.corr('spearman'))), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
#            square=True, cmap="YlGnBu")
#ax.set_title('二维数组热力图', fontsize = 18)
#ax.set_ylabel('features', fontsize = 18)
#ax.set_xlabel('pytho', fontsize = 18)
#————————————————
#版权声明：本文为CSDN博主「elibneh」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/henbile/article/details/80241597



#sns.set()
#sns.set_style('whitegrid', {'font.sans-serif':['simhei','Arial']})

#df2 = pd.DataFrame(np.random.rand(5, 17),columns = ['BIL','ALB','CA199','CEA','最大径','肿瘤边界', '包膜完整或不完整', '包膜','动脉期瘤周强化', '瘤内坏死','肿瘤破裂','静脉癌栓','肝硬化','瘤内动脉','瘤周低密度环','瘤与肝脏边界的低密度影','RIV'], index = ['GPC3', 'Ki67-14', 'CD34','ARG1', 'cK19'])
#print(df1)

#plt.figure(figsize=(20,8))
#sns.heatmap(data=df1, annot = True, fmn = '.2f', cmap = 'coolwarm')

plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文
print(df1.shape)


plt.figure(dpi=120)
sns.heatmap(data=df1, columns=['TBIL（nmol/L)', 'ALB（g/l)', 'CA199(ng/ml)（U/ml）', 'CEA(ng/ml)0-5', '最大径', '肿瘤边界', '包膜完整或不完整', '包膜', '动脉期瘤周强化', '瘤内坏死', '肿瘤破裂', '静脉癌栓', '肝硬化', '瘤内动脉', '瘤与肝脏边界的低密度影', 'RIV'], index=['GPC3', 'Ki67-14', 'CD34', 'ARG1', 'cK19'], annot=True, fmn='.2f', cmap='coolwarm')