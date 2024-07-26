import numpy as np
import pandas as pd
import miceforest as mf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
"""
#多重插补填充

#注意：
#1. 仅支持 csv ex
#2. 文件的首行是标题
#sns.set(font ='SimHei',font_scale=1.5)解决seaborn中文显示问题
#plt.rcParams['font.sans-serif']=['Arial Unicode MS']
#plt.rcParams['axes.unicode_minus'] = False
"""

#file_path = 'data/exp.xlsx'
#save_path = 'data/exp_imputing_mean.xlsx'
df = pd.read_excel('QST-23-11-14.xlsx', sheet_name='validation')
df =df.astype(float)
df.dtypes
#data_missing0 =data_missing0.astype(string)
#df['Y','X1','X2','X3','X4','X5','X6','X7'] = pd.factorize(df['Y','X1','X2','X3','X4','X5','X6','X7'])[0]
df = df.apply(pd.to_numeric)
df.dtypes
a = df.isnull().sum()#查看缺失值
print(a)

kernel = mf.ImputationKernel(
  df,
  datasets=4,
  save_all_iterations=True,
  save_models=1,
  random_state=10
)
kernel.mice(iteration=3, n_jobs=-1)
print(kernel)
complete_data = kernel.complete_data(0)
complete_data.to_excel('QST-23-11-23-validationCB.xlsx')