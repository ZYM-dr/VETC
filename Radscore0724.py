import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

def LassoScreening(X, y):
    """
    return: 特征重要性
    """
X = preprocessing.StandardScaler().fit_transform(X)
data = pd.read_excel('ki67radiomic.xlsx', sheet_name='ap')
X = data[data.columns[1:]]

y = data['Ki67']
print(X.shape)

#LASSO method
alphas = np.logspace(-10,-1,100, base=10)
print(alphas)

selector_lasso = LassoCV(alphas = alphas, cv = 5, max_iter = 3.242e+00)
print(selector_lasso.fit(X, y))


print(selector_lasso.alpha_)
print(selector_lasso.coef_)
print(X.columns[selector_lasso.coef_ != 0])
print(X[X.columns[selector_lasso.coef_ != 0]])
print('score')
print(selector_lasso.intercept_)
print(selector_lasso.mse_path_.shape)
print(selector_lasso.mse_path_.mean(axis = 1))
print(selector_lasso.alphas_)