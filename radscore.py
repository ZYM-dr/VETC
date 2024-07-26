import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

alphas = np.logspace(-10, 1, 100, base = 10)

# Create lasso regression with three alpha values
regr_cv = LassoCV(alphas=alphas, cv = 5, max_iter = 1e4,selection = 'random',random_state=42)
#regr_cv = LassoCV(alphas=alphas, cv = 5, max_iter = 1e6)

# Fit the linear regression
reg_scaler = RobustScaler()
# reg_scaler = MinMaxScaler()

regr_cv.fit(reg_scaler.fit_transform(X), y)
# regr_cv.fit(X, y)

not_zero = []
for coe in regr_cv.coef_:
    if coe !=0:
        not_zero.append(coe)

len(not_zero)

not_zero

# X_train.columns[regr_cv.coef_ != 0].tolist()
X_train.columns[regr_cv.coef_ != 0].tolist()

# not_zero
# X_train.columns[regr_cv.coef_ != 0].tolist()
coef_table = {'features':X_train.columns[regr_cv.coef_ != 0].tolist(),'coefficient':not_zero}
coef_table

coef_pd = pd.DataFrame.from_dict(coef_table)
coef_pd

coef_pd[coef_pd['features'] == 'wavelet-HLH_glszm_GrayLevelVariance']['coefficient'].values[0]

formula_res = 'signature = '
for col in coef_pd['features'].values.tolist():
    formula_res = formula_res+str(col)+"*"+ str(coef_pd[coef_pd['features'] == col]['coefficient'].values[0])
formula_res