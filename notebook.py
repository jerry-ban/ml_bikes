__author__ = 'jerry.ban'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from funcs.rd_forest import  *
from funcs.xgb import  *
import funcs.notebook_func as df_proc
import funcs.view_data as df_view
import math
import time
from datetime import datetime as dt

ML_ONLY = False
ML_ONLY = True
if not ML_ONLY:
    df_cnt = df_view.visualize_data()
else:
    df_cnt = df_proc.get_processed_df()

col_cate = ["zip_code", "start_station_id", "subscription_type", "events","date", "month", "weekday",  "cnt", "Fog-Rain", "Rain","Rain-Thunderstorm", "Subscriber"] + [x for x in list(df_cnt.columns) if x.find("__",0)>=0]
cols_to_be_scaled = [x for x in df_cnt.columns if x not in col_cate]

date_list = sorted(df_cnt["date"].unique())
cut_off_date = date_list[math.floor(len(date_list) *(1-0.25))] # "2015-03-01"
df_data_train = df_cnt[df_cnt["date"] < cut_off_date]
df_data_test = df_cnt[df_cnt["date"] >= cut_off_date]

cols_to_drop = ["zip_code", "start_station_id", "subscription_type", "events", "month", "weekday"]
df_X_train_o, df_Y_train_o = df_data_train.drop(cols_to_drop + ["cnt"], axis =1), df_data_train[["cnt"]]
df_X_test_o, df_Y_test_o   =  df_data_test.drop(cols_to_drop + ["cnt"], axis =1), df_data_test[["cnt"]]

stdc_x = StandardScaler()
df_X_train=df_X_train_o.drop(["date"], axis=1)
df_X_train.loc[:, cols_to_be_scaled] = stdc_x.fit_transform(df_X_train_o[cols_to_be_scaled])
df_Y_train = df_Y_train_o.copy()
df_X_test=df_X_test_o.drop(["date"], axis=1)
df_X_test.loc[:, cols_to_be_scaled] = stdc_x.fit_transform(df_X_test_o[cols_to_be_scaled])
df_Y_test = df_Y_test_o.copy()

df_X_train["const"] = 1.0
df_X_test["const"] = 1.0

###Y transform?

#X_train_o, X_test_o, Y_train_o, Y_test_o =df_X_train_o.values, df_X_test_o.values, df_Y_train_o.values, df_Y_test_o.values

## regression
import statsmodels.api as sm
ols_model = sm.OLS(df_Y_train.cnt,  df_X_train).fit()
y_pred_train = ols_model.predict(df_X_train) # make the predictions by the model
y_pred_test = ols_model.predict(df_X_test)
ols_model.summary()  # R2= 0.094

output_train = df_Y_train_o.copy()
output_train["date"] = df_X_train_o.date
output_train.rename(columns={"cnt": "actual"}, inplace = True)

output_test = df_Y_test_o.copy()
output_test["date"] = df_X_test_o.date
output_test.rename(columns={"cnt": "actual"}, inplace = True)

#result00 = run_rf(output_train, df_X_train, df_Y_train, output_test, df_X_test, df_Y_test)

r2_train= r2_score(df_Y_train.cnt,y_pred_train)
r2_test = r2_score(df_Y_test.cnt ,y_pred_test )
result_comp=df_proc.compare_results("OLS", ols_model, output_train, df_X_train, output_test, df_X_test)
result01={"r2": [r2_train, r2_test], "R2":[result_comp[1], result_comp[2]], "plot": result_comp[0] }


# ML linear regression result:
from sklearn import linear_model, model_selection
# sk_lmr = linear_model.LinearRegression()

#ML Ridge result
from sklearn.linear_model import Ridge
opt_pars = {"score": None, "alpha": None}
for this_alpha in [0, 0.01, .03, 0.1, 0.3, 1, 3, 10, 13, 20, 50, 100, 1000]:
    ridge_lm = Ridge(alpha = this_alpha).fit(df_X_train, df_Y_train.cnt)
    r2_train = ridge_lm.score(df_X_train, df_Y_train.cnt)
    r2_test = ridge_lm.score(df_X_test, df_Y_test.cnt)
    num_coeff_bigger = np.sum(abs(ridge_lm.coef_) > 1.0)
    if opt_pars["alpha"] is None or opt_pars["score"] < r2_train:
        opt_pars["alpha"] = this_alpha
        opt_pars["score"] = r2_train
    #print('Ridge: Alpha = {:.4f}\nnum abs(coeff) > 1.0: {}, r2 train: {:.4f}, r2 test: {:.4f}\n'.format(this_alpha, num_coeff_bigger, r2_train, r2_test))
ridge_lm_opt = Ridge(alpha = opt_pars["alpha"]).fit(df_X_train, df_Y_train.cnt)
r2_train = ridge_lm_opt.score(df_X_train, df_Y_train.cnt)
r2_test = ridge_lm_opt.score(df_X_test, df_Y_test.cnt)
num_coeff_bigger = np.sum(abs(ridge_lm_opt.coef_) > 1.0)
print('Ridge: Alpha = {:.4f}\nnum abs(coeff) > 1.0: {}, r2 train: {:.4f}, r2 test: {:.4f}'.format(opt_pars["alpha"], num_coeff_bigger, r2_train, r2_test))
result_comp=df_proc.compare_results("Ridge", ridge_lm_opt, output_train, df_X_train, output_test, df_X_test)
result02={"r2": [r2_train, r2_test], "R2":[result_comp[1], result_comp[2]], "plot": result_comp[0] }


# ML lasso result:
from sklearn.linear_model import Lasso
opt_pars = {"score": None, "alpha": None}
for this_alpha in [0.01, 0.3, 0.1, 0.3, 1, 2, 3, 5, 10, 20, 50, 100]:
    lasso_lm = Lasso(this_alpha , max_iter = 1000).fit(df_X_train, df_Y_train.cnt)
    r2_train = lasso_lm.score(df_X_train, df_Y_train.cnt)
    r2_test = lasso_lm.score(df_X_test, df_Y_test.cnt)
    if opt_pars["alpha"] is None or opt_pars["score"] < r2_train:
        opt_pars["alpha"] = this_alpha
        opt_pars["score"] = r2_train
    #print("Lasso: Alpha = {:.4f}\nFeatures kept: {}, r2 train: {:.4f}, r2 test: {:.4f}\n".format(this_alpha, np.sum(lasso_lm.coef_ != 0), r2_train, r2_test))
lasso_lm_opt = Ridge(alpha = opt_pars["alpha"]).fit(df_X_train, df_Y_train.cnt)
r2_train = lasso_lm_opt.score(df_X_train, df_Y_train.cnt)
r2_test = lasso_lm_opt.score(df_X_test, df_Y_test.cnt)
print("Lasso: Alpha = {:.4f}\nFeatures kept: {}, r2 train: {:.4f}, r2 test: {:.4f}".format(opt_pars["alpha"], np.sum(lasso_lm.coef_ != 0), r2_train, r2_test))
result_comp=df_proc.compare_results("Lasso", lasso_lm_opt, output_train, df_X_train, output_test, df_X_test)
result03={"r2": [r2_train, r2_test], "R2":[result_comp[1], result_comp[2]], "plot": result_comp[0] }

result04 = run_rf(output_train, df_X_train, df_Y_train, output_test, df_X_test, df_Y_test)
result05 = run_xgb(output_train, df_X_train, df_Y_train, output_test, df_X_test, df_Y_test)

plt.show()
time.sleep(5)
print("work done...!")
#
# # ML RandomForest result:
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# rf_estimator = RandomForestRegressor(random_state=1)
# opt_pars = {"score": None, "alpha": None}
# param_grid = {
#             "n_estimators"      : [5,10,30],
#             "max_features"      : ["auto", "log2"],
#             "bootstrap": [True, False],
#             }
# rf_grid = GridSearchCV(rf_estimator, param_grid)
# rf_grid.fit(df_X_train, df_Y_train.cnt)
# r2_train = rf_grid.best_score_
# opt_pars = rf_grid.best_params_
# rf_opt = RandomForestRegressor(random_state=1).set_params(**opt_pars)
# r2_train = rf_opt.score(df_X_train, df_Y_train.cnt)
# r2_test = rf_opt.score(df_X_test, df_Y_test.cnt)
# df_proc.compare_results("RandomForest", rf_opt, output_train, df_X_train, output_test, df_X_test)
#
#
#
#
