__author__ = 'jerry.ban'

import numpy as np
import pandas as pd
from sklearn import preprocessing
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV
import funcs.notebook_func as df_proc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def run_xgb(output_train, df_X_train, df_Y_train, output_test, df_X_test, df_Y_test):
    xgb_estimator = XGBRegressor()
    param_grid = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [30]}

    opt_pars = {"score": None, "alpha": None}
    xgb_grid = GridSearchCV(xgb_estimator, param_grid)
    xgb_grid.fit(df_X_train, df_Y_train.cnt)
    r2_train = xgb_grid.best_score_
    opt_pars = xgb_grid.best_params_
    # n_estimators = 30,max_features='log2',bootstrap=True,  max_depth=None
    xgb_opt = XGBRegressor(random_state=1).set_params(**opt_pars)
    xgb_opt.fit(df_X_train, df_Y_train.cnt)
    r2_train = xgb_opt.score(df_X_train, df_Y_train.cnt)
    r2_test = xgb_opt.score(df_X_test, df_Y_test.cnt)
    result = df_proc.compare_results("XGBoost", xgb_opt, output_train, df_X_train, output_test, df_X_test)
    return {"r2": [r2_train, r2_test], "R2":[result[1], result[2]], "plot": result[0] }