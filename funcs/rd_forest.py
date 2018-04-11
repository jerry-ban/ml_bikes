__author__ = 'jerry.ban'

import funcs.notebook_func as df_proc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def run_rf(output_train, df_X_train, df_Y_train, output_test, df_X_test, df_Y_test):
    rf_estimator = RandomForestRegressor(random_state=1,n_jobs=-1)
    opt_pars = {"score": None, "alpha": None}
    param_grid = {
                "n_estimators"      : [5,10,30],
                "max_features"      : ["auto", "log2"],
                "bootstrap": [True, False],
                "max_depth": [4,5]
                }
    rf_grid = GridSearchCV(rf_estimator, param_grid)
    rf_grid.fit(df_X_train, df_Y_train.cnt)
    r2_train = rf_grid.best_score_
    opt_pars = rf_grid.best_params_
    # n_estimators = 30,max_features='log2',bootstrap=True,  max_depth=None
    rf_opt = RandomForestRegressor(random_state=1).set_params(**opt_pars)
    rf_opt.fit(df_X_train, df_Y_train.cnt)
    r2_train = rf_opt.score(df_X_train, df_Y_train.cnt)
    r2_test = rf_opt.score(df_X_test, df_Y_test.cnt)
    result = df_proc.compare_results("RandomForest", rf_opt, output_train, df_X_train, output_test, df_X_test)
    return {"r2": [r2_train, r2_test], "R2":[result[1], result[2]], "plot": result[0] }
