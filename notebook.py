__author__ = 'jerry.ban'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import funcs.notebook_func as df_proc
import funcs.view_data as df_view
import math
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
ols_pred = ols_model.predict(df_X_test) # make the predictions by the model
ols_model.summary()  # R2= 0.094

output_test = df_Y_test_o.copy()
output_test["date"] = df_X_test_o.date
output_test.rename(columns={"cnt": "actual"}, inplace = True)
output_test["predicted"] = ols_pred
output_test_group = output_test.groupby("date").sum()

output_train = df_Y_train_o.copy()
output_train["date"] = df_X_train_o.date
output_train.rename(columns={"cnt": "actual"}, inplace = True)
output_train["predicted"] = ols_model.predict(df_X_train)
output_train_group = output_train.groupby("date").sum()

R2_train = 1 - sum((output_train_group.actual - output_train_group.predicted)**2) / sum((output_train_group.actual - output_train_group.actual.mean())**2)
print("R2 @OLS: %s" % R2_train)

plt.figure()
ax1 = output_test_group.actual.plot(color = "red",  grid = True, label = "True")
ax2 = output_test_group.predicted.plot(color = "blue", grid = True, label = "Predict")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1, l1, loc=2) #plt.legend(h1+h1, l1+l2, loc=2)
plt.show()

# ML linear regression result:
from sklearn import linear_model, model_selection
sk_lmr = linear_model.LinearRegression()
sk_lmr.fit(df_X_train, df_Y_train.cnt)
scores = model_selection.cross_val_score(sk_lmr, df_X_train, df_Y_train.cnt, cv=5)

sk_lmr.fit(df_X_train, df_Y_train.cnt)
Y_pred = sk_lmr.predict(df_X_test)
sk_lmr.score(df_X_test, df_Y_test.cnt)


# ML lasso result:



