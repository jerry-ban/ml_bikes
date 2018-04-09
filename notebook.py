__author__ = 'jerry.ban'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

trip = pd.read_csv("trip.csv")
weather = pd.read_csv("weather.csv")
station = pd.read_csv("station.csv")

def getDate(df):
    for d in df:
        yield d.date()

def getHour(df):
    for d in df:
        yield d.hour

def getWeekday(df):
    for d in df:
        yield d.weekday()

def getMonth(df):
    for d in df:
        yield d.month

station.index = station.id
station = station.drop('id', axis=1)
station.head()

station.plot.barh(x="name", y="dock_count", figsize=(10,30))

trip.index = trip.id
trip=trip.drop("id", axis=1)
dates=pd.to_datetime(trip.start_date, format="%m/%d/%Y %H:%M")
trip["datetime"]=dates
trip["date"]= list(getDate(dates))
trip["month"] = list(getMonth(dates))
trip["hour"] = list(getHour(dates))
trip["weekday"]= list(getWeekday(dates))
trip["zip_code"] = trip.zip_code.astype(str)

trip = trip[trip.duration<=60*60]

print(trip.shape)

weather.index = pd.to_datetime(weather.date)
weather['date'] = list(getDate(weather.index))
weather['weekday'] = list(getWeekday(weather.index))

weather.precipitation_inches = weather.precipitation_inches.replace("T", 0.01)
weather.precipitation_inches = weather.precipitation_inches.astype(float)

weather.zip_code = weather.zip_code.astype(str)
# change events to categorical variables
tmp_weather = pd.get_dummies(weather.events, drop_first=True)
tmp_weather['Rain'] = tmp_weather.Rain + tmp_weather.rain
weather = pd.concat([tmp_weather, weather],axis=1)
weather = weather.drop('rain',axis=1)

df = weather.merge(trip.drop('weekday', axis=1), on=['date', 'zip_code'], how='left')
df.head()

df = df[df.duration.notnull()]

#normalize data
df_dur = df.drop(['date', 'events', 'zip_code', 'start_date', 'start_station_name', 'start_station_id', 'weekday',
                  'end_date', 'end_station_name', 'end_station_id', 'bike_id', 'subscription_type', 'datetime', 'month', 'hour'], axis=1)
stdc_dur = StandardScaler()
tmp_df_dur = df_dur.drop(['Fog-Rain', 'Rain', 'Rain-Thunderstorm'], axis=1)
tmp_df_dur = tmp_df_dur.fillna(method='ffill')
df_dur_std = pd.DataFrame(stdc_dur.fit_transform(tmp_df_dur), columns=tmp_df_dur.columns, index = df_dur.index)
for c in tmp_df_dur.columns:
    df_dur[c] = df_dur_std[c]
df_weekday = pd.get_dummies(df.weekday, drop_first=True, prefix='weekday_')
df_month = pd.get_dummies(df.month, drop_first=True, prefix='month_')
df_hour = pd.get_dummies(df.hour, drop_first=True, prefix='hour_')
df_subscription_type = pd.get_dummies(df.subscription_type, drop_first=True)

df_dur = pd.concat([df_dur, df_weekday, df_month, df_hour, df_subscription_type], axis=1)
df_dur.index = df.date
df_dur.head()

df_dur_train = df_dur.iloc[:91153,:]
df_dur_test = df_dur.iloc[91153:,:]

###charts
plt.figure()
df.weekday.hist(bins=7)

plt.figure()
df.month.hist(bins=12)

plt.figure()
df.hour.hist(bins=24)

dur = df.duration / 60
plt.figure()
dur.plot.hist(bins=60)

plt.figure()
df.plot.scatter('mean_temperature_f', 'duration', title="mean_temp vs duration")

df.events.value_counts().plot.bar()

tmp_df = df[['hour', 'duration']].dropna()
tmp_df['hour'] = tmp_df['hour'].astype(int)
sns.boxplot(x='hour', y='duration', data=tmp_df).set_title("counts vs hour time")

freq_trip = trip.groupby(['date','zip_code', 'start_station_id']).size()
freq_trip.rename('cnt', inplace=True)
freq_trip = freq_trip.to_frame().reset_index()
df2 = weather.merge(freq_trip, on=['date', 'zip_code'], how='left')

df2.start_station_id = df2.start_station_id.fillna(0)
df2.start_station_id = df2.start_station_id.astype(int)

df2.cnt = df2.cnt.fillna(0)

df2['month'] = list(getMonth(df2['date']))

df2.head()
df2.shape

# remove categorical variables
df_cnt = df2.drop(['date', 'events', 'zip_code', 'start_station_id', 'weekday', 'month'], axis=1)

tmp_df_cnt = df_cnt.drop(['Fog-Rain', 'Rain', 'Rain-Thunderstorm'], axis=1)
tmp_df_cnt = tmp_df_cnt.fillna(method='ffill')

cols_to_be_scaled = tmp_df_cnt.columns.drop("cnt")
# stdc_cnt = StandardScaler()
# df_cnt_std = pd.DataFrame(stdc_cnt.fit_transform(tmp_df_cnt), columns=tmp_df_cnt.columns, index = df_cnt.index)
#
# for c in tmp_df_cnt.columns:
#     df_cnt[c] = df_cnt_std[c]

# get dummy variables from categorical variables, except zip
df_weekday = pd.get_dummies(df2.weekday, drop_first=True, prefix='weekday_')
df_month = pd.get_dummies(df2.month, drop_first=True, prefix='month_')
df_start_station_id = pd.get_dummies(df2.start_station_id, drop_first=True, prefix='start_station_id_')

df_cnt = pd.concat([df_cnt, df_weekday, df_month, df_start_station_id], axis=1)
df_cnt.index = df2.date
df_cnt.head()
df_cnt.loc[:, cols_to_be_scaled]=df_cnt[cols_to_be_scaled].fillna(method='ffill')

#ML data processing
df_X, df_Y = df_cnt.drop('cnt', axis=1), df_cnt[['cnt']]
df_X_train_o, df_X_test_o, df_Y_train_o, df_Y_test_o = train_test_split(df_X, df_Y, test_size=0.3)
X_train_o, X_test_o, Y_train_o, Y_test_o =df_X_train_o.values, df_X_test_o.values, df_Y_train_o.values, df_Y_test_o.values

stdc_x = StandardScaler()
df_X_train=df_X_train_o.copy()
df_X_train.loc[:, cols_to_be_scaled] = stdc_x.fit_transform(df_X_train_o[cols_to_be_scaled])
df_X_test=df_X_test_o.copy()
df_X_test.loc[:, cols_to_be_scaled] = stdc_x.fit_transform(df_X_test_o[cols_to_be_scaled])
X_train = df_X_train.values
X_test=df_X_test.values
Y_train = df_Y_train_o.cnt.values
Y_test = df_Y_test_o.cnt.values

#
# for c in tmp_df_cnt.columns:
#     df_cnt[c] = df_cnt_std[c]
## can skip the followin chart section, directly go to regression parts
df_cnt_train = pd.DataFrame(X_train, columns= df_cnt.columns.drop('cnt'))
df_cnt_train['cnt'] = Y_train

df_cnt_test = pd.DataFrame(X_test, columns= df_cnt.columns.drop('cnt'))
df_cnt_test['cnt'] = Y_test


df_cnt_train = df_cnt.iloc[:32531,:]
df_cnt_test = df_cnt.iloc[32531:,:]

plt.figure()
df2.cnt.hist(bins=30)

tmp_df2 = df2.drop(['date', 'events', 'zip_code'], axis=1).fillna(method='ffill')
tmp_df2.head()

plt.figure()
sns.heatmap(df_cnt.corr())

## regression
import statsmodels.api as sm
X_ols = sm.add_constant(X_train)
ols_model = sm.OLS(Y_train, X_ols).fit()
ols_pred = ols_model.predict(sm.add_constant(X_test)) # make the predictions by the model
ols_model.summary()

plt.figure()
output_chart = df_Y_test_o.copy()
output_chart.rename({"cnt": "actual"}, inplace = True)
output_chart["Predicted"] = pd.Series(data = ols_pred)
ax1 = output_chart.actual.groupby(level = 0).plot(color = "red", grid = True, lable = "True")
ax2 = output_chart.predicted.groupby(level = 0).plot(color = "blue", grid = True, lable = "Predict")


# OLM result:
from sklearn import linear_model, model_selection
sk_lmr = linear_model.LinearRegression()
sk_lmr.fit(X_train, Y_train)
scores = model_selection.cross_val_score(sk_lmr, X_train, Y_train, cv=5)

sk_lmr.fit(X_train, Y_train)
sk_lmr.score(X_test, Y_test)
Y_pred = sk_lmr.predict(X_test)




