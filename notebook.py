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

print(trip.head())

weather.index = pd.to_datetime(weather.date)
weather['date'] = list(getDate(weather.index))
weather['weekday'] = list(getWeekday(weather.index))

weather.precipitation_inches = weather.precipitation_inches.replace("T", 0.01)
weather.precipitation_inches = weather.precipitation_inches.astype(float)

weather.zip_code = weather.zip_code.astype(str)

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

plt.figure()
df.weekday.hist(bins=7)

plt.figure()
df.month.hist(bins=12)
