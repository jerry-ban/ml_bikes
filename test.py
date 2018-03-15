__author__ = 'jerry.ban'

import pandas as pd
import numpy as np
import funcs.funcs_file as iofiles
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

file_trip = "trip.csv"
file_station = "station.csv"
file_weather = "weather.csv"
file_list = [file_trip, file_station, file_weather]
# rows: 669959,70, 3665
Trip, Station,Weather = iofiles.get_df_from_files(file_list, n_rows =[10**5,None, None])
#Trip, Station,Weather = iofiles.get_df_from_files(file_list)
Trip = iofiles.process_df_date_col_trip(Trip, stage =1)
Station = iofiles.process_df_date_col_station(Station, stage =1)
Weather = iofiles.process_df_date_col_weather(Weather, stage =1)

Trip_count = Trip.groupby(["date", "date_num"]).size().reset_index(name="count")
#Trip_count["date_num"]= (Trip_count["date"] - Trip_count["date"].min()).astype("timedelta64[D]")
Trip_count.describe()

logging.info("Insight investigating")

#fig, ax = plt.subplots()
logging.info("Time series of trip count")
sns.lmplot('date_num', 'count', data= Trip_count)
ax = plt.gca().xaxis
ax.set_ticklabels(Trip_count["date"].dt.strftime("%Y-%m-%d"))

logging.info("Time series of trip count based on weekend/weekdate")
#sns.regplot(Trip_count["date"].apply(lambda x: str(x)), Trip_count["count"])
isweekend_date = Trip.groupby(["date", "date_num", "is_weekend"]).size().reset_index(name = "count")
#isweekend_date["date_num"]= (isweekend_date["date"] - isweekend_date["date"].min()).astype("timedelta64[D]")
#fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.lmplot('date_num', 'count', hue = "is_weekend", data= isweekend_date, fit_reg= False)  # row="sex", col="time" then it's chart matrix
sns.lmplot('date_num', 'count', col = "is_weekend", data= isweekend_date)  # row="sex", col="time" then it's chart matrix

logging.info("time-diff and week-diff")
Trip_hour_wday = Trip.groupby(["start_wday", "start_hour"]).size().reset_index(name ="count")
chart_df = Trip_hour_wday[["start_wday", "start_hour", "count"]].pivot("start_wday", "start_hour", "count")
plt.figure()
chart_ax = sns.heatmap(chart_df, cmap="YlGnBu")
#
# s_min = Trip_hour_wday["count"].min()
# s_max = Trip_hour_wday["count"].max()
# Trip_hour_wday["count_gray"]= Trip_hour_wday["count"].apply(lambda x: 255 - round(255.0*(x-s_min)/(s_max-s_min)))
# plt.figure()
# plt.gray()
# new_chart= Trip_hour_wday[["start_wday", "start_hour", "count_gray"]].pivot(values="count_gray", columns="start_hour", index = "start_wday")
# plt.imshow(new_chart)
#f, ax = plt.subplots(figsize=(6, 15))
df_chart= Trip.groupby(["date", "is_weekend", "subscription_type"]).size().reset_index(name="count")
plt.figure()
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

pal = ["#9b59b6", "#e74c3c"]
dfc1= df_chart[df_chart["is_weekend"]=="weekday"].pivot("date", "subscription_type", "count")
dfc1.plot.area(ax = ax1, title ="weekday", color=pal)
dfc2= df_chart[df_chart["is_weekend"]=="weekend"].pivot("date", "subscription_type", "count")
dfc2.plot.area(ax=ax2, title ="weekend", color=pal)


Station_trip = pd.merge(Trip, Station,how = "inner", left_on="start_station_id", right_on="id", )
#Station_trip_count.groupby(["date", "subscription_type", "city" ])["count"].unstack("city")  #pivot with multiple index

logging.info("city wise charts")
Station_trip_count = Station_trip.groupby(["date", "subscription_type", "city" ]).size().reset_index(name="count")
g = sns.FacetGrid(Station_trip_count, col='city', hue='subscription_type', col_wrap=4, sharey=False, )
g = g.map(plt.plot, 'date', 'count')
g = g.map(plt.fill_between, 'date', 'count', alpha=0.2).set_titles("{col_name} city")
g = g.set_titles("{col_name}")
plt.subplots_adjust(top=0.92)
g.add_legend()

logging.info("city wise charts")