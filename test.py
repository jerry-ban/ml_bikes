__author__ = 'jerry.ban'

import pandas as pd
import numpy as np
import funcs.funcs_file as iofiles
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine_examples as pe

from mpl_toolkits.basemap import Basemap
from matplotlib import cm



#import gmplot

from sklearn.preprocessing import MinMaxScaler
zip_filters = ["94107","94063","94301","94041","95113"]

file_trip = "trip.csv"
file_station = "station.csv"
file_weather = "weather.csv"
file_list = [file_trip, file_station, file_weather]
# rows: 669959,70, 3665
Trip, Station,Weather = iofiles.get_df_from_files(file_list, n_rows =[None,None, None])
#Trip, Station,Weather = iofiles.get_df_from_files(file_list)
Trip = iofiles.process_df_date_col_trip(Trip, stage =1)
#Trip = Trip[Trip["zip_code"].isin(zip_filters)]
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


Station_trip = pd.merge(Trip, Station,how = "inner", left_on="start_station_id", right_on="id", suffixes = ["",".y"])
#Station_trip_count.groupby(["date", "subscription_type", "city" ])["count"].unstack("city")  #pivot with multiple index

logging.info("city wise charts")
Station_trip_count = Station_trip.groupby(["date", "subscription_type", "city" ]).size().reset_index(name="count")
g = sns.FacetGrid(Station_trip_count, col='city', hue='subscription_type', col_wrap=3, sharey=False, )
g = g.map(plt.plot, 'date', 'count')
g = g.map(plt.fill_between, 'date', 'count', alpha=0.2).set_titles("{col_name} city")
g = g.set_titles("{col_name}")
plt.subplots_adjust(top=0.92)
g.add_legend()

logging.info("city-weekday charts")
Station_wday_count = Station_trip.groupby(["start_wday", "city" ]).size().reset_index(name="count")
wday_map = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday": 4, "Friday":5, "Saturday":6, "Sunday": 7}
Station_wday_count["wday"] = Station_wday_count["start_wday" ].map(wday_map)
city_map = {'Mountain View':1, 'Palo Alto':2, 'Redwood City':3, 'San Francisco':4, 'San Jose':5}
Station_wday_count["city_num"] = Station_wday_count["city" ].map(city_map)
plt.figure()
ax = plt.gca()
Station_wday_count.plot.scatter(ax=ax, x="wday", y="city_num", s=Station_wday_count["count"]/12, title = "city vs weekday" )
ax.xaxis.set_ticks(np.arange(1, 7.5, 1))
ax.set_xticklabels(sorted([x for x in wday_map.keys()], key= lambda x: wday_map[x]))
ax.yaxis.set_ticks(np.arange(1, 5.5, 1))
#ax.set_yticklabels(Station_wday_count["city"])
ax.set_yticklabels(sorted([x for x in city_map.keys()], key= lambda x: city_map[x]))

logging.info("show trip in geographic map(geomap), EBD....")
Trip_map = Station_trip.groupby(["long", "lat", "zip_code"]).size().reset_index(name="count")
#
# lb_long, lb_lat, rt_long, rt_lat = -122.4990, 37.31072, -121.7800, 37.88100
# lon_c = float(lb_long + rt_long) / 2.0
# lat_c = float(lb_lat + rt_lat) / 2.0
#
#
# fig = plt.figure()
# ax = plt.gca()
# m = Basemap(resolution='i', lon_0=lon_c, lat_0=lat_c, llcrnrlon=lb_long,llcrnrlat=lb_lat,urcrnrlon= rt_long,urcrnrlat=rt_lat)
# x, y = m(Trip_map['long'].values, Trip_map['lat'].values)
# m.scatter(x, y,markersize=Trip_map['count'].values)
# m.plot(x,y, )
# m.drawcounties()
# m.drawstates()
# m.drawcounties()
# m.fillcontinents()


Station_trip.rename(columns={ "lat": "start_lat", "long": "start_long"}, inplace=True)

End_station_trip = pd.merge(Trip, Station, how = "inner", left_on= "end_station_id", right_on ="id", suffixes=["", ".y"])
End_station_trip.rename(columns={"lat": "end_lat", "long": "end_long"}, inplace=True)

Station_trip_sf = Station_trip[Station_trip["city"]=="San Francisco"]
Road_df = pd.merge(Station_trip_sf[["id","start_station_id", "start_lat", "start_long","city"]], End_station_trip[["id","end_station_id","end_lat", "end_long","city"]]
                   , how="inner", on = "id", suffixes=[".x",".y"] )

logging.info(" show route map, tbd...")
Road_df_count= Road_df.groupby(["start_lat", "start_long", "end_lat", "end_long"]).size().reset_index(name="count")

#Road_df.plot.scatter(x="start_long", y="start_lat",
    #s=Road_df['count']/10#, label="population",  c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, alpha=0.4 , figsize=(10,7),                )
#plt.show()

logging.info("weather data processing")
#fill in median value for all columns' missing


Weather.plot(x="date", y="max_sea_level_pressure_inches")
#zip_city_match = Station_trip.groupby(["zip_code", "city"]).size().reset_index(name="count")
#zip_city_match = zip_city_match[zip_city_match["zip_code"].isin(zip_filters)]

#Weather["city"] = Weather["zip_code"].map(dict(zip(zip_city_match["zip_code"], zip_city_match["city"])))
#Weather.dropna(subset=["city"], inplace = True)

logging.info("weather data processing")
Station_name= Station[["name", "city"]].copy()
Station_name.clumns=["start_station_name","city"]

Trip_num= pd.merge(Trip, Station, how = "inner", left_on = "start_station_name", right_on="name")
#Trip_num = Trip_num.groupby(["date", "zip_code", "start_hour"]).size().reset_index(name="count")
Trip_num = Trip_num.groupby(["date", "zip_code", "start_month", "start_day", "start_wday",  "start_hour", "is_weekend" ]).size().reset_index(name="count")

df_station_weather = pd.merge(Trip_num, Weather, how = "inner", left_on =["date","zip_code"], right_on =["date", "zip_code"])
df_v2 = df_station_weather.drop(['month', 'wday','day'], axis =1)
df_v2[["count"]].plot.hist()
plt.figure()
df_v2["start_month"].plot.hist(bins=12)
df_v2["is_weekend"] = df_v2["is_weekend"].map({"weekend": 1, "weekday": 0})
#df_v2.drop("zip_code", axis = 1,  inplace=True) # only keep cities, since we only choose 5 zipcode and one for each city

if df_v2.isnull().values.sum() >0:
    raise Exception("NaN values exist! processing data correctly, please!")

#############
logging.info("preparing data for Machine Learning Training...")
df_v2_dummies = pd.get_dummies(df_v2, columns = ["year", "start_month", "start_day",  "start_wday", "start_hour", "zip_code","events"])

train_source = df_v2_dummies[df_v2_dummies["date"] < "2015-03-31"]
len(train_source)*1.0/len(df_v2)
train_source.shape

test_source =  df_v2_dummies[df_v2_dummies["date"] >= "2015-03-31"]

X_train = train_source.drop(["date", "count"], axis =1)
y_train =train_source["count"]
train_date = train_source["date"]
X_test = test_source.drop(["date", "count"], axis =1)
y_test =test_source["count"]
test_date=test_source["date"]
##########*** OLM model ***##########
import statsmodels.api as sm
olm = sm.OLS(y_train, X_train).fit()
olm = sm.OLS(y_train, X_train.astype(float)).fit()
y_hat = olm.predict(X_train)
ax = plt.gca()
ax.plot()

from sklearn import linear_model
l_m = linear_model.LinearRegression()
l_m.fit(X_train, y_train)
l_m.score(X_test, y_test)
score_list ={"OLM": l_m.score(X_test, y_test) }

sns.lmplot('date_num', 'count', hue = "is_weekend", data= isweekend_date, fit_reg= False)  # row="sex", col="time" then it's chart matrix


from sklearn.model_selection import train_test_split

train_data = df_v2
#df_v2$day <- as.factor(df_v2$day)