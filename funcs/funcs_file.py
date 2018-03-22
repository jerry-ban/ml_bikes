__author__ = 'jerry.ban'

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd
import numpy as np
import sys

def date_col_parser(x):
    return pd.datetime.strptime(x)

def get_df_from_files(file_list, n_rows =[None, None, None]):
    file_trip = file_list[0]
    file_station = file_list[1]
    file_weather = file_list[2]
    # if n_rows[0] is None:
    #     trip = pd.read_csv(file_trip, header=0, parse_dates=["start_date", "end_date"], chunksize=10000)
    # else:
    #     trip = pd.read_csv(file_trip, nrows = n_rows[0], header=0, parse_dates=["start_date", "end_date"],chunksize=10000)
    if n_rows[0] is None:
        trip = pd.read_csv(file_trip, header=0)
    else:
        trip = pd.read_csv(file_trip, nrows = n_rows[0], header=0)

    # sys.getsizeof(df)/10**6
    # df.memory_usage(deep=True).sum() / 10**6
    if n_rows[1] is None:
        station = pd.read_csv(file_station )
    else:
        station = pd.read_csv(file_station, nrows = n_rows[1])

    if n_rows[1] is None:
        weather = pd.read_csv(file_weather)
    else:
        weather = pd.read_csv(file_weather, nrows = n_rows[2])

    return trip, station, weather

def process_df_date_col_trip(df, stage =1):
    df["start_date"] = pd.to_datetime(df["start_date"],format="%m/%d/%Y %H:%M")
    #df["end_date"]   = pd.to_datetime(df["end_date"],format="%m/%d/%Y %H:%M").dt.date
    df["end_date"]   = pd.to_datetime(df["end_date"],format="%m/%d/%Y %H:%M")
    df["date"]   = pd.DatetimeIndex(df["start_date"]).normalize()# df["start_date"].apply(lambda x: x.date())
    df["date_num"] = (df["date"] - df["date"].min()).astype("timedelta64[D]")

    df["start_month"] = df["start_date"].dt.month
    #df["start_wday"] = df["start_date"].dt.weekday #TBD will convert to labels
    df["start_wday"] = df["start_date"].dt.weekday_name
    df["start_hour"] = df["start_date"].dt.hour
    df["end_month"] = df["end_date"].dt.month
    df["end_wday"] = df["end_date"].dt.weekday_name #TBD will convert to labels
    df["end_hour"] = df["end_date"].dt.hour
    df["duration"] = df["duration"] / 60

    #TBD will convert to labels
    df["is_weekend"] = df["start_wday"].apply( lambda x: "weekday" if x  in ["Sunday", "Saturday"] else "weekend")
    #TBD will convert to labels
    df["is_weekend_v2"] = df["end_wday"].apply( lambda x: "weekday" if x  in ["Sunday", "Saturday"] else "weekend")


    df["end_date"] =  pd.DatetimeIndex(df["end_date"]).normalize()
    return df

def process_df_date_col_station(df, stage =1):
    df["installation_date"] = pd.to_datetime(df["installation_date"],format="%m/%d/%Y")

    return df

def process_df_date_col_weather(df, stage =1):
    df["date"] = pd.to_datetime(df["date"],format="%m/%d/%Y")
    df["events"].fillna("Normal", inplace=True) #set missing value as normal
    df['events'].replace("rain", "Rain", inplace=True)

    df.dropna(axis=1, how='all', inplace=True) # drop columns if all values are 0, means no information useful from that column

    df_median = df.groupby("date").agg("median") # auto set index as date
    ds_median_all = df.median().to_dict()
    # cols_to_drop = ds_median_all[ds_median_all.isnull()].index.tolist()
    # df_median.drop(cols_to_drop, axis=1, inplace=True)  # drop columns with all na
    for col in df_median.columns:
        df_median.loc[df_median[col].isnull(), col] = ds_median_all[col]
    #df_median.loc[ df_median.isnull()] = pd.DataFrame(data =ds_median_all, index = [0])

    df['precipitation_inches'].replace("T", np.nan, inplace=True)
    df['precipitation_inches'] = df['precipitation_inches'].astype(float)

    df= pd.merge(df, df.groupby('date')['precipitation_inches'].median().reset_index(name="precipitation_median"),
                             how = "inner", on = "date")
    df.loc[df['precipitation_inches'].isnull(),"precipitation_inches"] = df["precipitation_median"]
    df['precipitation_inches'].fillna(0, inplace=True)

    df = pd.merge(df, df.groupby('date')['max_gust_speed_mph'].median().reset_index(name="gust_median"),how="inner", on="date")
    df.loc[df['max_gust_speed_mph'].isnull(),"max_gust_speed_mph"] = df["gust_median"]
    df['max_gust_speed_mph'].fillna(0, inplace=True)

    df.loc[df['gust_median'].isnull(), "gust_median"] = ds_median_all.get("max_gust_speed_mph", 0)
    # df_median.isnull().sum()  # to check if someday has null for all zipcodes


    df_median.drop("zip_code",axis=1, inplace=True)

    df.set_index("date", inplace=True)
    df.update(df_median, join="left", overwrite=True, filter_func=pd.isnull) # assign

    df.reset_index(inplace=True)
    df["month"] = df['date'].dt.month
    df["wday"] = df['date'].dt.weekday_name
    df["year"] = df['date'].dt.year
    df["day"] = df['date'].dt.day
    cal = calendar()
    holidays = cal.holidays(start=df["date"].min(), end=df["date"].max(), return_name=True)
    df['isholiday'] = df['date'].isin(holidays.index)

    #weather = weather %>% group_by(max_wind_Speed_mph) %>% mutate(gust_median = median(max_gust_speed_mph, na.rm=T))
    return df

