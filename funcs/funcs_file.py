__author__ = 'jerry.ban'


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
    df["date"] = pd.to_datetime(df["date"])
    return df

def process_df_date_col(df):

    return df