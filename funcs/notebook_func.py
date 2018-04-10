__author__ = 'jerry.ban'

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd
from datetime import datetime
import numpy as np
import sys

def get_processed_df():
    df= pd.read_csv("df_processed.gz",  compression='gzip', sep="\t", header=0, converters={'date': lambda d: datetime.strptime(d, "%Y-%m-%d")})
     # index_col=0,
    return df

def normalize_data(df_cnt):
    df = pd.DataFrame()
    return df
def calc_r2(y, y_pred):
    r2 = 0.0

    return r2