__author__ = 'jerry.ban'

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd
from datetime import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt

def get_processed_df():
    df= pd.read_csv("df_processed.gz",  compression='gzip', sep="\t", header=0, converters={'date': lambda d: datetime.strptime(d, "%Y-%m-%d")})
     # index_col=0,
    return df

def compare_results(algorithm_label, algorithm_model, Output_train, X_train, Output_test, X_test):
    col_train_pred = ".".join([algorithm_label, "train", "predicted"])
    Output_train[col_train_pred] = algorithm_model.predict(X_train)
    Output_train_group = Output_train.groupby("date").sum()

    col_test_pred = ".".join([algorithm_label, "test", "predicted"])
    Output_test[col_test_pred] = algorithm_model.predict(X_test)
    Output_test_group = Output_test.groupby("date").sum()

    R2_train = 1 - sum((Output_train_group.actual - Output_train_group[col_train_pred])**2) / sum((Output_train_group.actual - Output_train_group.actual.mean())**2)
    R2_test = 1 - sum((Output_test_group.actual - Output_test_group[col_test_pred])**2) / sum((Output_test_group.actual - Output_test_group.actual.mean())**2)
    print("{} - R2 scores: [{:.4f}, {:.4f}]".format(algorithm_label, R2_train, R2_test) )
    print("\n")

    # plt.figure()
    # ax1 = Output_test_group.actual.plot(color = "red",  grid = True, label = "True")
    # ax2 = Output_test_group[col_test_pred].plot(color = "blue", grid = True, label = "Predict")
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # plt.legend(h1, l1, loc=2) #plt.legend(h1+h1, l1+l2, loc=2)
    # plt.title("Test Dataset")
    # plt.show()

    fig=plt.figure()
    fig.suptitle("Algorithm: {}".format(algorithm_label))

    plt.subplot(121) #ax1 = fig.add_subplot(111)
    ax1 = Output_train_group.actual.plot(color = "red",  grid = True, label = "True")
    ax1 = Output_train_group[col_train_pred].plot(color = "blue", grid = True, label = "Predict")
    h1, l1 = ax1.get_legend_handles_labels()
    plt.legend(h1, l1, loc=2) #plt.legend(h1+h1, l1+l2, loc=2)
    plt.title("Train Dataset, R2={:.4f}]".format(R2_train))

    ax2=plt.subplot(122)
    Output_test_group.actual.plot(color = "red",  grid = True, label = "True")
    Output_test_group[col_test_pred].plot(color = "blue", grid = True, label = "Predict")
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h2, l2, loc=2) #plt.legend(h1+h1, l1+l2, loc=2)
    ax2.set_title("Test Dataset, R2={:.4f}]".format(R2_test))

    plt.show()
    return (fig, R2_train, R2_test)

def normalize_data(df_cnt):
    df = pd.DataFrame()
    return df
def calc_r2(y, y_pred):
    r2 = 0.0

    return r2